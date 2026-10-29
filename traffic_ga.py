"""Traffic signal timing optimisation with a Genetic Algorithm (core code).

The notebook Traffic_Signal_GA.ipynb is built from the "# %% [name]" sections of this file
(see make_notebook.py), so the notebook and this module always contain the same code.
"""

# %% [setup]
import math
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Intersection settings -------------------------------------------------------------
DIRECTIONS = ("N", "S", "E", "W")  # chromosome order and signal phase order
SAT_FLOW_VPH = 1800                # saturation flow: 1800 veh/h = one vehicle every 2 s of green
AMBER_S = 3                        # amber after every green (no departures)
ALL_RED_S = 2                      # all-red after every amber (no departures)
LOST_TIME_S = len(DIRECTIONS) * (AMBER_S + ALL_RED_S)  # 20 s lost per cycle

# ---- Timing constraints ----------------------------------------------------------------
G_MIN, G_MAX = 10, 60    # each green time (s)
C_MIN, C_MAX = 40, 180   # whole cycle = sum(greens) + LOST_TIME_S (s)

# ---- Simulation timeline (s) -------------------------------------------------------------
WARMUP_S = 300     # roads fill up; vehicles arriving now are NOT measured
MEASURE_S = 3600   # vehicles arriving in this 1-hour window are measured
DRAIN_S = 600      # extra time so measured vehicles can finish crossing

# ---- Random seeds (separate seeds for optimisation and for final testing) ---------------
TRAIN_SEEDS = [0, 1, 2]                # the GA / Random Search only ever see these
TEST_SEEDS = list(range(1000, 1020))   # 20 unseen traffic patterns for the final comparison
RUN_SEEDS = [0, 1, 2]                  # 3 independent GA / Random Search runs

# ---- GA defaults ---------------------------------------------------------------------------
GA_PARAMS = dict(pop_size=30, generations=40, p_crossover=0.8, p_mutation=0.2,
                 mutation_sigma=5.0, tournament_k=3, n_elites=2)


def validate_constraints(g_min=G_MIN, g_max=G_MAX, c_min=C_MIN, c_max=C_MAX, lost=LOST_TIME_S):
    """Return the allowed range (s_lo, s_hi) for the SUM of the four greens.

    Each green must be in [g_min, g_max] and the cycle in [c_min, c_max], so the green sum must be
    in [max(4*g_min, c_min - lost), min(4*g_max, c_max - lost)]. If that range is empty, no
    valid signal plan exists and we stop immediately with an error.
    """
    if not (0 < g_min <= g_max):
        raise ValueError(f"need 0 < g_min <= g_max, got {g_min}, {g_max}")
    s_lo = max(4 * g_min, c_min - lost)
    s_hi = min(4 * g_max, c_max - lost)
    if s_lo > s_hi:
        raise ValueError(f"impossible constraints: green sum must be in [{s_lo}, {s_hi}]")
    return s_lo, s_hi


S_LO, S_HI = validate_constraints()  # 40 and 160 with the defaults
print(f"Each green in [{G_MIN}, {G_MAX}] s; sum of greens in [{S_LO}, {S_HI}] s "
      f"(cycle {S_LO + LOST_TIME_S}-{S_HI + LOST_TIME_S} s); lost time {LOST_TIME_S} s per cycle")


# %% [scenarios]
# Arrival rates in vehicles per hour, in the order N, S, E, W.
SCENARIOS = {
    "Light":    {"rates": (150, 150, 150, 150), "description": "low traffic on all four roads"},
    "Moderate": {"rates": (300, 300, 250, 250), "description": "medium traffic, N-S slightly busier"},
    "Uneven":   {"rates": (550, 200, 150, 150), "description": "North road much busier than the others"},
}


def flow_ratio(rates):
    """Y = sum(arrival rate / saturation flow). A fixed signal can only keep up if Y < 1 - L/C."""
    return sum(r / SAT_FLOW_VPH for r in rates)


scenario_table = pd.DataFrame(
    [{"scenario": name, **dict(zip(DIRECTIONS, s["rates"])), "total_veh_per_h": sum(s["rates"]),
      "Y": round(flow_ratio(s["rates"]), 3), "description": s["description"]}
     for name, s in SCENARIOS.items()]
)
print(f"Stability limit for the longest allowed cycle: Y < {1 - LOST_TIME_S / C_MAX:.3f}")
scenario_table


# %% [traffic]
def generate_arrivals(rates, seed, total_s=WARMUP_S + MEASURE_S + DRAIN_S):
    """Synthetic traffic: number of vehicles arriving on each road in every second.

    Arrivals are Poisson with mean rate/3600 per second. The same seed always gives the same
    traffic, so every controller can be tested on exactly the same vehicles.
    Returns an int array of shape (total_s, 4).
    """
    if len(rates) != 4 or any(r < 0 for r in rates):
        raise ValueError("rates must be four non-negative numbers (veh/h)")
    rng = np.random.default_rng(seed)
    per_second = np.asarray(rates, dtype=float) / 3600.0
    return rng.poisson(per_second, size=(total_s, 4))


def make_arrival_sets(rates, seeds, **kwargs):
    return [generate_arrivals(rates, s, **kwargs) for s in seeds]


# %% [constraints]
def is_valid(greens):
    """True if a plan satisfies every constraint: 4 integers, each in [G_MIN, G_MAX], sum in [S_LO, S_HI]."""
    return (len(greens) == 4
            and all(float(g).is_integer() and G_MIN <= g <= G_MAX for g in greens)
            and S_LO <= sum(greens) <= S_HI)


def _shift(values, room, amount, sign):
    """Move `amount` seconds out of (sign=-1) or into (sign=+1) the genes, proportional to `room`.

    Leftover single seconds go to the genes with the largest remainders (ties: lowest index),
    so the result is exact, deterministic, and never exceeds any gene's room.
    """
    total_room = sum(room)
    base = [amount * r // total_room for r in room]
    remainders = [amount * r % total_room for r in room]
    leftover = amount - sum(base)
    for i in sorted(range(4), key=lambda i: (-remainders[i], i))[:leftover]:
        base[i] += 1
    return [v + sign * b for v, b in zip(values, base)]


def repair(greens):
    """Turn ANY four numbers into a valid plan (constraint handling).

    1. Round each gene to whole seconds and clip it to [G_MIN, G_MAX].
    2. If the sum is above S_HI, remove the excess in proportion to each gene's slack (g - G_MIN).
       If it is below S_LO, add the deficit in proportion to each gene's headroom (G_MAX - g).
    This always succeeds: total slack >= excess because S_HI >= 4*G_MIN, and total headroom >=
    deficit because S_LO <= 4*G_MAX (guaranteed by validate_constraints).
    """
    if len(greens) != 4:
        raise ValueError("a plan needs exactly four green times")
    g = [min(max(math.floor(float(x) + 0.5), G_MIN), G_MAX) for x in greens]
    total = sum(g)
    if total > S_HI:
        g = _shift(g, [x - G_MIN for x in g], total - S_HI, sign=-1)
    elif total < S_LO:
        g = _shift(g, [G_MAX - x for x in g], S_LO - total, sign=+1)
    plan = tuple(int(x) for x in g)
    assert is_valid(plan), (greens, plan)
    return plan


def cycle_length(greens):
    return sum(greens) + LOST_TIME_S


def green_schedule(greens):
    """For each second of one cycle, which roads have green: list of [bool]*4.

    Phase order N -> S -> E -> W; each green is followed by amber and all-red (nobody moves).
    """
    schedule = []
    for i, g in enumerate(greens):
        for _ in range(g):
            schedule.append([j == i for j in range(4)])
        schedule.extend([[False] * 4] * (AMBER_S + ALL_RED_S))
    return schedule


# %% [simulator]
def simulate(arrivals, greens, warmup_s=WARMUP_S, measure_s=MEASURE_S, drain_s=DRAIN_S):
    """Simulate one fixed-time signal plan on one traffic pattern, second by second.

    Every second, for each road: (1) arriving vehicles join the back of the queue;
    (2) if the road has green it earns 0.5 'departure credit' (1800 veh/h) and every full credit
    lets the front vehicle leave; (3) the queue length is recorded.

    Measured vehicles are those ARRIVING inside the measurement window. Each one's waiting time
    is departure second - arrival second, even if it leaves during the drain period. A measured
    vehicle still waiting at the very end is 'censored' and counted with the time waited so far.
    """
    if not is_valid(greens):
        raise ValueError(f"invalid plan {greens}; use repair() first")
    total_s = warmup_s + measure_s + drain_s
    if arrivals.shape != (total_s, 4):
        raise ValueError(f"arrivals must have shape ({total_s}, 4), got {arrivals.shape}")
    sat_per_s = SAT_FLOW_VPH / 3600.0
    w_start, w_end = warmup_s, warmup_s + measure_s
    schedule = green_schedule(greens)
    cycle = len(schedule)
    arr = arrivals.tolist()

    queues = [deque() for _ in range(4)]  # each entry = arrival second of a waiting vehicle
    credit = [0.0] * 4
    waits = [[] for _ in range(4)]        # waiting times of measured vehicles, per road
    arrived, departed = [0] * 4, [0] * 4
    served_in_window = 0
    queue_sum = 0

    for t in range(total_s):
        green_now = schedule[t % cycle]
        in_window = w_start <= t < w_end
        for i in range(4):
            q = queues[i]
            n_new = arr[t][i]
            if n_new:
                arrived[i] += n_new
                q.extend([t] * n_new)
            if green_now[i]:
                c = credit[i] + sat_per_s
                while c >= 1.0 and q:
                    t_arr = q.popleft()
                    c -= 1.0
                    departed[i] += 1
                    if in_window:
                        served_in_window += 1
                    if w_start <= t_arr < w_end:
                        waits[i].append(t - t_arr)
                # an empty queue cannot 'bank' more than one vehicle's worth of credit
                credit[i] = c if q else min(c, 1.0)
            else:
                credit[i] = 0.0
            if in_window:
                queue_sum += len(q)

    n_censored = 0
    for i in range(4):
        for t_arr in queues[i]:
            if w_start <= t_arr < w_end:
                waits[i].append(total_s - t_arr)
                n_censored += 1

    all_waits = [w for road in waits for w in road]
    n = len(all_waits)
    result = {
        "avg_wait": sum(all_waits) / n if n else 0.0,  # MAIN METRIC (s per vehicle)
        "vehicles_measured": n,
        "n_censored": n_censored,
        "vehicles_served": served_in_window,           # departures during the 1-hour window
        "avg_queue": queue_sum / measure_s,            # vehicles waiting (all roads), time average
        "max_wait": max(all_waits) if n else 0,
    }
    for d, road in zip(DIRECTIONS, waits):
        result[f"avg_wait_{d}"] = sum(road) / len(road) if road else 0.0
    result["arrived"] = arrived
    result["departed"] = departed
    result["final_queue"] = [len(q) for q in queues]
    return result


# %% [fitness]
class CachedFitness:
    """Wraps an objective plan -> float (lower is better) and remembers every plan it has scored.

    unique_evals counts DIFFERENT plans actually simulated; it is the fair 'budget' used to
    compare the GA with Random Search. A cache hit returns the same value without simulating.
    """

    def __init__(self, objective):
        self.objective = objective
        self.cache = {}
        self.calls = 0

    @property
    def unique_evals(self):
        return len(self.cache)

    def __call__(self, greens):
        plan = tuple(int(g) for g in greens)
        assert is_valid(plan), f"invalid plan reached the fitness function: {plan}"
        self.calls += 1
        if plan not in self.cache:
            self.cache[plan] = float(self.objective(plan))
        return self.cache[plan]


def traffic_fitness(train_sets):
    """Fitness of a plan = average waiting time (s/veh) averaged over the TRAINING traffic patterns."""
    def objective(plan):
        return float(np.mean([simulate(a, plan)["avg_wait"] for a in train_sets]))
    return CachedFitness(objective)


# %% [controllers]
def fixed_equal_plan():
    """Conventional baseline: the same green for every road (25 s each, cycle 120 s)."""
    return repair([25, 25, 25, 25])


def random_plan(rng):
    """A random valid plan: four uniform values in [G_MIN, G_MAX], then repaired."""
    return repair(rng.uniform(G_MIN, G_MAX, size=4))


# %% [ga]
def tournament_select(population, fitnesses, k, rng):
    """Pick k different individuals at random; the one with the LOWEST waiting time wins."""
    contenders = rng.choice(len(population), size=k, replace=False)
    winner = min(contenders, key=lambda i: fitnesses[i])
    return population[winner]


def uniform_crossover(parent1, parent2, rng):
    """Each gene of child 1 comes from parent 1 or parent 2 with probability 1/2 (child 2 gets the other)."""
    mask = rng.random(4) < 0.5
    child1 = [a if m else b for a, b, m in zip(parent1, parent2, mask)]
    child2 = [b if m else a for a, b, m in zip(parent1, parent2, mask)]
    return repair(child1), repair(child2)


def mutate(individual, p_mutation, sigma, rng):
    """Each gene, with probability p_mutation, gets a random change ~ Normal(0, sigma) seconds."""
    genes = [g + rng.normal(0.0, sigma) if rng.random() < p_mutation else g for g in individual]
    return repair(genes)


def run_ga(fitness, pop_size=30, generations=40, p_crossover=0.8, p_mutation=0.2,
           mutation_sigma=5.0, tournament_k=3, n_elites=2, seed=0, verbose=False):
    """Genetic Algorithm that MINIMISES fitness (average waiting time).

    Returns the best plan, its fitness and a per-generation history table.
    """
    if not (2 <= tournament_k <= pop_size and 0 <= n_elites < pop_size):
        raise ValueError("need 2 <= tournament_k <= pop_size and 0 <= n_elites < pop_size")
    rng = np.random.default_rng(seed)
    population = [random_plan(rng) for _ in range(pop_size)]
    best_plan, best_fit = None, math.inf
    history = []

    for gen in range(generations + 1):  # generation 0 is the random initial population
        fitnesses = [fitness(ind) for ind in population]
        i_best = int(np.argmin(fitnesses))
        if fitnesses[i_best] < best_fit:
            best_plan, best_fit = population[i_best], fitnesses[i_best]
        history.append({"generation": gen, "best": fitnesses[i_best], "mean": float(np.mean(fitnesses)),
                        "worst": max(fitnesses), "best_so_far": best_fit,
                        "unique_evals": getattr(fitness, "unique_evals", None)})
        if verbose:
            print(f"gen {gen:3d}  best {fitnesses[i_best]:8.2f}  mean {np.mean(fitnesses):8.2f}")
        if gen == generations:
            break

        # elitism: the best n_elites individuals survive unchanged
        order = np.argsort(fitnesses)
        next_pop = [population[i] for i in order[:n_elites]]
        while len(next_pop) < pop_size:
            p1 = tournament_select(population, fitnesses, tournament_k, rng)
            p2 = tournament_select(population, fitnesses, tournament_k, rng)
            if rng.random() < p_crossover:
                c1, c2 = uniform_crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2
            next_pop.append(mutate(c1, p_mutation, mutation_sigma, rng))
            next_pop.append(mutate(c2, p_mutation, mutation_sigma, rng))
        population = next_pop[:pop_size]

    return {"best_plan": best_plan, "best_fitness": best_fit, "history": pd.DataFrame(history)}


# %% [random_search]
def run_random_search(fitness, budget, seed=0, max_tries=100_000):
    """Evaluate `budget` DIFFERENT random valid plans and keep the best one."""
    rng = np.random.default_rng(seed)
    best_plan, best_fit = None, math.inf
    tries = 0
    while fitness.unique_evals < budget and tries < max_tries:
        tries += 1
        plan = random_plan(rng)
        if plan in fitness.cache:
            continue  # already tried: draw again so every evaluation is a new plan
        f = fitness(plan)
        if f < best_fit:
            best_plan, best_fit = plan, f
    return {"best_plan": best_plan, "best_fitness": best_fit}


# %% [evaluation]
def evaluate_on_test(plan, test_sets, test_seeds):
    """Simulate a plan on every unseen test traffic pattern; one row per test seed."""
    rows = []
    for seed, arrivals in zip(test_seeds, test_sets):
        r = simulate(arrivals, plan)
        rows.append({"test_seed": seed, **{k: v for k, v in r.items()
                                           if k not in ("arrived", "departed", "final_queue")}})
    return pd.DataFrame(rows)


def plan_columns(plan):
    return {f"g_{d}": g for d, g in zip(DIRECTIONS, plan)} | {"cycle": cycle_length(plan)}


def run_comparison(scenarios=SCENARIOS, test_seeds=TEST_SEEDS, run_seeds=RUN_SEEDS,
                   ga_params=GA_PARAMS, verbose=True):
    """Experiment 1 + 2: Fixed-Equal vs Random Search vs GA on every scenario.

    For each scenario: GA and Random Search are optimised on the TRAIN seeds (3 runs each,
    Random Search gets the same number of unique evaluations as the GA run with the same seed),
    then every plan is evaluated on the same unseen TEST seeds.
    Returns (test_results, runs, ga_histories) as DataFrames.
    """
    test_rows, runs, histories = [], [], []
    for name, sc in scenarios.items():
        t0 = time.perf_counter()
        train_sets = make_arrival_sets(sc["rates"], TRAIN_SEEDS)
        test_sets = make_arrival_sets(sc["rates"], test_seeds)

        def record(controller, run, plan, train_fit, n_evals):
            res = evaluate_on_test(plan, test_sets, test_seeds)
            test_rows.append(res.assign(scenario=name, controller=controller, run=run, **plan_columns(plan)))
            runs.append({"scenario": name, "controller": controller, "run": run, **plan_columns(plan),
                         "train_fitness": train_fit, "test_avg_wait": res["avg_wait"].mean(),
                         "unique_evals": n_evals})

        fixed = fixed_equal_plan()
        record("Fixed-Equal", 0, fixed, traffic_fitness(train_sets)(fixed), 0)
        for run in run_seeds:
            ga_fit = traffic_fitness(train_sets)
            ga = run_ga(ga_fit, seed=run, **ga_params)
            histories.append(ga["history"].assign(scenario=name, run=run))
            rs_fit = traffic_fitness(train_sets)
            rs = run_random_search(rs_fit, budget=ga_fit.unique_evals, seed=run)
            record("Random Search", run, rs["best_plan"], rs["best_fitness"], rs_fit.unique_evals)
            record("GA", run, ga["best_plan"], ga["best_fitness"], ga_fit.unique_evals)
            if verbose:
                print(f"  {name:8s} run {run}: GA {ga['best_plan']} train {ga['best_fitness']:.2f} | "
                      f"RS {rs['best_plan']} train {rs['best_fitness']:.2f} | "
                      f"{ga_fit.unique_evals} unique evaluations each")
        if verbose:
            print(f"{name}: done in {time.perf_counter() - t0:.1f} s")
    return pd.concat(test_rows, ignore_index=True), pd.DataFrame(runs), pd.concat(histories, ignore_index=True)


CONTROLLERS = ["Fixed-Equal", "Random Search", "GA"]


def summarize(test_results):
    """Results table: mean and std over the test seeds (GA / RS runs are averaged per seed first)."""
    metrics = ["avg_wait", "vehicles_served", "avg_queue", "n_censored"]
    per_seed = test_results.groupby(["scenario", "controller", "test_seed"])[metrics].mean().reset_index()
    table = per_seed.groupby(["scenario", "controller"])[metrics].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table = table.reset_index()
    fixed = table[table.controller == "Fixed-Equal"].set_index("scenario")["avg_wait_mean"]
    table["improvement_vs_fixed_%"] = [100 * (fixed[s] - w) / fixed[s]
                                       for s, w in zip(table.scenario, table.avg_wait_mean)]
    order = {c: i for i, c in enumerate(CONTROLLERS)}
    scen_order = {s: i for i, s in enumerate(SCENARIOS)}
    table = table.sort_values(by=["scenario", "controller"],
                              key=lambda col: col.map(scen_order if col.name == "scenario" else order))
    return table.reset_index(drop=True)


def run_mutation_experiment(scenario="Uneven", p_values=(0.05, 0.2, 0.5), test_seeds=TEST_SEEDS,
                            run_seeds=RUN_SEEDS, ga_params=GA_PARAMS, verbose=True):
    """Optional Experiment 3: effect of the mutation probability on the GA (one scenario)."""
    rates = SCENARIOS[scenario]["rates"]
    train_sets = make_arrival_sets(rates, TRAIN_SEEDS)
    test_sets = make_arrival_sets(rates, test_seeds)
    rows, histories = [], []
    for p in p_values:
        for run in run_seeds:
            fit = traffic_fitness(train_sets)
            ga = run_ga(fit, seed=run, **{**ga_params, "p_mutation": p})
            test_wait = evaluate_on_test(ga["best_plan"], test_sets, test_seeds)["avg_wait"].mean()
            rows.append({"p_mutation": p, "run": run, **plan_columns(ga["best_plan"]),
                         "train_fitness": ga["best_fitness"], "test_avg_wait": test_wait,
                         "unique_evals": fit.unique_evals})
            histories.append(ga["history"].assign(p_mutation=p, run=run))
            if verbose:
                print(f"  p_mutation={p}: run {run} -> {ga['best_plan']} train {ga['best_fitness']:.2f} "
                      f"test {test_wait:.2f} ({fit.unique_evals} unique evaluations)")
    return pd.DataFrame(rows), pd.concat(histories, ignore_index=True)


# %% [plots]
COLORS = {"Fixed-Equal": "#8c8c8c", "Random Search": "#dd8452", "GA": "#4c72b0"}


def log_axis(ax):
    """Log-scale y axis with plain-number tick labels (e.g. 40, 60, 100 instead of 4x10^1)."""
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 4, 5, 6, 8)))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_wait_bars(summary, path=None):
    """Bar chart of average waiting time per scenario and controller (error bars = std over test seeds)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    scen = list(SCENARIOS)
    x = np.arange(len(scen))
    width = 0.26
    for k, ctrl in enumerate(CONTROLLERS):
        s = summary[summary.controller == ctrl].set_index("scenario").reindex(scen)
        bars = ax.bar(x + (k - 1) * width, s["avg_wait_mean"], width, yerr=s["avg_wait_std"], capsize=3,
                      label=ctrl, color=COLORS[ctrl])
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x, scen)
    log_axis(ax)
    ax.set_ylabel("Average waiting time (s/vehicle, log scale)")
    ax.set_xlabel("Traffic scenario")
    ax.set_title("Experiment 1: average waiting time on unseen test traffic\n(mean ± std over test seeds)")
    ax.legend(frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_convergence(histories, runs, path=None):
    """Best and mean fitness per generation (mean over GA runs) with Fixed-Equal and Random Search levels."""
    scen = list(SCENARIOS)
    fig, axes = plt.subplots(1, len(scen), figsize=(5 * len(scen), 4))
    for ax, name in zip(np.atleast_1d(axes), scen):
        h = histories[histories.scenario == name]
        g = h.groupby("generation")
        best_mean = g["best_so_far"].mean()
        ax.plot(best_mean.index, best_mean, color=COLORS["GA"], lw=2.5, label="GA best (mean of runs)")
        ax.fill_between(best_mean.index, g["best_so_far"].min(), g["best_so_far"].max(),
                        color=COLORS["GA"], alpha=0.2, label="GA best (range over runs)")
        ax.plot(best_mean.index, g["mean"].mean(), color="#4c72b0", ls=":", label="GA population mean")
        r = runs[runs.scenario == name]
        fixed = r[r.controller == "Fixed-Equal"]["train_fitness"].iloc[0]
        rs = r[r.controller == "Random Search"]["train_fitness"].mean()
        ax.axhline(fixed, color=COLORS["Fixed-Equal"], ls="--", label=f"Fixed-Equal ({fixed:.1f} s)")
        ax.axhline(rs, color=COLORS["Random Search"], ls="-.", label=f"Random Search, same budget ({rs:.1f} s)")
        log_axis(ax)
        ax.set_title(name)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness = avg waiting time on train traffic (s/veh)")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Experiment 2: GA convergence (lower is better)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_green_times(runs, path=None):
    """Green time of each road: Fixed-Equal vs the GA's best-on-train plan, per scenario."""
    scen = list(SCENARIOS)
    fig, axes = plt.subplots(1, len(scen), figsize=(4.2 * len(scen), 3.6), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), scen):
        r = runs[runs.scenario == name]
        fixed = r[r.controller == "Fixed-Equal"].iloc[0]
        ga = r[r.controller == "GA"].sort_values("train_fitness").iloc[0]
        x = np.arange(4)
        for off, row, ctrl in ((-0.2, fixed, "Fixed-Equal"), (0.2, ga, "GA")):
            vals = [row[f"g_{d}"] for d in DIRECTIONS]
            bars = ax.bar(x + off, vals, 0.4, color=COLORS[ctrl], label=f"{ctrl} (cycle {int(row['cycle'])} s)")
            ax.bar_label(bars, fontsize=8)
        ax.set_xticks(x, DIRECTIONS)
        ax.set_title(f"{name}  {SCENARIOS[name]['rates']} veh/h")
        ax.set_xlabel("Road")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    np.atleast_1d(axes)[0].set_ylabel("Green time (s)")
    np.atleast_1d(axes)[0].set_ylim(0, G_MAX + 15)
    fig.suptitle("Optimised green-light durations (GA best run) vs Fixed-Equal")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_mutation(mut_results, mut_histories, path=None):
    """Experiment 3: convergence and test waiting time for each mutation probability."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for p, h in mut_histories.groupby("p_mutation"):
        best = h.groupby("generation")["best_so_far"].mean()
        ax1.plot(best.index, best, label=f"p_mutation = {p}")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best fitness, mean of runs (s/veh)")
    log_axis(ax1)
    ax1.set_title("Convergence")
    ax1.legend(frameon=False)
    agg = mut_results.groupby("p_mutation")[["train_fitness", "test_avg_wait"]].agg(["mean", "std"])
    x = np.arange(len(agg))
    for col, marker, label in (("train_fitness", "o", "train traffic"), ("test_avg_wait", "s", "test traffic")):
        ax2.errorbar(x, agg[(col, "mean")], yerr=agg[(col, "std")], marker=marker, capsize=4, label=label)
    ax2.set_xticks(x, [str(p) for p in agg.index])
    ax2.set_xlabel("Mutation probability per gene")
    ax2.set_ylabel("Average waiting time (s/veh)")
    ax2.set_title("Final result (mean ± std over runs)")
    ax2.legend(frameon=False)
    fig.suptitle("Experiment 3: effect of mutation probability (Uneven scenario)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
