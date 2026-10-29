"""Developer tool: build Traffic_Signal_GA.ipynb from the sections of traffic_ga.py.

Each "# %% [name]" section of traffic_ga.py becomes one code cell, placed between the markdown
explanations below, so the notebook and the module always contain identical code.

    python make_notebook.py
    jupyter nbconvert --to notebook --execute --inplace Traffic_Signal_GA.ipynb   # embed outputs
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
COLAB_URL = ("https://colab.research.google.com/github/MohitSingh351/traffic-signal-ga/"
             "blob/main/Traffic_Signal_GA.ipynb")


def module_sections():
    text = (ROOT / "traffic_ga.py").read_text()
    parts = re.split(r"^# %% \[(\w+)\]\n", text, flags=re.M)
    return {name: body.strip("\n") for name, body in zip(parts[1::2], parts[2::2])}


def md(text):
    return ("markdown", text.strip("\n"))


def code(text):
    return ("code", text.strip("\n"))


def section(name):
    return ("section", name)


CELLS = [
    md(f"""
# 🚦 Traffic Signal Timing Optimization Using Genetic Algorithms
### A Simulation-Based Comparative Study (Soft Computing project)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

**Goal.** Choose the green-light duration of each road at a four-way intersection so that vehicles
wait as little as possible. A **Genetic Algorithm (GA)**, written from scratch with NumPy, searches
for the best green times. Each candidate is scored by running a **traffic simulation** on synthetic
traffic. The GA is compared with a conventional **Fixed-Equal** signal and with **Random Search**
given the same evaluation budget.

**How to run.** In Colab, choose *Runtime → Run all*. Nothing needs to be installed: the notebook
uses only NumPy, Pandas and Matplotlib, which Colab already provides. A full run takes about
2–3 minutes. Set `QUICK = True` in Section 12 for a faster demo run.

> **Scope.** All traffic is **synthetic** (randomly generated in Python), and the simulator is
> deliberately simple. This is not real traffic data, a real deployment or a real-time controller.
> The GA finds one good *fixed-time* plan for a given traffic pattern, offline.

**Contents**
1. Settings and constraints
2. Traffic scenarios
3. Synthetic traffic generation
4. Constraint handling (repair)
5. Traffic simulator
6. Fitness function
7. Controllers (Fixed-Equal, random plans)
8. Genetic Algorithm
9. Random Search
10. Self-checks
11. Experiment and plotting functions
12. Experiment 1: controller comparison
13. Experiment 2: GA convergence
14. Optimised green times
15. Experiment 3 (optional): mutation probability
16. Observations, limitations and conclusion
"""),
    md("""
## 1. Settings and constraints

**The intersection**
- **Phases.** There are four roads, N, S, E and W. Exactly **one road has green at a time**, in the
  fixed order N → S → E → W, and then the cycle repeats.
- **Lost time.** Every green is followed by **3 s amber** and **2 s all-red**, when no vehicle moves.
  That gives `L = 4 × 5 = 20 s` of lost time per cycle.
- **Saturation flow.** While its light is green, a queue discharges at **1800 vehicles/hour**, which is
  one vehicle every 2 s.

**Decision variables (the chromosome).** The four green times `[g_N, g_S, g_E, g_W]`, in whole seconds.

**Constraints**
- Each green: `10 ≤ g ≤ 60` s. Below 10 s is unsafe; above 60 s makes the other roads wait too long.
- Cycle length: `40 ≤ C = g_N + g_S + g_E + g_W + 20 ≤ 180` s. With 10 s minimum greens, the
  shortest cycle that can actually occur is 4·10 + 20 = 60 s.
- Together these mean the **sum of the greens** must lie in `[max(4·10, 40−20), min(4·60, 180−20)] = [40, 160]`.
  `validate_constraints()` computes this range and raises an error if the settings make it empty,
  i.e. if no valid plan could exist.

**Simulation timeline.** 300 s warm-up, then a 3600 s (1-hour) measurement window, then 600 s of drain.

**Seeds.** The GA and Random Search optimise on **train seeds 0–2** only. Every controller is
then evaluated on **20 unseen test seeds (1000–1019)**. This keeps the GA from simply memorising
the traffic it was trained on.
"""),
    section("setup"),
    md("""
## 2. Traffic scenarios

Arrival rates are given in vehicles per hour, in the order N, S, E, W.

**Y = Σ (arrival rate ÷ saturation flow)** measures how "full" the intersection is. A fixed-time
signal can only keep up if `Y < 1 − L/C`. With the longest allowed cycle (180 s), that means **Y < 0.889**.
All three scenarios are below this limit, so a well-chosen plan can serve them.
"""),
    section("scenarios"),
    md("""
## 3. Synthetic traffic generation

In every second, the number of vehicles arriving on each road is drawn from a **Poisson
distribution** with mean `rate / 3600`. Poisson arrivals are the standard model for random,
independent vehicle arrivals.

Every traffic pattern is created from a **seed**, so the same seed always gives exactly the same
vehicles. This is what makes the comparison fair: every controller is tested on identical traffic.
"""),
    section("traffic"),
    md("Quick check: the generated rates should be close to the target rates (they differ slightly because the arrivals are random)."),
    code("""
total_s = WARMUP_S + MEASURE_S + DRAIN_S
check = []
for name, sc in SCENARIOS.items():
    a = generate_arrivals(sc["rates"], seed=0)
    measured = a.sum(axis=0) / total_s * 3600
    check.append({"scenario": name, **{f"target_{d}": r for d, r in zip(DIRECTIONS, sc["rates"])},
                  **{f"generated_{d}": round(m) for d, m in zip(DIRECTIONS, measured)}})
pd.DataFrame(check)
"""),
    md("""
## 4. Constraint handling: the repair operator

Crossover and mutation can create invalid plans, for example a green of 7 s or a cycle longer than
180 s. Instead of throwing such plans away, every new plan passes through **`repair()`**:

1. **Round and clip.** Round each gene to whole seconds and clip it to `[10, 60]`.
2. **Sum too large** (cycle too long). Remove the extra seconds in proportion to each gene's
   *slack* (`g − 10`).
3. **Sum too small.** Add the missing seconds in proportion to each gene's *headroom* (`60 − g`).
4. **Leftover seconds.** Hand out any leftover single seconds by the *largest-remainder* rule, so the
   result is exact and deterministic.

**Why this always works.** Whenever the sum is too large, the genes have enough slack to absorb
the excess, and whenever it is too small they have enough headroom to absorb the deficit.
`validate_constraints()` guarantees both. The function ends with `assert is_valid(plan)`, and the
fitness function also asserts validity, so **an invalid plan can never be evaluated**.
"""),
    section("constraints"),
    code("""
for raw in ([5, 80, 30, 30], [60, 60, 60, 60], [3.7, 0, -10, 12], [25, 25, 25, 25]):
    plan = repair(raw)
    print(f"{str(raw):22s} -> {plan}  sum={sum(plan):3d}  cycle={cycle_length(plan):3d} s  valid={is_valid(plan)}")
"""),
    md("""
## 5. Traffic simulator

The simulator is a **queue simulation with 1-second steps**. Each road has a first-in-first-out
queue. Every second, for each road, three things happen:

1. Arriving vehicles join the back of the queue.
2. If the road has green, it earns 0.5 "departure credit" (1800 veh/h). Each full credit lets the
   front vehicle leave, so one vehicle leaves every 2 s of green.
3. The queue length is recorded.

**What is measured**
- **Average waiting time** (the main metric, in seconds per vehicle).
  - Only vehicles that **arrive during the 1-hour measurement window** are measured.
  - Warm-up vehicles are skipped, because the simulation starts with empty roads, which is
    unrealistic.
  - A vehicle's waiting time is `departure second − arrival second`. It is tracked into the drain
    period if needed.
  - **Censored vehicles.** A measured vehicle still waiting at the very end is counted with the
    time it has waited so far, and reported in `n_censored`. A non-zero value means the plan
    cannot keep up with demand.
- **Vehicles served:** vehicles that crossed the stop line during the 1-hour window.
- **Average queue length:** vehicles waiting on all roads, averaged over the window.

**Assumptions and limitations**
- **Point queues.** Vehicles wait "at" the stop line; the physical length of the queue is ignored.
- One lane group per road, with **no turning movements**, pedestrians or vehicle types.
- Constant saturation flow, with no start-up delay beyond the credit rule. No interaction with
  neighbouring intersections.
- Arrivals are Poisson with a **constant** average rate (no rush-hour peaks).
- The waiting time is therefore a *simulated stop-line waiting time*. It is useful for comparing
  plans, but it is not the official delay measure used in traffic engineering manuals.
"""),
    section("simulator"),
    md("Example: the Fixed-Equal plan (25 s each) on one Uneven traffic pattern. North receives 550 veh/h but 25 s of green can serve only about 360 veh/h, so its queue keeps growing."),
    code("""
example = simulate(generate_arrivals(SCENARIOS["Uneven"]["rates"], seed=0), (25, 25, 25, 25))
{k: (round(v, 2) if isinstance(v, float) else v) for k, v in example.items()}
"""),
    md("""
## 6. Fitness function

**Fitness of a plan = its average waiting time, averaged over the 3 training traffic patterns.
Lower is better, so the GA *minimises* fitness.** The same definition is used everywhere: in the GA,
in Random Search, and for the Fixed-Equal train value.

Averaging over three patterns makes the fitness less dependent on one lucky or unlucky random draw.

`CachedFitness` remembers every plan it has already scored. A repeated plan is not simulated again,
and `unique_evals` counts how many *different* plans were really simulated. That count is the fair
budget used for Random Search.
"""),
    section("fitness"),
    md("""
## 7. Controllers

- **Fixed-Equal (baseline):** every road gets 25 s of green, giving a 120 s cycle. This is the
  conventional, untuned signal that the GA must beat. Without a baseline, we could not tell whether
  the GA's plan is actually good.
- **Random plan:** four uniform random numbers in `[10, 60]`, repaired. This is used both for the
  GA's first population and for Random Search, so both start from the same kind of plans.
"""),
    section("controllers"),
    md("""
## 8. Genetic Algorithm (from scratch)

| GA concept | In this project |
|---|---|
| **Gene** | One green time, e.g. `g_N = 34` s |
| **Chromosome / individual** | One signal plan `[g_N, g_S, g_E, g_W]` |
| **Population** | 30 plans |
| **Fitness** | Average waiting time on the training traffic (minimise) |
| **Selection** | **Tournament** (k = 3): pick 3 plans at random; the one with the lowest waiting time becomes a parent |
| **Crossover** | **Uniform**, probability 0.8: each gene is copied from either parent with probability ½; the second child gets the other parent's gene |
| **Mutation** | Each gene, with probability 0.2, gets a random change drawn from Normal(0, 5 s) |
| **Constraint handling** | `repair()` after initialisation, crossover and mutation |
| **Elitism** | The 2 best plans are copied unchanged into the next generation, so the best solution is never lost |
| **Termination** | 40 generations |
| **History** | Best, mean and worst fitness of every generation |

**One generation:** evaluate everyone → keep the 2 elites → repeat (select 2 parents by tournament
→ crossover → mutate → repair) until there are 30 plans again.

**Why a GA here?** The waiting time comes from a random simulation, so there is no formula or
gradient to optimise. The search space is large: 51⁴ ≈ 6.8 million plans, each costing a simulation.
A GA needs only fitness values, and it combines good partial solutions (crossover) with small local
changes (mutation).
"""),
    section("ga"),
    md("**Try it:** a short GA run on the Uneven scenario, printing every generation."),
    code("""
demo_fitness = traffic_fitness(make_arrival_sets(SCENARIOS["Uneven"]["rates"], TRAIN_SEEDS))
demo = run_ga(demo_fitness, **{**GA_PARAMS, "generations": 15}, seed=0, verbose=True)
print("best plan:", demo["best_plan"], f"-> {demo['best_fitness']:.2f} s/veh,",
      demo_fitness.unique_evals, "different plans simulated")
"""),
    md("""
## 9. Random Search (comparison with the same budget)

Random Search evaluates *different* random valid plans and keeps the best one. It is given exactly
the **same number of unique evaluations** that the GA used in the run with the same seed.

If the GA beats Random Search, the improvement comes from the GA's selection, crossover and mutation,
not just from trying many plans.
"""),
    section("random_search"),
    md("""
## 10. Self-checks

These quick tests confirm that the building blocks behave correctly before the experiments run.
The same checks, and more, are in `tests/test_traffic_ga.py`, which can be run with `pytest`.
"""),
    code("""
rng = np.random.default_rng(0)
# 1. repair always produces a valid plan, even from extreme inputs
assert all(is_valid(repair(rng.uniform(-100, 300, 4))) for _ in range(5000))
assert repair((34, 12, 10, 10)) == (34, 12, 10, 10)          # valid plans are left unchanged
# 2. impossible settings are rejected
try:
    validate_constraints(g_min=50)
    raise AssertionError("should have failed")
except ValueError:
    pass
# 3. simulator: no vehicle is lost, and zero traffic gives zero waiting
a = generate_arrivals(SCENARIOS["Moderate"]["rates"], seed=5)
r = simulate(a, (30, 30, 25, 25))
assert all(r["arrived"][i] == r["departed"][i] + r["final_queue"][i] for i in range(4))
assert simulate(np.zeros_like(a), (25, 25, 25, 25))["avg_wait"] == 0
# 4. hand-worked case: car on N at t=0 leaves at t=1 (1 s); car on S at t=0 waits for S's green -> 16 s
tiny = np.zeros((70, 4), dtype=int); tiny[0, 0] = tiny[0, 1] = 1
r = simulate(tiny, (10, 10, 10, 10), warmup_s=0, measure_s=10, drain_s=60)
assert (r["avg_wait_N"], r["avg_wait_S"]) == (1, 16)
# 5. same seed -> same traffic -> same result
assert simulate(generate_arrivals((300, 300, 250, 250), 9), (30, 30, 25, 25)) == \\
       simulate(generate_arrivals((300, 300, 250, 250), 9), (30, 30, 25, 25))
# 6. fitness is the mean over the train seeds, and the cache does not change it
sets = make_arrival_sets((300, 300, 250, 250), [0, 1])
fit = traffic_fitness(sets)
assert abs(fit((30, 30, 25, 25)) - np.mean([simulate(s, (30, 30, 25, 25))["avg_wait"] for s in sets])) < 1e-9
assert fit((30, 30, 25, 25)) == fit((30, 30, 25, 25)) and fit.unique_evals == 1
# 7. the GA finds a known optimum on a toy problem, and its best-so-far never gets worse (elitism)
toy = run_ga(CachedFitness(lambda p: float(np.abs(np.array(p) - [37, 14, 22, 11]).sum())), generations=60, seed=1)
assert toy["best_plan"] == (37, 14, 22, 11)
assert (np.diff(toy["history"]["best_so_far"]) <= 0).all()
print("All self-checks passed ✔")
"""),
    md("""
## 11. Experiment and plotting functions

**`run_comparison()`** does the following for each scenario:
1. Build the 3 training traffic patterns and the 20 test traffic patterns.
2. Score the Fixed-Equal plan.
3. Run the GA 3 times (seeds 0, 1, 2). After each GA run, run Random Search with the same seed and
   **the same number of unique evaluations**.
4. Simulate every resulting plan on the **same 20 unseen test patterns**.

**`summarize()`** builds the results table. GA and Random Search are first averaged over their
3 runs for each test seed. The table then reports the **mean and standard deviation over the 20 test
seeds**, plus the improvement over Fixed-Equal.
"""),
    section("evaluation"),
    section("plots"),
    md("""
## 12. Experiment 1: controller comparison

`QUICK = True` gives a fast demo run (5 test seeds, 15 generations). The numbers discussed in
Section 16 come from the **full run** (`QUICK = False`).
"""),
    code("""
QUICK = False

test_seeds = TEST_SEEDS[:5] if QUICK else TEST_SEEDS
ga_params = {**GA_PARAMS, "generations": 15} if QUICK else GA_PARAMS
results_dir, figures_dir = Path("results"), Path("figures")
results_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

start = time.perf_counter()
test_results, runs, histories = run_comparison(test_seeds=test_seeds, ga_params=ga_params)
summary = summarize(test_results)
print(f"Experiment 1 finished in {time.perf_counter() - start:.0f} s")

test_results.to_csv(results_dir / "e1_test_results.csv", index=False)
runs.to_csv(results_dir / "e1_runs.csv", index=False)
summary.round(3).to_csv(results_dir / "e1_summary.csv", index=False)
histories.to_csv(results_dir / "e2_ga_history.csv", index=False)
"""),
    md("**Results table** (unseen test traffic, mean ± std over test seeds). Waiting time is in s/vehicle, vehicles served is per hour, queue is in vehicles."),
    code("""
table = pd.DataFrame({
    "Scenario": summary["scenario"],
    "Controller": summary["controller"],
    "Avg waiting time (s)": [f"{m:.1f} ± {s:.1f}" for m, s in zip(summary.avg_wait_mean, summary.avg_wait_std)],
    "Vehicles served (/h)": summary["vehicles_served_mean"].round(0).astype(int),
    "Avg queue (veh)": summary["avg_queue_mean"].round(1),
    "Still waiting at end": summary["n_censored_mean"].round(1),
    "Improvement vs Fixed": [f"{x:.1f} %" for x in summary["improvement_vs_fixed_%"]],
})
table
"""),
    md("**Best plan found in each run** (the train fitness is what the optimiser saw; the test average is on unseen traffic):"),
    code("""
runs.round(2)
"""),
    code("""
fig = plot_wait_bars(summary, figures_dir / "e1_avg_wait_bars.png")
plt.show()
"""),
    md("""
## 13. Experiment 2: GA convergence

- The **solid line** is the best fitness found so far in each generation, averaged over the 3 GA runs.
  The shaded band is the range across those runs.
- The **dotted line** is the population's mean fitness.
- The **dashed lines** show the Fixed-Equal plan and the result Random Search reached with the same budget.
- The y-axis is logarithmic, because early random plans can be very poor.
"""),
    code("""
fig = plot_convergence(histories, runs, figures_dir / "e2_convergence.png")
plt.show()

conv = histories.groupby(["scenario", "generation"])["best_so_far"].mean().unstack()
pd.DataFrame({
    "best at generation 0": conv[0].round(2),
    "best at generation 10": conv[min(10, conv.columns.max())].round(2),
    "best at final generation": conv[conv.columns.max()].round(2),
    "Fixed-Equal (train)": runs[runs.controller == "Fixed-Equal"].set_index("scenario")["train_fitness"].round(2),
}).loc[list(SCENARIOS)]
"""),
    md("## 14. Optimised green-light durations"),
    code("""
fig = plot_green_times(runs, figures_dir / "green_times.png")
plt.show()
"""),
    md("""
## 15. Experiment 3 (optional): effect of the mutation probability

The GA is run on the Uneven scenario with per-gene mutation probability 0.05, 0.2 (the default) and
0.5, 3 runs each, with everything else unchanged. `unique_evals` shows how many different plans each
run actually explored.
"""),
    code("""
mut_results, mut_histories = run_mutation_experiment(test_seeds=test_seeds, ga_params=ga_params)
mut_results.to_csv(results_dir / "e3_mutation.csv", index=False)
fig = plot_mutation(mut_results, mut_histories, figures_dir / "e3_mutation.png")
plt.show()
mut_results.groupby("p_mutation")[["train_fitness", "test_avg_wait", "unique_evals"]].agg(["mean", "std"]).round(2)
"""),
    md("""
## 16. Observations, limitations and conclusion

*The numbers below come from the full run (`QUICK = False`). The code is deterministic, so
re-running the notebook reproduces them exactly.*

**Experiment 1: controller comparison** (average waiting time on 20 unseen test patterns)

| Scenario | Fixed-Equal | Random Search | GA | GA vs Fixed |
|---|---|---|---|---|
| Light | 43.0 s | 36.5 s | **26.5 s** | 38.3 % lower |
| Moderate | 53.6 s | 53.2 s | **48.5 s** | 9.5 % lower |
| Uneven | 506.9 s* | 46.4 s | **39.6 s** | 92.2 % lower |

- **Light traffic.** The GA chose the shortest possible greens, about 10 s each, giving a 61 s cycle.
  With little traffic, a short cycle means a vehicle waits less for its next green. Fixed-Equal's
  120 s cycle makes vehicles wait for no reason.
- **Moderate traffic.** The GA gave the busier N and S roads more green (24, 24, 20, 20) and
  reduced waiting by about 10 %. The gain is smaller here because the equal plan is already
  reasonable for nearly balanced traffic.
- **Uneven traffic.** \\*Fixed-Equal cannot cope. With 25 s of green, North can serve only about
  360 veh/h, but 550 veh/h arrive.
  - North's queue grows all hour, and about 150 measured vehicles per test pattern are still
    waiting at the end.
  - Its 506.9 s is therefore a *lower bound* that would grow with a longer simulation.
  - The GA gives North a much longer green (34 s against 14/12/10 s) and serves about 1056
    vehicles/h instead of about 866.
- **Vehicles served** is almost the same for all controllers in Light and Moderate, because every
  plan can serve the demand there. The difference is in **how long** vehicles wait.
- **GA vs Random Search.** With the same number of evaluations, the GA was better in every
  scenario, so selection, crossover and mutation clearly help beyond random trial and error.

**Experiment 2: convergence.** Yes, the GA improves over successive generations.
- The best fitness drops quickly in the first ~10–15 generations and then levels off, and the
  3 independent runs end very close to each other.
- The population mean falls sharply too, as poor plans are replaced by better ones.
- In Light and Uneven, even the best *random* starting plan beats Fixed-Equal. In Moderate the GA
  passes the Fixed-Equal level after about 8 generations.
- The GA passes the Random Search result after about 4–12 generations, i.e. well before using its
  full budget, and keeps improving after that.

**Experiment 3: mutation probability.** The differences are small (under 0.5 s on test traffic),
so they should be read as trends:
- With **p = 0.05**, the population quickly fills with copies. The runs explored only about 230
  different plans, ended slightly worse and varied more between runs.
- With **p = 0.5**, the GA explored the most plans (about 900) and all 3 runs found the same plan.
- The default **0.2** sits in between.

**Advantages**
- The GA needs no formula or gradient, only a way to score a plan.
- It adapts the plan to the traffic pattern automatically.
- Constraints are always respected.
- It is easy to extend, for example by changing the fitness function.

**Limitations**
- Synthetic traffic and a simplified simulator (point queues, no turns, pedestrians or vehicle types).
- A single isolated intersection.
- Constant demand.
- The result is a fixed-time plan, not real-time control.
- The GA is stochastic, so it is run several times; more training patterns would reduce the risk
  of fitting the training traffic.

**Conclusion.** A from-scratch Genetic Algorithm, guided only by simulated waiting times, found
signal timings that reduced the average waiting time by about 10–92 % compared with an equal
fixed-time signal on unseen traffic. It also beat Random Search with the same budget. The largest
gains come where traffic is unbalanced, and those are exactly the situations where an equal split
is a poor choice.

**Future scope**
- Time-varying (rush-hour) demand.
- Several coordinated intersections.
- Comparison with Webster's classical formula.
- Calibration with real traffic counts.
- Adaptive control that re-optimises as traffic changes.
"""),
]


def main():
    sections = module_sections()
    used = [name for kind, name in CELLS if kind == "section"]
    missing = set(sections) - set(used)
    assert not missing, f"sections not placed in the notebook: {missing}"
    cells = []
    for kind, content in CELLS:
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": content.splitlines(keepends=True)})
        else:
            src = sections[content] if kind == "section" else content
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": src.splitlines(keepends=True)})
    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    out = ROOT / "Traffic_Signal_GA.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out.name} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
