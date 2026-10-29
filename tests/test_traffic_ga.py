"""Basic tests for the simulator, constraint handling, fitness and GA. Run with: pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import traffic_ga as t  # noqa: E402

TOTAL = t.WARMUP_S + t.MEASURE_S + t.DRAIN_S


# ---- constraints and repair --------------------------------------------------------------
def test_default_green_sum_range():
    assert (t.S_LO, t.S_HI) == (40, 160)


@pytest.mark.parametrize("kwargs", [dict(g_min=50), dict(g_min=30, g_max=20), dict(c_max=50, g_min=20)])
def test_impossible_constraints_rejected(kwargs):
    with pytest.raises(ValueError):
        t.validate_constraints(**kwargs)


def test_repair_always_valid():
    rng = np.random.default_rng(0)
    inputs = [rng.uniform(-100, 300, 4) for _ in range(5000)]
    inputs += [[0, 0, 0, 0], [1000] * 4, [-5, 999, 0.4, 61], [10.5, 10.5, 10.5, 10.5]]
    for g in inputs:
        assert t.is_valid(t.repair(g))


def test_repair_keeps_valid_plans_unchanged():
    for plan in [(10, 10, 10, 10), (60, 60, 20, 20), (25, 25, 25, 25), (34, 12, 10, 10)]:
        assert t.repair(plan) == plan


def test_is_valid_rejects_bad_plans():
    for bad in [(9, 30, 30, 30), (61, 30, 30, 30), (60, 60, 60, 60), (25.5, 25, 25, 25), (25, 25, 25)]:
        assert not t.is_valid(bad)


def test_green_schedule_structure():
    plan = (20, 15, 10, 30)
    sched = np.array(t.green_schedule(plan))
    assert len(sched) == t.cycle_length(plan) == 95
    assert sched.sum(axis=1).max() == 1               # never two roads green together
    assert list(sched.sum(axis=0)) == list(plan)      # each road gets exactly its green time
    assert sched[:20, 0].all() and not sched[20:25].any()  # N first, then 5 s amber + all-red


# ---- traffic generation ------------------------------------------------------------------
def test_arrivals_reproducible_and_rate_close():
    a1 = t.generate_arrivals((300, 300, 250, 250), seed=7)
    a2 = t.generate_arrivals((300, 300, 250, 250), seed=7)
    assert a1.shape == (TOTAL, 4) and (a1 == a2).all()
    long = t.generate_arrivals((600, 0, 300, 100), seed=1, total_s=200_000)
    rate = long.sum(axis=0) / 200_000 * 3600
    assert np.allclose(rate, (600, 0, 300, 100), rtol=0.05, atol=1)


# ---- simulator ---------------------------------------------------------------------------
def test_zero_traffic():
    r = t.simulate(np.zeros((TOTAL, 4), dtype=int), (25, 25, 25, 25))
    assert r["avg_wait"] == 0 and r["vehicles_measured"] == 0 and r["avg_queue"] == 0


def test_conservation():
    a = t.generate_arrivals((550, 200, 150, 150), seed=3)
    r = t.simulate(a, (25, 25, 25, 25))
    for i in range(4):
        assert r["arrived"][i] == r["departed"][i] + r["final_queue"][i] == a[:, i].sum()


def test_hand_worked_example():
    # No warm-up, 10 s window, plan (10,10,10,10): N green 0-9, S green 15-24.
    # One car on N at t=0: credit 0.5 at t=0, 1.0 at t=1 -> leaves at t=1 (wait 1 s).
    # One car on S at t=0: red until t=15; credit 0.5 at t=15, 1.0 at t=16 -> wait 16 s.
    a = np.zeros((70, 4), dtype=int)
    a[0, 0] = a[0, 1] = 1
    r = t.simulate(a, (10, 10, 10, 10), warmup_s=0, measure_s=10, drain_s=60)
    assert r["avg_wait_N"] == 1 and r["avg_wait_S"] == 16 and r["avg_wait"] == 8.5


def test_no_departures_on_red_and_censoring():
    # A car arriving on W during N's green cannot leave before W's green (starts at 45 s).
    a = np.zeros((40, 4), dtype=int)
    a[5, 3] = 1
    r = t.simulate(a, (10, 10, 10, 10), warmup_s=0, measure_s=10, drain_s=30)
    assert r["departed"][3] == 0 and r["n_censored"] == 1 and r["avg_wait"] == 40 - 5


def test_warmup_vehicles_not_measured():
    a = np.zeros((100, 4), dtype=int)
    a[3, 0] = 1  # arrives during the 10 s warm-up
    r = t.simulate(a, (10, 10, 10, 10), warmup_s=10, measure_s=30, drain_s=60)
    assert r["vehicles_measured"] == 0 and r["departed"][0] == 1


def test_more_green_less_wait_for_that_road():
    a = t.generate_arrivals((550, 200, 150, 150), seed=4)
    assert t.simulate(a, (40, 20, 20, 20))["avg_wait_N"] < t.simulate(a, (20, 20, 20, 20))["avg_wait_N"]


def test_simulate_rejects_invalid_plan():
    with pytest.raises(ValueError):
        t.simulate(np.zeros((TOTAL, 4), dtype=int), (5, 25, 25, 25))


# ---- fitness, GA and random search -----------------------------------------------------------
def test_fitness_is_mean_over_train_seeds_and_cached():
    sets = t.make_arrival_sets((300, 300, 250, 250), [0, 1])
    fit = t.traffic_fitness(sets)
    plan = (30, 30, 25, 25)
    expected = np.mean([t.simulate(a, plan)["avg_wait"] for a in sets])
    assert fit(plan) == pytest.approx(expected) and fit(plan) == pytest.approx(expected)
    assert fit.unique_evals == 1 and fit.calls == 2


def test_crossover_genes_come_from_parents_when_valid():
    rng = np.random.default_rng(0)
    p1, p2 = (10, 20, 30, 40), (40, 30, 20, 10)  # both sum to 100, so children need no repair
    for _ in range(50):
        c1, c2 = t.uniform_crossover(p1, p2, rng)
        assert all(g in (a, b) for g, a, b in zip(c1, p1, p2))
        assert all(g in (a, b) for g, a, b in zip(c2, p1, p2))


def test_mutation_output_valid():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        assert t.is_valid(t.mutate((60, 60, 20, 20), p_mutation=1.0, sigma=20, rng=rng))


def test_ga_finds_known_optimum_on_toy_problem():
    target = np.array([37, 14, 22, 11])
    fit = t.CachedFitness(lambda p: float(np.abs(np.array(p) - target).sum()))
    res = t.run_ga(fit, pop_size=30, generations=60, seed=1)
    assert res["best_plan"] == tuple(target) and res["best_fitness"] == 0
    assert (np.diff(res["history"]["best_so_far"]) <= 0).all()  # elitism: never gets worse


def test_ga_reproducible_and_valid_on_traffic():
    sets = t.make_arrival_sets((550, 200, 150, 150), [0], total_s=TOTAL)
    r1 = t.run_ga(t.traffic_fitness(sets), pop_size=10, generations=5, seed=3)
    r2 = t.run_ga(t.traffic_fitness(sets), pop_size=10, generations=5, seed=3)
    assert r1["best_plan"] == r2["best_plan"] and t.is_valid(r1["best_plan"])


def test_random_search_uses_exact_budget():
    fit = t.CachedFitness(lambda p: float(sum(p)))
    res = t.run_random_search(fit, budget=150, seed=0)
    assert fit.unique_evals == 150 and t.is_valid(res["best_plan"])
