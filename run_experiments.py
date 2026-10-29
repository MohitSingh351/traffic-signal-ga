"""Run all experiments from the command line (same code and results as the notebook).

    python run_experiments.py           # full run (~2 min)
    python run_experiments.py --quick   # 5 test seeds, 15 generations (fast smoke test)
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import traffic_ga as t  # noqa: E402

quick = "--quick" in sys.argv
test_seeds = t.TEST_SEEDS[:5] if quick else t.TEST_SEEDS
ga_params = {**t.GA_PARAMS, "generations": 15} if quick else t.GA_PARAMS
results, figures = Path("results"), Path("figures")
results.mkdir(exist_ok=True)
figures.mkdir(exist_ok=True)

t0 = time.perf_counter()
print("Experiment 1 + 2: controller comparison and GA convergence")
test_results, runs, histories = t.run_comparison(test_seeds=test_seeds, ga_params=ga_params)
summary = t.summarize(test_results)
test_results.to_csv(results / "e1_test_results.csv", index=False)
runs.to_csv(results / "e1_runs.csv", index=False)
summary.round(3).to_csv(results / "e1_summary.csv", index=False)
histories.to_csv(results / "e2_ga_history.csv", index=False)
t.plot_wait_bars(summary, figures / "e1_avg_wait_bars.png")
t.plot_convergence(histories, runs, figures / "e2_convergence.png")
t.plot_green_times(runs, figures / "green_times.png")

print("Experiment 3: mutation probability")
mut_results, mut_histories = t.run_mutation_experiment(test_seeds=test_seeds, ga_params=ga_params)
mut_results.to_csv(results / "e3_mutation.csv", index=False)
t.plot_mutation(mut_results, mut_histories, figures / "e3_mutation.png")

print(summary[["scenario", "controller", "avg_wait_mean", "avg_wait_std", "vehicles_served_mean",
               "avg_queue_mean", "n_censored_mean", "improvement_vs_fixed_%"]].round(2).to_string(index=False))
print(f"done in {time.perf_counter() - t0:.0f} s; tables in {results}/, plots in {figures}/")
