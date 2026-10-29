# 🚦 Traffic Signal Timing Optimization Using Genetic Algorithms

**A Simulation-Based Comparative Study** (Soft Computing course project)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohitSingh351/traffic-signal-ga/blob/main/Traffic_Signal_GA.ipynb)

This project uses a Genetic Algorithm (GA), written from scratch in NumPy, to choose the green-light
duration of each road at a four-way intersection so that vehicles wait as little as possible. Every
candidate plan is scored by a simple traffic simulation running on synthetic traffic.

The GA is compared with two other controllers:
- an **equal fixed-time signal** (Fixed-Equal);
- **Random Search** with the same evaluation budget.

All comparisons use three traffic scenarios and 20 unseen test traffic patterns.

> The traffic data is **synthetic** and the simulator is deliberately simplified. There is no
> real-world data, no deployment, and no real-time control.

## Files

| File | Purpose |
|---|---|
| `Traffic_Signal_GA.ipynb` | **Main deliverable.** A self-contained notebook with explanations, all code, the experiments, tables and plots. It is already executed, so the outputs are visible on GitHub. |
| `traffic_ga.py` | The same code as a Python module: settings, scenarios, traffic generator, repair, simulator, fitness, GA, Random Search, experiments and plots |
| `run_experiments.py` | Runs all experiments from the terminal and writes `results/` and `figures/` (same numbers as the notebook) |
| `tests/test_traffic_ga.py` | 22 basic tests covering the simulator, constraint handling, fitness and the GA |
| `make_notebook.py` | Developer tool that rebuilds the notebook from the sections of `traffic_ga.py`, so the two never differ |
| `results/*.csv` | Result tables from the actual run |
| `figures/*.png` | Plots from the actual run |
| `REPORT.md` | Report material: abstract through references, with the actual results |
| `VIVA.md` | 20 likely viva questions with short answers |

## How to run

**Google Colab (easiest)**
1. Click the *Open in Colab* badge above.
2. Choose **Runtime → Run all**.

No installation is needed. A full run takes about 2–3 minutes. Set `QUICK = True` in Section 12 for a faster demo.

**Locally**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                      # 22 tests
python run_experiments.py      # ~1 min on a laptop; writes results/ and figures/
jupyter notebook Traffic_Signal_GA.ipynb   # optional (pip install notebook)
```

## Results

Average waiting time on 20 unseen test traffic patterns, in seconds per vehicle (mean ± std):

| Scenario | Fixed-Equal | Random Search | GA | GA vs Fixed |
|---|---|---|---|---|
| Light (150/150/150/150 veh/h) | 43.0 ± 1.2 | 36.5 ± 0.8 | **26.5 ± 1.1** | −38.3 % |
| Moderate (300/300/250/250) | 53.6 ± 3.7 | 53.2 ± 3.7 | **48.5 ± 2.3** | −9.5 % |
| Uneven (550/200/150/150) | 506.9 ± 53.3* | 46.4 ± 2.9 | **39.6 ± 2.6** | −92.2 % |

\*Fixed-Equal cannot serve North's 550 veh/h with 25 s of green. Its queue keeps growing, and about
150 measured vehicles per pattern are still waiting at the end of the simulation, so this value is a lower bound.

See `REPORT.md` for the full discussion.
