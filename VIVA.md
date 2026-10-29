# Viva Preparation: Likely Questions and Short Answers

## Problem and motivation

**1. Why did you choose this problem?**
Traffic signals with poorly chosen green times make vehicles wait unnecessarily. Choosing green
times is a real optimisation problem with constraints and random traffic, which makes it a good,
easy-to-understand application of a soft computing technique.

**2. What exactly are you optimising?**
The four green-light durations `[g_N, g_S, g_E, g_W]` of a four-way intersection. The objective is
to **minimise the average waiting time per vehicle**, subject to 10–60 s per green and a cycle
length of 40–180 s.

**3. Is this real-time / adaptive control?**
No. The GA works **offline**: it finds one good fixed-time plan for a given traffic pattern. Real-time
adaptive control is listed as future scope.

## Soft computing and the GA

**4. What is soft computing?**
A set of techniques that accept approximation and uncertainty in exchange for good, practical
solutions to hard problems. Examples are genetic algorithms, fuzzy logic, neural networks and swarm
intelligence. Hard computing, by contrast, needs exact models and exact solutions.

**5. Why is a Genetic Algorithm appropriate here?**
- The waiting time comes from a **random simulation**, so there is no formula and no gradient to optimise.
- The search space is large: 51⁴ ≈ 6.8 million plans, each needing a simulation.
- A GA needs only fitness values, searches many solutions in parallel, and combines good partial
  solutions.

**6. What are a gene, a chromosome and a population in your project?**
- A **gene** is one green time, for example `g_N = 34` s.
- A **chromosome** (individual) is one complete signal plan `[g_N, g_S, g_E, g_W]`.
- The **population** is 30 such plans in each generation.

**7. What is your fitness function?**
The average waiting time per vehicle (s/veh) produced by simulating the plan on 3 training traffic
patterns, averaged over the three. **Lower is better**, so the GA minimises it. Averaging over several
patterns reduces the effect of a single random draw.

**8. How does tournament selection work?**
- Pick 3 plans at random from the population.
- The one with the lowest waiting time becomes a parent.
- Better plans win more often, but weaker plans sometimes get through, which maintains diversity.
- Larger tournaments mean stronger selection pressure.

**9. How does uniform crossover work?**
- For each of the 4 genes, a coin flip decides whether child 1 copies it from parent 1 or parent 2.
  Child 2 gets the other parent's gene.
- Example: parents `[34,14,12,10]` and `[25,25,25,25]` can produce `[34,25,12,25]` and `[25,14,25,10]`.
- Crossover happens with probability 0.8; otherwise the parents are copied unchanged.

**10. How does mutation work, and why is it needed?**
- Each gene, with probability 0.2, gets a random change drawn from a normal distribution with σ = 5 s.
  Example: 34 → 31.
- Mutation explores nearby plans and keeps diversity. Without it, the population would quickly
  become copies of one plan (premature convergence), as Experiment 3 showed with p = 0.05.

**11. What is elitism?**
The 2 best plans of each generation are copied unchanged into the next one. This guarantees the
best solution is never lost, so the best fitness never gets worse from one generation to the next.

**12. How do you make sure the GA never produces invalid green times?**
The **repair operator** runs after initialisation, crossover and mutation:
1. Round each gene and clip it to [10, 60].
2. If the total green is above 160 s (cycle > 180), remove the excess in proportion to each gene's
   slack above 10. If it is below 40 s, add the deficit in proportion to each gene's headroom below 60.

There is always enough slack or headroom, and the settings are validated at start-up. The fitness
function also asserts that every plan is valid before simulating it.

**13. When does your GA stop?**
After 40 generations, with a population of 30. The convergence plot shows that most of the
improvement happens within the first 10–15 generations.

## Simulation and evaluation

**14. How does your traffic simulation work?**
- **Steps:** it moves forward one second at a time. Arrivals on each road are random (Poisson) at
  that road's rate.
- **Queues:** vehicles wait in a first-in-first-out queue.
- **Signal:** only one road has green at a time (N → S → E → W), and each green is followed by 3 s
  amber and 2 s all-red.
- **Discharge:** a green road releases one vehicle every 2 s (1800 veh/h).
- **Measurement:** each measured vehicle's waiting time is its departure time minus its arrival time.

**15. Why the warm-up and drain periods?**
- **Warm-up.** The simulation starts with empty roads, which is unrealistic, so the first 300 s are
  not measured.
- **Drain.** It gives vehicles that arrived during the 1-hour measurement window time to leave, so
  their full waiting time is counted.
- **Censored vehicles.** Vehicles still waiting at the very end are counted with their wait so far
  and reported separately.

**16. Why do you need a baseline controller?**
Without a baseline, there is no way to tell whether the GA's plan is actually good. The Fixed-Equal
controller (25 s for every road) represents a conventional untuned signal. Random Search, with the
same budget, checks that the GA's operators help, not just the large number of plans tried.

**17. How did you make the comparison fair?**
- **Same traffic.** All controllers are tested on exactly the same simulated vehicles, using fixed seeds.
- **Same conditions.** They share the same simulator and the same constraints.
- **Equal budget.** Random Search evaluates the same number of distinct plans as the GA.
- **Separate test traffic.** The GA and Random Search optimise on 3 training patterns, but results
  are reported on 20 different, unseen test patterns.
- **Repeated runs.** The GA is run 3 times with different seeds.

**18. What results did you get?**
On unseen test traffic, the GA reduced the average waiting time compared with Fixed-Equal:

| Scenario | Fixed-Equal | GA | Reduction |
|---|---|---|---|
| Light | 43.0 s | 26.5 s | 38 % |
| Moderate | 53.6 s | 48.5 s | 9.5 % |
| Uneven | 506.9 s | 39.6 s | 92 % |

It also beat Random Search in every scenario. In Uneven, the equal plan cannot serve North's
550 veh/h (25 s of green serves only about 360 veh/h), so its queue keeps growing.

**19. Why does the GA give such short greens in Light traffic?**
With little traffic, queues clear quickly, so long greens are wasted. A shorter cycle (61 s instead
of 120 s) means each vehicle waits less for its road's next green.

## Limitations and future work

**20. What are the limitations, and how would you improve the project?**

*Limitations:*
- synthetic traffic and a simplified simulator (point queues, no turns, pedestrians or vehicle types);
- constant demand and a single isolated intersection;
- an offline fixed-time plan.

*Improvements:*
- time-varying (peak-hour) plans;
- coordination of several intersections;
- a more detailed simulator such as SUMO;
- comparison with Webster's method or other metaheuristics such as PSO;
- calibration with real traffic counts;
- adaptive real-time control.
