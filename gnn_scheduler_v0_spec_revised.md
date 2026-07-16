# Adaptive Computational Scheduling for GNNs — Version 0 (Revised)

I am beginning Version 0 of a research project investigating adaptive computational
scheduling for Graph Neural Networks (GNNs). This is **NOT** intended to be
production code. Instead, it should be a clean, modular, and highly readable
research framework that will serve as the basis for several future papers.

The primary objective is to validate a new scheduling architecture, **NOT** to
maximize predictive accuracy. Simplicity, readability, and modularity are far
more important than optimization.

## Overall Goal

Implement a baseline two-layer Graph Convolutional Network (GCN) trained on
synthetic Stochastic Block Model (SBM) data.

Then implement a second version that is identical except that it includes a
computational scheduler based on Thompson Sampling.

The only architectural difference between the two models should be the
scheduling layer. Everything else should remain identical.

## Libraries

Use only common Python libraries.

- torch
- torch_geometric
- networkx
- numpy
- matplotlib
- scipy (if needed)
- sklearn

Avoid unnecessary dependencies.

## Code Organization

```
project/
│
├── data.py
│      Synthetic SBM generation
│
├── model.py
│      Standard 2-layer GCN
│      GAT-style attention layer (see "Standard Attention" below)
│
├── scheduler.py
│      Thompson Sampling
│      MED computation
│      Bayesian Expected Utility
│      Budget bookkeeping
│      Cumulative per-node pass-cost tracking
│
├── experiment.py
│      Runs V0 experiments
│
├── metrics.py
│      Accuracy
│      Runtime
│      Budget
│
├── plots.py
│      Three publication-quality figures
│
└── main.py
```

Keep every file concise. Use clear comments. Favor readability over cleverness.

## Dataset

Generate a synthetic Stochastic Block Model using NetworkX.

Start with approximately:
- 300 nodes
- 3 communities
- adjustable intra-community probability
- adjustable inter-community probability

Convert the graph into a PyTorch Geometric graph. The community labels become
the classification targets.

## Baseline GCN

Implement a standard two-layer GCN. Nothing fancy. No attention. No scheduler.
No architecture changes. This serves as the control.

## Scheduler Version

Duplicate the exact same GCN. The only difference is the scheduler. The
scheduler should include the following components.

---

### Graph Definition

Let `G = (V, E)` where:

- `V` is the node set
- `E` is the edge set

Each node `i` possesses a learned embedding `h_i`.

---

### Standard Attention *(revised)*

Version 0 now uses real GAT-style single-layer attention instead of the flat
placeholder `A_i = 1`.

Raw attention score between node `i` and neighbor `j`:

```
e_ij = LeakyReLU( a^T [ W h_i || W h_j ] )
```

- `W` — shared learnable linear projection applied to every node embedding
- `a` — shared learnable attention vector
- `||` — concatenation
- LeakyReLU negative slope: **0.2** (GAT default — flag if you want this
  configurable instead of hardcoded)

Normalized attention weight (softmax over node `i`'s neighborhood):

```
a_ij = exp(e_ij) / Σ_{k ∈ N(i)} exp(e_ik)
```

**Node-level attention score for the MED formula.** The original MED formula
treats `A_i` as a single scalar per node, but `a_ij` is edge-wise. For V0, define:

```
A_i = mean_{j ∈ N(i)} a_ij
```

i.e. the average attention weight node `i` assigns to its neighbors. This is a
placeholder aggregation choice — flag if you'd rather use `max_j a_ij` (peak
attention) or the attention received rather than given. Keep this swappable;
it's exactly the kind of thing V0 should make easy to change later.

The framework should still allow attention to be replaced or extended later
(e.g., multi-head attention is a natural V1 direction).

---

### Uncertainty

Use predictive entropy:

```
U_i = - Σ_c p_i log(p_i)
```

computed from the model output probabilities.

---

### Bayesian Expected Utility

For Version 0, estimate:

```
BEU_i = U_i - λ C_i
```

using uncertainty as the first-order estimator. Use `λ = 0.1` initially. Make
this easy to change later.

---

### Computational Cost *(revised)*

`C_i` is no longer a flat constant. It now tracks **cumulative additional
passes** through node `i` within a run:

- The node's *first* pass in a run is free — it is not counted, so genuine
  first-exploration is never penalized.
- Each additional (repeat) pass increments the cost by 1:
  - 1st extra pass → `C_i = 1`
  - 2nd extra pass → `C_i = 2`
  - 3rd extra pass → `C_i = 3`
  - ...and so on, cumulative for the life of the run.

This means `scheduler.py` needs a per-node pass counter (e.g. a dict or tensor
indexed by node id) that persists across scheduling rounds within a run and
resets at the start of each new run/graph. Keep this bookkeeping isolated in
`scheduler.py` so `model.py` stays untouched by cost logic.

---

### Meditation

Compute:

```
MED_i = 0.35 A_i + 0.35 U_i + 0.20 BEU_i - 0.10 C_i
```

Keep these weights as constants. Do **NOT** learn them. Note that `A_i` and
`C_i` here now carry real signal (attention-derived and pass-count-derived,
respectively) rather than being a flat placeholder — worth watching in early
runs since this changes the dynamic range of the MED distribution compared to
the original V0 draft.

---

### Laplace Smoothing

Apply:

```
MED_i' = (MED_i + α) / (1 + α)
```

with `α = 0.01` to prevent starvation.

---

### Thompson Sampling

Implement a lightweight Thompson Sampling scheduler. The scheduler should:

- sample from each node
- determine which nodes receive an additional message passing step
- update its beliefs each iteration

Keep this implementation intentionally simple.

---

### Computational Budget

Maintain `b_i` for every node, and `B = Σ_i b_i`.

The budget is bookkeeping **only**. It should never affect scheduling
decisions. (Note: this is now distinct from `C_i`, which *does* affect
scheduling via BEU/MED — `B`/`b_i` remain a pure accounting ledger, separate
from the cumulative pass-cost used in the utility calculation.)

---

## Metrics

Collect exactly these metrics.

**Performance**
- Accuracy
- Precision
- Recall
- F1 Score

**Computational**
- Runtime
- Total Computational Budget
- Average Budget Per Node
- Scheduler Efficiency = Accuracy / Computational Budget

**Scheduler Diagnostics**
- Node Selection Distribution
- MED Distribution

Print these in a clean experiment summary.

## Experiments

Run:
- Standard GCN
- Scheduled GCN

using identical random seed, optimizer, learning rate, epochs, and
architecture. Repeat over several random SBM graphs. Average the results.

## Plots

Create ONLY THREE clean publication-quality figures.

**Figure 1 — Accuracy Comparison**
Simple bar chart, Baseline vs Scheduled.

**Figure 2 — Runtime vs Computational Budget**
Scatter plot showing Baseline and Scheduler.

**Figure 3 — Histogram of MED Scores**
Demonstrates where computation was being allocated.

No more figures. Keep them simple.

## Coding Style

Prioritize readability, modularity, documentation, clean function names,
simple interfaces. Avoid premature optimization, unnecessary abstraction, and
long classes. Prefer small reusable functions.

## Important

This is Version 0 of a research framework. The scheduler is the experiment.
The GCN should remain as close to the standard implementation as possible.
Every design decision should preserve that philosophy — including the two
revisions above: real attention and cumulative pass-cost make `A_i` and `C_i`
meaningful signals rather than placeholders, but everything downstream (MED
weights, Thompson Sampling, budget bookkeeping, metrics, plots) stays exactly
as originally specified.
