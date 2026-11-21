# Research design

## Question and estimand

The project asks whether point-in-time industry relationships add out-of-sample information to stock-level financial factors for CSI 300 constituents. The core estimand is the change in weekly cross-sectional Rank IC on untouched 2024–2025 dates. A separate economic estimand is the change in net Top-10 portfolio performance after historical costs.

## Hypotheses

- **H1 (primary):** GCN versus Shared MLP tests graph message passing against a stock-only nonlinear model.
- **H2 (primary/secondary):** GAT versus Industry MLP is the primary test of learned graph aggregation beyond a simple peer mean; GCN versus Industry MLP is a secondary robustness comparison.
- **H3 (primary):** GAT versus GCN tests learned neighbor weights against normalized averaging on the same real graph.
- **H4 (two primary mechanism contrasts):** GAT Real versus GAT Random, and GAT Real versus GAT Self-loops Only, test whether gains depend on real cross-stock industry edges.
- **H5 (economic):** ensemble predictions are translated into a weekly Top-10 portfolio under historical Base costs. Top-5/Top-20 and other cost settings are sensitivities.

The primary paired-prediction family is Holm-corrected. Secondary comparisons are separately Holm-corrected and labeled secondary. Effect size, moving-block 95% interval, raw two-sided bootstrap p-value, and adjusted p-value are reported.

## Common information set

All M0–M7 models receive the same twelve signal-date factors and target the same next-open to following-rebalance-open excess return. Cross-sectional winsorization, median imputation, and z-scoring are performed separately on each signal date; they do not use a future date or test-period fitted scaler. The Industry MLP additionally receives the contemporaneous mean factor vector of other members in the same historical industry, making it the economically meaningful non-graph baseline.

## Graph models

For node features \(X\), a one-layer GCN computes

\[
H^{(1)}=\sigma\!\left(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}XW\right),
\]

where \(\tilde A=A+I\). GAT assigns an edge coefficient

\[
\alpha_{ij}=\operatorname{softmax}_{j\in\mathcal N(i)}
\left(\operatorname{LeakyReLU}\left(a^\top[Wh_i\Vert Wh_j]\right)\right)
\]

and aggregates \(\sum_j\alpha_{ij}Wh_j\). Attention only reweights predeclared edges; it does not discover arbitrary industrial relations. Unlike full Transformer self-attention, GAT attention is sparse and restricted to graph neighbors, while Transformer attention usually compares every token and commonly uses query/key/value projections.

## Frozen time protocol

Training is 2016–2021, development/validation is 2022–2023, and the final holdout is 2024–2025. Labels crossing a split boundary are purged. This version deliberately does not refit on train+validation: validation selects hidden width and early-stopping epoch under one common rule, while weights are fit only on training. This sacrifices some data efficiency for a simpler, uniform audit trail.

## Randomness

Non-random neural models use training seeds 42, 3407, and 2026. Random graph controls use an anchored simplification:

1. Fixed graph seed 1042 with all three training seeds estimates parameter-initialization variation and forms the reported random-graph ensemble.
2. Fixed training seed 42 with graph seeds 13407 and 12026, together with 1042, estimates graph-randomization variation.
3. Moving-block bootstrap across ordered test dates estimates date-sampling uncertainty.

This is five fits per random-graph model rather than nine. It labels uncertainty sources but does not estimate a full training-seed × graph-seed interaction.

## Statistical versus economic meaning

Rank IC measures whether predictions order stocks similarly to realized returns, not whether a tradeable portfolio earns money. MSE can improve without ranks improving. A Rank IC improvement may be overwhelmed by turnover, blocked trades, costs, and concentration. Statistical significance concerns sampling uncertainty; economic significance concerns size, stability, cost sensitivity, and drawdown. Neither substitutes for the other.
