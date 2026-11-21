# Frozen FULL_RUN protocol

Protocol ID: `CSI300_INDUSTRY_GNN_FULL_V2_LOCAL`. Status: frozen before opening the 2024–2025 final output.

## Dates and observations

- Raw window: 2015-09-01 to 2026-01-15, including warm-up and final label exit.
- Train: 2016-01-01 to 2021-12-31.
- Validation/development: 2022-01-01 to 2023-12-31.
- Final test: 2024-01-01 to 2025-12-31.
- Weekly signal at the last open day of the week; execution is the next open; exit is the next rebalance open.
- Rows whose scheduled label exit crosses a split boundary are excluded from that split.

## Models and common budget

M0 Ridge; M1 Shared MLP; M2 Industry MLP; M3 GCN Real; M4 GAT Real; M5 GCN Random; M6 GAT Random; M7 GAT Self-only. No LSTM or Transformer.

Neural search widths are 32 and 64. Encoders and graph networks use one layer, GAT uses two heads, dropout 0.15, AdamW learning rate 0.001, weight decay 0.0001, maximum epochs 150, patience 20, and minimum improvement \(10^{-6}\). Batch size is eight weekly graphs. All neural models receive the same width search and stopping rule. Selection seed is 42. Ridge alphas are 1 and 10. AMP is frozen off because it was not validated on the target laptop before freezing.

## Random seeds

- Training: `[42, 3407, 2026]` for every neural architecture.
- Random graphs: `[1042, 13407, 12026]`.
- Random-graph main ensemble: all three training seeds on graph 1042.
- Graph sensitivity: training seed 42 on all three graph seeds.
- Degree-preserving rewiring: three attempted swaps per edge; the graph audit records structure over time.

## Predictions and portfolios

Reported model predictions are the arithmetic mean of three `main_training_seed` replicas. Per-replica metrics are in `training_seed_results.csv`; topology-sensitivity replicas are labeled `graph_randomization` and are not mixed into the main ensemble.

The primary strategy is weekly Top-10 equal weight under historical Base costs: 2 bps commission per side, 5 bps slippage per side, and sell stamp duty of 10 bps through 2023-08-27 and 5 bps thereafter. Top-5/Top-20, low/high costs, and fixed-low-tax counterfactual are sensitivity results. Turnover is buy plus sell notional divided by pre-trade equity. Sharpe uses daily net returns and a zero risk-free rate. Annualized excess return is strategy CAGR minus benchmark CAGR.

## Predeclared paired tests

Primary Holm family: GCN−Shared MLP (H1), GAT−Industry MLP (H2), GAT−GCN (H3), GAT Real−GAT Random (H4a), and GAT Real−GAT Self-only (H4b). GCN−Industry MLP is secondary. Each uses same-date Rank IC differences, a circular moving-block bootstrap with block length four weeks and 5,000 repetitions, and reports paired dates, mean effect, 95% interval, raw p-value, and family-specific Holm-adjusted p-value.

No hyperparameter, factor, graph rule, test, or trading rule may be changed after reviewing final results without creating a new protocol ID and labeling the new analysis exploratory.
