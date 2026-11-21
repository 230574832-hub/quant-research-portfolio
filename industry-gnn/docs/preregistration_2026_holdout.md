# Pre-registration — 2026 untouched holdout (five-arm cross-industry graph experiment)

**Protocol id:** `CSI300_INDUSTRY_GNN_HOLDOUT_2026` · **Version:** 1.0 · **Status:** FROZEN before any 2026 outcome exists
**Created (UTC):** 2026-11-11T10:08:28Z · **Source root cited below:** local read-only results directory (all relative paths are under it).
**Evidence labels.** 2026 results = `CONFIRMATORY_PREREGISTERED_HOLDOUT` only if gates G1–G2 (§7) pass; otherwise no 2026 number is reported. 2024–2025 = `EXPLORATORY_POST_HOC` (`phase7_cross_industry_style/configs/phase7_arms.yaml:12-23`).

## 1. Purpose
Review priority #3: freeze a few principal hypotheses and test them on data that never entered development. Prior artefacts record that 2024-01-01..2025-12-31 was used repeatedly (`phase7_cross_industry_style/src/phase7/constants.py:29-35`). The untouched window is **2026-01-16..2026-11-10**: the previous snapshot's `FULL_RUN.data.end` was 2026-01-15 (`v4_phase5_final/data/v4_only/config.yaml:204-207`), and the last development weekly return period ended 2026-01-02 (`phase7_cross_industry_style/results/phase7_economic_backtest/weekly_portfolio_returns.csv`, final row `2025-12-27/2026-01-02`).

## 2. Untouchable-window rule
1. Any `signal_date` in **[2026-01-16, 2026-11-10]** is holdout. No 2026 row may enter training, early stopping, hyperparameter or threshold choice, feature-normalisation statistics, arm graph-property tuning, or any date selection after outcomes are seen.
2. Training 2016-01-01..2021-12-31 and validation 2022-01-01..2023-12-31 stay frozen (`config.yaml:120-127`; `constants.py:29-35`); 2024–2025 remains a development-only replay window.
3. Pre-2026 data may be used only as trailing input history (60-session correlation lookback, 60-day features), never as a label source for holdout dates.
4. Feature normalisation is per-signal-date cross-sectional only (`src/industry_gnn/features.py:156-169`), so 2026 distributional information cannot enter any statistic.
5. **Unblinding order:** build panel → train and score 2016–2025 → write 2026 predictions → hash the prediction file → join 2026 labels → compute every endpoint once. No post-unblinding re-runs, subsets, endpoint edits, or family edits.
6. Signal dates are those of the frozen `_weekly_schedule` (`features.py:137-158`) inside the window with ≥90% label coverage; the final schedule date (no next signal) is excluded. Planning value **N = 40**, expected 38–42. If N < 36 the analysis is still reported but declared underpowered and no directional claim is made.

## 3. Frozen design
Arms exactly as frozen (`phase7_arms.yaml:80-102`, `constants.py:147-210`): **A** self-only, **B** intra-industry (≤4 out-neighbours), **C** cross-industry (≤4), **D** hybrid (2 intra + 2 cross), **E** matched random cross (same node/edge counts, per-node in/out degree, cross-industry count and share; weights recomputed from the new pair's history).
Relationship score: 60 completed sessions, ≥40 joint sessions, |corr| ≥ 0.05, 17:00 decision, `trade_date ≤ signal_date` (`constants.py:136-138`).
Architecture/training never retuned: 12 raw + 12 peer factors, hidden 32, 1 GAT layer, dropout 0.15, gate bias −3, AdamW 1e-3/1e-4, ≤150 epochs, patience 20, validation-MSE early stopping, training seeds 42/3407/2026 (`phase7_arms.yaml:104-129`). No 2024+ data is fitted; 2024–2025 is used only by gate G2.

## 4. Hypotheses, endpoints, multiplicity family
**Primary family (3):** **H1 = C−A** (any relational graph vs none), **H2 = C−B** (cross- vs intra-industry), **H3 = C−E** (real cross-industry vs matched random). The old file froze {C−B, D−B, C−E} (`phase7_arms.yaml:131-138`); D−B and D−C are reported as secondary, outside multiplicity control. This re-specification is disclosed and frozen before any 2026 outcome.

**Co-primary endpoints (holdout dates only):**
- **P1** mean per-date RankIC — Spearman(seed-mean `y_pred`, `target_excess_return`) per signal date, then mean over dates.
- **P2** annualised net Top-10 return at base cost — conservative open execution, 3-seed-mean ranking, blocked buys not replaced, blocked sells carried, no same-day high/low (`results/phase7_economic_backtest/backtest_execution_config.json`: 2 bps commission, 5 bps slippage, sell stamp duty 5 bps since 2023-08-28).

**Secondary (descriptive, unadjusted, no significance claims):** Sharpe, max drawdown, turnover, low/high cost, Top-5/Top-20, seed dispersion, graph-instance dispersion, blocked-order counts. No composite score exists.

## 5. Statistical plan
Date-paired circular moving-block bootstrap, block length **4 weekly dates**, **5,000** repetitions, two-sided p = 2·min(P(mean≤0), P(mean≥0)), seeds 42 + 97·index; Holm within each endpoint family of 3 (`src/phase7/statlib.py:30-142`; `src/phase7/evaluation.py:101-124`). p-resolution 2×10⁻⁴. Annualised return difference = 52 × mean weekly difference and CI bounds ×52, matching `gnnenh/stats.py:200`. No alternative test may replace this one.

## 6. Power limitation (honest)
MDE at α = 0.05 two-sided, 80% power, using **realised 2024–2025 weekly net returns** (`daily_portfolio_returns.csv`, 103 weeks):

`MDE_ann = 52 · (z₀.₉₇₅ + z₀.₈) · σ_diff / √N = 52 · 2.8016 · σ / √N`, variance × (1+ρ)/(1−ρ), ρ ≥ 0.

| contrast | σ_weekly | MDE N=40, ρ=0 | MDE N=40, ρ=0.2 | 95% CI half-width |
|---|---:|---:|---:|---:|
| H1 C−A | 1.847% | **42.5 pp** | 52.1 pp | ±29.8 pp |
| H2 C−B | 2.196% | **50.6 pp** | 62.0 pp | ±35.4 pp |
| H3 C−E | 1.700% | **39.2 pp** | 48.0 pp | ±27.4 pp |

Arithmetic (H2): SE = 2.196%/√40 = 0.347%/week; MDE_weekly = 2.8016 × 0.347% = 0.973%; ×52 = **50.6 ppt**. RankIC MDE: 0.0505 (C−A), 0.0446 (C−B), 0.0197 (C−E). The exploratory 2024–2025 point estimates were +7.3 / +12.5 / −5.0 ppt — every prior effect is 4–8× below MDE. **Conclusion: 2026 is a low-power check of direction and sign stability, not a definitive test; a null result is not evidence of no effect.** Full table in `power_analysis.csv`.

## 7. Random-graph control and gates
Arm E uses **5 fresh matched-random instances per holdout date** (instance ids k=1..5, per-date seed `2026000+1000k+date_index`, distinct from 1042/13407/12026 and `ARM_RANDOM_SEED_BASE=70042`, `constants.py:143`), crossed with 3 training seeds = 15 fits; A–D = 12 fits (27 total, ≈0.5–1.5 GPU-h; `results/task5_compute_budget.json`). Per instance/date assert node count, edge count, per-node in/out degree, cross-industry edge count and share, and weight distribution; report the realised swap fraction and across-instance dispersion of P1/P2. Extra control instances reduce control noise only and cannot shift E's expectation.

**Gates before unblinding:** **G1** data contract (`dataset_contract.md`) passes. **G2** pipeline replay — retrain 2016–2021 / validate 2022–2023, rescore 2024–2025; per-arm mean RankIC must reproduce published values within ±0.01 and the C−B return sign. If either gate fails: **STOP**, report the failure, compute no 2026 outcome; a new pre-registration is required.

## 8. Pre-committed interpretation rules
- A contrast is **detected** only if Holm-adjusted p < 0.05 **and** the estimate has the hypothesised sign (H1, H2, H3 all > 0).
- **H3 gates the headline.** If C−E is not detected on P2: report *“no identified cross-industry-specific economic gain”*, whatever C−A/C−B show. If C−E is not detected on P1: *“no identified cross-industry-specific predictive gain”*.
- A cross-industry-value claim requires H3 detection on **both** P1 and P2; C−A/C−B are supporting only.
- If C−E is detected **negative**, report *“the matched random graph outperforms the cross-industry graph”* without reinterpretation.
- Never convert non-detection into “no effect”; always attach the MDE. P1 and P2 are reported side by side, never combined.

## 9. Reported regardless of outcome
Per-date RankIC and weekly returns for all five arms; all three contrasts on both endpoints with raw and Holm p, 95% CIs, share of positive dates, N; all secondaries; seed and instance dispersion; blocked-order counts; three cost scenarios; the MDE table; gate results; a dated deviations log. Every number carries its evidence label.

## 10. Confounds, scope, change control
2026 is one ~10-month regime; external validity is limited. Seeds are optimisation replicates, not independent market periods. The correlation graph is a statistical-relatedness proxy, not supply-chain truth, and is not HIST hidden concepts (`phase7_arms.yaml`, `mechanism_claim_boundary`). Negative autocorrelations are never used to shrink variance. **Amendments before unblinding** require a dated, versioned change with a written reason; **after unblinding** no change to arms, endpoints, family, tests or decision rules is permitted — deviations are reported with their effect. Frozen SHA-256 values and the machine-readable copy are in `preregistration_2026_holdout.json`; the data contract is `dataset_contract.md`.
