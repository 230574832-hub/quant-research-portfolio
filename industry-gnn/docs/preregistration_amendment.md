# Amendment v1.1 — written and frozen BEFORE any 2026 outcome is joined

- **Protocol:** `CSI300_INDUSTRY_GNN_HOLDOUT_2026` · **Base version:** 1.0 (2026-11-11T10:08:28Z)
- **Amendment time (UTC):** 2026-11-11T11:05:00Z (before the holdout predictions file exists; see
  `reports/holdout/holdout_predictions_manifest.json` written later)
- **Trigger:** independent review of the first evaluation/training draft. All changes below tighten
  the estimand or remove a degree of freedom; none changes the hypotheses, arms, splits, seeds or costs.
- **What did NOT change:** H1 = C−A, H2 = C−B, H3 = C−E; the five arm definitions; train 2016–2021 /
  validation 2022–2023; training seeds 42/3407/2026; Top-10, base cost, conservative open execution;
  the 2026-01-16..2026-10-30 window; the `EXPLORATORY` label on 2024–2025.

## A. Random-graph arm E: per-instance aggregation is now the primary rule
The base text said "three-seed-mean scores" without stating how the 5 random-graph instances combine.
The first draft averaged all 15 E fits into one prediction vector before ranking, while arm C averaged
only 3 training seeds. That would confound **graph randomness** with **ensemble size**, and ranking +
Top-10 selection is nonlinear, so it can change expected strategy performance.

**Amended primary rule.** For each random-graph instance k = 1..5: average its 3 training seeds →
one prediction vector → rank → Top-10 → backtest. The primary E estimate is the mean of the five
instance-level estimates, with instance dispersion reported. Pairwise C−E is formed per instance and
averaged per week before the moving-block bootstrap.
**Supplementary (robustness only):** the 15-fit ensemble prediction.

## B. P2 estimand: mean weekly net-return difference is primary
The base text named P2 "annualised net return" while the statistical plan used
`52 × mean(r_C − r_E)`, which is an arithmetic annualisation of a weekly difference, not the
difference of two compounded annualised returns. Different objects were being named together.

**Amended.** Primary P2 estimand = **mean weekly net-return difference** with its moving-block 95% CI.
Arithmetic annualisation (`52 × estimate`) is reported as a derived quantity; compounded net returns,
Sharpe and drawdown are descriptive. The MDE table keeps its original arithmetic but is now explicitly
labelled as an arithmetic annualisation.

## C. G2 is an implementation-consistency gate, not an outcome-direction gate
The base text required "per-arm mean RankIC within ±0.01 **and** the C−B return sign". Tying a gate to
an outcome direction lets the analyst iterate implementations until the old sign reappears.

**Amended.** G2 passes iff (i) the input fingerprints are the frozen ones, (ii) the rebuilt graphs and
training curves are structurally consistent, and (iii) per-arm 2024–2025 RankIC is within **±0.02** of
the published exploratory values. The C−B sign is reported for information only.

## D. Power disclosure
The MDE table (39–51 pp annualised at N=40) assumes a single-contrast α=0.05 and does not embed the
Holm correction over three comparisons; with N=41 the realised MDE is not smaller. The holdout is a
low-power direction/interval check, not a definitive test. A null is not evidence of no effect.

## E. Evidence-label wording
The 2026 window is described as a **frozen historical holdout validation whose outcomes were first
unlocked after 2026-11-11**, not as "prospective" or "untouched by anyone". The vendor data for
2026 already existed; what the freeze guarantees is that *this* analysis pipeline had not joined those
labels before the prediction file was hashed. Any earlier contact with 2026 strategy outcomes by other
tools must be disclosed separately; none is known to this pipeline.

## F. Acceptance gates
Gates A1–A14 are implemented as specified: A2 now checks train/validation/test **exit** dates; A4
freezes once and afterwards only compares read-only; A9 checks the 12 peer features as well as the 12
raw features; A10 validates the real as-of window per signal date; A14 re-hashes and compares; any
failed gate exits non-zero. Results: `reports/holdout/holdout_acceptance_v2.json`.

## G. Reporting commitments unaffected
All pre-committed interpretation rules, the H3 headline rule, the reporting list and the change-control
policy remain as in v1.0 §8–§10. This amendment is dated and hashed before unblinding.
