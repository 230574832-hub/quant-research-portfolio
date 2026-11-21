# Holdout deviations log (post-unblinding)

Protocol: `CSI300_INDUSTRY_GNN_HOLDOUT_2026`, v1.0 + amendment v1.1.
Prediction freeze: `holdout_predictions_by_instance.parquet`, sha256 `8e9a2399f210b8cf…`
(written before labels were joined).

| time (UTC) | deviation | effect on results | disposition |
|---|---|---|---|
| 2026-11-11T11:26Z | First implementation grouped portfolio daily returns into **42 calendar weeks** (W-FRI) instead of the 41 holding periods defined by the signal dates | moved the P2 sample from 41 to 42, mixing one partial week; C−E changed by <0.0002 per period | corrected before reporting: `period_net()` keys returns by entry date; final files carry `periods=41` |
| 2026-11-11T11:27Z | **G2 outcome-direction check failed**: retrained C − B annualised return on 2024–2025 is −1.88 pp, while the published frozen value is +16.78 pp; RankIC reproduces within 0.0007–0.0101 | per the amendment, G2 is technical only; the sign flip is reported, not used to gate | reported as a finding: economic ordering is not reproducible from RankIC proximity alone |
| 2026-11-11T11:31Z | Added a **post-hoc Ridge baseline** (12-factor OLS, alpha selected on validation) as a descriptive comparator | it reproduces 2024–2025 exactly (132.80 %, Sharpe 1.414, RankIC 0.03097) and loses 21.1 % in 2026 | labelled `POST_HOC_DESCRIPTIVE`, outside the pre-registered family |
| 2026-11-11T11:31Z | Random-graph arm E: 5 instances × 3 training seeds; the 15-fit ensemble is supplementary | primary E = mean of five instance-level estimates; instance dispersion reported | consistent with the v1.1 amendment §A |

## Not deviations
- No hypothesis, arm definition, split, seed, cost or window changed after the prediction file
  was hashed.
- The 2024–2025 window remains `EXPLORATORY_POST_HOC`; 2026 remains a frozen historical holdout
  whose labels were first joined after the hash, and is **not** described as prospective.
