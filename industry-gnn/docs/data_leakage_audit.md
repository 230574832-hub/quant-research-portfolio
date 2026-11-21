# Data leakage audit

This status is based on source inspection and synthetic tests in revision 0.4.0. Real TuShare data must still pass the laptop preflight artifacts.

| Risk | Control | Status / residual limitation |
|---|---|---|
| Future exit price filters the universe | `decision_eligible` uses signal-date membership plus signal-date history. Entry/exit availability only sets `label_available`; unlabeled rows remain in the prediction universe. | Implemented and regression-tested with a missing future quote. |
| Survivorship bias | CSI 300 membership uses the latest historical index-weight snapshot not after each signal date, rather than today’s constituents. | Implemented; accuracy depends on TuShare historical snapshots. |
| Industry look-ahead | Membership uses historical `in_date`/`out_date` intervals plus a one-calendar-day effective lag. | Implemented; TuShare does not expose a separate historical announcement timestamp here, so residual risk is disclosed. |
| Adjustment-factor look-ahead | Returns use each date’s downloaded factor, never a factor joined from a future row. | Date-local; TuShare may revise history after corporate actions, so retain data fingerprints and cache timestamp. |
| Fundamental timing | The version uses market-observable `daily_basic` fields on the signal date, not quarterly statements or future report announcements. | No statement endpoint is used. Same-day publication latency remains a source-data assumption. |
| Standardization leakage | Winsorization, median fill, and z-score are calculated within each signal-date cross-section. | Implemented and tested. The full contemporaneously observable universe is intentional. |
| Label/price mismatch | Target is adjusted next-open to following-rebalance-open excess return, aligned with the simulator’s execution convention. | Missing labels are excluded from fitting/statistics, not ranking. |
| Split leakage | Chronological splits purge observations whose exit crosses split end. Test is 2024–2025. | Implemented and tested. 2022–2023 is development, not independent evidence. |
| Hyperparameter/test leakage | Width/alpha and stopping use only training and validation. No train+validation refit is performed. | Frozen. Test-driven iteration requires a new protocol ID. |
| Cache contamination | Model key incorporates source, config and data fingerprints; data/prepared keys include dates and feature schema; sidecars gate reuse. | Implemented. Old checkpoints with another hash are ignored. |

The audit cannot prove that a vendor never retrospectively revised a record. Retain `data_file_fingerprints.json`, `experiment_config.yaml`, and the cache so the exact downloaded version is traceable.
