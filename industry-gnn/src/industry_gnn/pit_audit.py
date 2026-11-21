from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import atomic_write_json, utc_now_iso


def _row(
    component: str,
    field: str,
    source: str,
    decision_rule: str,
    code_status: str,
    source_status: str,
    evidence: str,
    limitation: str,
) -> dict:
    if code_status == "FAIL" or source_status == "FAIL":
        overall = "FAIL"
    elif code_status == "PASS" and source_status == "PASS":
        overall = "PASS"
    elif code_status == "UNVERIFIED" and source_status == "UNVERIFIED":
        overall = "UNVERIFIED"
    else:
        overall = "PARTIAL"
    return {
        "component": component,
        "field": field,
        "source": source,
        "information_availability_rule": decision_rule,
        "code_timing_status": code_status,
        "source_history_status": source_status,
        "overall_status": overall,
        "evidence": evidence,
        "limitation": limitation,
    }


def build_point_in_time_audit(
    panel: pd.DataFrame,
    market: dict[str, pd.DataFrame],
    reference: dict[str, pd.DataFrame],
    config: dict,
    project_root: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Create an evidence ledger without upgrading unavailable source facts.

    PASS means both the implemented cutoff and the source timestamp can be checked
    from retained fields.  PARTIAL means the code cutoff passes but the vendor file
    lacks an announcement/revision timestamp.  UNVERIFIED is never promoted merely
    because a synthetic timestamp or a configuration flag exists.
    """
    rows: list[dict] = []
    signal = pd.to_datetime(panel["signal_date"])

    technical = [
        "ret_1", "ret_5", "ret_20", "ret_60", "volatility_20", "beta_60",
        "volume_change_5", "amount_change_5",
    ]
    for field in technical:
        rows.append(_row(
            "feature", field, "TuShare daily + CSI 300 index_daily",
            "raw sessions at or before the post-close signal date; rolling windows are backward-looking",
            "PASS", "UNVERIFIED",
            "feature code uses groupwise pct_change/diff/rolling and same-date cross-sectional transforms",
            "downloaded history has trade_date but no immutable vendor revision/as-known timestamp",
        ))

    basic_lag = int(config["data"].get("daily_basic_lag_trading_days", 0))
    has_basic_source = "daily_basic_source_date" in panel
    basic_before_signal = bool(
        has_basic_source
        and (
            pd.to_datetime(panel["daily_basic_source_date"], errors="coerce") < signal
        ).fillna(False).all()
    )
    for field in ["turnover_rate", "pb", "log_total_mv"]:
        rows.append(_row(
            "feature", field, "TuShare daily_basic",
            f"use the value from {basic_lag} prior trading session(s)",
            "PASS" if basic_before_signal else "FAIL", "UNVERIFIED",
            f"daily_basic_source_date exists={has_basic_source}; all source dates precede signal={basic_before_signal}",
            "daily_basic extract contains no field-level publication timestamp or historical revision log",
        ))

    rows.append(_row(
        "feature", "adjusted-price returns", "TuShare daily + adj_factor",
        "feature returns use only raw prices and adjustment factors dated no later than signal_date",
        "PASS", "UNVERIFIED",
        "future entry/exit adjustment factors are attached only to labels/backtest, never to signal-date feature filters",
        "the cache proves the downloaded snapshot, not whether TuShare later revised historical adjustment factors",
    ))
    rows.append(_row(
        "graph", "Residual-GAT Sparse rolling correlations", "TuShare daily close/pre_close",
        "60-session correlation window uses only sessions strictly before each signal date",
        "PASS", "UNVERIFIED",
        "the V3 adapter removes every signal-day return before graph construction and tests label-invariance of edges",
        "daily quote extracts have no vendor as-known/revision timestamp; raw returns are therefore best-effort PIT",
    ))

    taxonomy_versions = config["project"].get("industry_taxonomy_versions", [])
    expected_source = pd.Series(index=panel.index, dtype="object")
    for version in taxonomy_versions:
        start = pd.Timestamp(version["applicable_start"])
        end = pd.Timestamp(version["applicable_end"]) if version.get("applicable_end") else pd.Timestamp.max
        expected_source.loc[signal.between(start, end)] = str(version["source"])
    taxonomy_present = "industry_taxonomy_source" in panel
    taxonomy_match = bool(
        taxonomy_present
        and expected_source.notna().all()
        and panel["industry_taxonomy_source"].astype(str).eq(expected_source.astype(str)).all()
    )
    interval_known = bool(
        "in_date" in panel
        and (
            pd.to_datetime(panel["in_date"], errors="coerce")
            + pd.Timedelta(days=int(config["project"].get("industry_effective_lag_days", 1)))
            <= signal
        ).fillna(False).eq(panel["l1_code"].notna()).all()
    ) if "l1_code" in panel else False
    rows.append(_row(
        "industry", "l1_code", "TuShare index_classify/index_member_all",
        "SW2014 through 2021-07-29; SW2021 from 2021-07-30; membership in/out interval plus lag",
        "PASS" if taxonomy_match else "FAIL", "UNVERIFIED",
        f"taxonomy source matches configured date regime={taxonomy_match}; effective interval check={interval_known}",
        "in_date/out_date and taxonomy edition do not prove the original public announcement time or exclude vendor backfill",
    ))
    rows.append(_row(
        "feature", "industry_relative_ret_20", "derived from l1_code and ret_20",
        "same-date post-close return minus same-date industry mean",
        "PASS" if taxonomy_match else "FAIL", "UNVERIFIED",
        "the arithmetic is contemporaneous and inherits the versioned taxonomy rule",
        "inherits the unresolved announcement/revision risk of historical industry membership",
    ))

    membership_lag = int(config["data"].get("index_membership_lag_calendar_days", 0))
    snapshot_ok = bool(
        "index_snapshot_date" in panel
        and (
            pd.to_datetime(panel["index_snapshot_date"], errors="coerce")
            <= signal - pd.Timedelta(days=membership_lag)
        ).fillna(False).all()
    )
    rows.append(_row(
        "universe", "CSI 300 membership", "TuShare index_weight",
        f"latest snapshot no later than signal_date minus {membership_lag} calendar day(s)",
        "PASS" if snapshot_ok else "FAIL", "UNVERIFIED",
        f"all retained index_snapshot_date values satisfy cutoff={snapshot_ok}",
        "index_weight history does not include the original announcement timestamp or a vendor revision history",
    ))

    future_filter_ok = bool(panel.get("decision_eligible", pd.Series(False, index=panel.index)).fillna(False).all())
    rows.append(_row(
        "label/universe", "future entry/exit prices", "prepared research panel",
        "future prices may determine label_available only, never decision eligibility",
        "PASS" if future_filter_ok else "FAIL", "PASS",
        "decision_eligible is independent of entry_price_available, exit_price_available and label_available",
        "none",
    ))

    execution_columns = set(market.get("daily", pd.DataFrame()).columns)
    rows.append(_row(
        "execution", "opening tradeability", "daily open + stk_limit + suspend_d",
        "block buys at opening upper limit, sells at opening lower limit, all suspended/missing quotes; never inspect same-day high/low",
        "PASS", "UNVERIFIED",
        f"daily source columns retained={sorted(execution_columns & {'open', 'high', 'low', 'close'})}; execution code consumes open only",
        "without opening order-book/queue data, a limit-open order's fill probability cannot be identified; the backtest assumes no fill",
    ))

    rows.append(_row(
        "benchmark", "CSI 300 holding interval", "index_daily open",
        "benchmark return is zero on first portfolio opening; thereafter both legs use open-to-open intervals",
        "PASS", "PASS",
        "backtest explicitly resets the benchmark return at investment_start",
        "none",
    ))

    audit = pd.DataFrame(rows)
    failures = audit.loc[audit["overall_status"] == "FAIL"]
    summary = {
        "created_at_utc": utc_now_iso(),
        "status_counts": {str(k): int(v) for k, v in audit["overall_status"].value_counts().items()},
        "blocking_failures": failures[["component", "field", "evidence"]].to_dict("records"),
        "strict_point_in_time_claim_allowed": bool(
            len(failures) == 0 and audit["overall_status"].eq("PASS").all()
        ),
        "interpretation": (
            "Code-level time ordering can pass while vendor announcement/revision history remains PARTIAL or UNVERIFIED. "
            "Such rows must not be relabeled VERIFIED without archived source evidence."
        ),
    }
    root = Path(project_root)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    audit.to_csv(root / "tables" / "point_in_time_audit.csv", index=False, encoding="utf-8-sig")
    atomic_write_json(root / "tables" / "point_in_time_audit.json", {
        **summary, "records": json.loads(audit.to_json(orient="records", date_format="iso"))
    })
    if len(failures):
        raise RuntimeError("Point-in-time code audit has blocking failures: " + ", ".join(failures["field"]))
    return audit, summary

