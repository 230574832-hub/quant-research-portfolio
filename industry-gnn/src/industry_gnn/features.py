from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import atomic_write_json, utc_now_iso


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def _dedupe(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    available = [key for key in keys if key in frame.columns]
    return frame.sort_values(available).drop_duplicates(available, keep="last") if available else frame


def make_daily_feature_frame(
    market: dict[str, pd.DataFrame], benchmark: pd.DataFrame, config: dict | None = None
) -> pd.DataFrame:
    daily = _dedupe(market["daily"].copy(), ["ts_code", "trade_date"])
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    for column in ["open", "high", "low", "close", "pre_close", "vol", "amount"]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")

    adj = _dedupe(market["adj_factor"].copy(), ["ts_code", "trade_date"])
    adj["trade_date"] = pd.to_datetime(adj["trade_date"])
    adj["adj_factor"] = pd.to_numeric(adj["adj_factor"], errors="coerce")
    frame = daily.merge(adj[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")

    basic = _dedupe(market["daily_basic"].copy(), ["ts_code", "trade_date"])
    basic["trade_date"] = pd.to_datetime(basic["trade_date"])
    basic_columns = ["ts_code", "trade_date", "turnover_rate", "volume_ratio", "pb", "total_mv", "circ_mv"]
    basic = basic[[c for c in basic_columns if c in basic]].sort_values(["ts_code", "trade_date"])
    lag = int((config or {}).get("data", {}).get("daily_basic_lag_trading_days", 0))
    value_columns = [column for column in basic_columns[2:] if column in basic]
    basic["daily_basic_source_date"] = basic.groupby("ts_code")["trade_date"].shift(lag)
    if lag:
        basic[value_columns] = basic.groupby("ts_code")[value_columns].shift(lag)
    frame = frame.merge(basic, on=["ts_code", "trade_date"], how="left")

    frame["adj_open"] = frame["open"] * frame["adj_factor"]
    frame["adj_close"] = frame["close"] * frame["adj_factor"]
    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    benchmark = benchmark.copy()
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    benchmark["benchmark_open"] = pd.to_numeric(benchmark["open"], errors="coerce")
    benchmark["benchmark_close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    benchmark = benchmark.sort_values("trade_date")
    benchmark["benchmark_ret_1"] = benchmark["benchmark_close"].pct_change(fill_method=None)
    frame = frame.merge(
        benchmark[["trade_date", "benchmark_open", "benchmark_close", "benchmark_ret_1"]],
        on="trade_date",
        how="left",
    )

    groups = frame.groupby("ts_code", group_keys=False)
    for horizon in [1, 5, 20, 60]:
        frame[f"ret_{horizon}"] = groups["adj_close"].pct_change(horizon, fill_method=None)
    frame["volatility_20"] = groups["ret_1"].transform(
        lambda series: series.rolling(20, min_periods=15).std() * np.sqrt(252)
    )
    frame["volume_change_5"] = groups["vol"].transform(
        lambda series: np.log1p(series).diff(5)
    )
    frame["amount_change_5"] = groups["amount"].transform(
        lambda series: np.log1p(series).diff(5)
    )
    frame["log_total_mv"] = np.log(frame.get("total_mv", np.nan).where(frame.get("total_mv", np.nan) > 0))

    beta_parts = []
    for _, stock in frame.groupby("ts_code", sort=False):
        stock = stock.copy()
        covariance = stock["ret_1"].rolling(60, min_periods=40).cov(stock["benchmark_ret_1"])
        market_variance = stock["benchmark_ret_1"].rolling(60, min_periods=40).var()
        stock["beta_60"] = covariance / market_variance.replace(0, np.nan)
        beta_parts.append(stock)
    return pd.concat(beta_parts, ignore_index=True) if beta_parts else frame


def assign_industry_asof(
    rows: pd.DataFrame,
    industry_members: pd.DataFrame,
    lag_days: int = 1,
    taxonomy_versions: list[dict] | None = None,
    strict_taxonomy: bool = False,
) -> pd.DataFrame:
    members = industry_members.copy()
    members["in_date"] = pd.to_datetime(members["in_date"], errors="coerce")
    members["out_date"] = pd.to_datetime(members.get("out_date"), errors="coerce")
    members["known_from"] = members["in_date"] + pd.Timedelta(days=int(lag_days))
    result = []
    for signal_date, section in rows.groupby("signal_date", sort=True):
        version_source = None
        date_members = members
        if taxonomy_versions:
            applicable = []
            for version in taxonomy_versions:
                start = pd.Timestamp(version["applicable_start"])
                end = pd.Timestamp(version["applicable_end"]) if version.get("applicable_end") else pd.Timestamp.max
                if start <= pd.Timestamp(signal_date) <= end:
                    applicable.append(version)
            if len(applicable) != 1:
                message = f"Signal date {pd.Timestamp(signal_date).date()} maps to {len(applicable)} taxonomy versions."
                if strict_taxonomy:
                    raise RuntimeError(message)
                version_source = None
            else:
                version_source = str(applicable[0]["source"])
                if "taxonomy_source" not in members:
                    if strict_taxonomy:
                        raise RuntimeError("Versioned taxonomy requested but taxonomy_source is absent from industry data.")
                else:
                    date_members = members.loc[members["taxonomy_source"].astype(str) == version_source]
                    if strict_taxonomy and date_members.empty:
                        raise RuntimeError(f"No industry membership records were downloaded for {version_source}.")
        active = date_members.loc[
            (date_members["known_from"] <= signal_date)
            & (date_members["out_date"].isna() | (date_members["out_date"] > signal_date))
        ].copy()
        active = active.sort_values(["ts_code", "in_date"]).drop_duplicates("ts_code", keep="last")
        columns = [column for column in ["ts_code", "l1_code", "l1_name", "in_date", "out_date"] if column in active]
        merged = section.merge(active[columns], on="ts_code", how="left")
        merged["industry_taxonomy_source"] = version_source or "legacy_unspecified"
        result.append(merged)
    return pd.concat(result, ignore_index=True) if result else rows.assign(l1_code=np.nan)


def _weekly_schedule(signal_dates: Iterable[pd.Timestamp], open_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    signals = pd.DatetimeIndex(signal_dates).sort_values()
    opens = pd.DatetimeIndex(open_dates).sort_values()
    rows = []
    for idx, signal in enumerate(signals[:-1]):
        entry_options = opens[opens > signal]
        exit_options = opens[opens > signals[idx + 1]]
        if len(entry_options) and len(exit_options):
            rows.append(
                {
                    "signal_date": signal,
                    "entry_date": entry_options[0],
                    "exit_date": exit_options[0],
                    "next_signal_date": signals[idx + 1],
                }
            )
    return pd.DataFrame(rows)


def _cross_sectional_transform(panel: pd.DataFrame, features: list[str], lower: float, upper: float) -> pd.DataFrame:
    output = panel.copy()
    for _, indices in output.groupby("signal_date").groups.items():
        index = list(indices)
        for feature in features:
            values = pd.to_numeric(output.loc[index, feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = values.median()
            values = values.fillna(0.0 if pd.isna(median) else median)
            lo, hi = values.quantile([lower, upper])
            values = values.clip(lo, hi)
            std = values.std(ddof=0)
            output.loc[index, feature] = (values - values.mean()) / (std if std and np.isfinite(std) else 1.0)
    return output


def _add_peer_features(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    output = panel.copy()
    has_industry = output["l1_code"].notna()
    valid_industry = output["l1_code"].fillna("UNKNOWN")
    group_keys = [output["signal_date"], valid_industry]
    for feature in features:
        totals = output[feature].groupby(group_keys).transform("sum")
        counts = output[feature].groupby(group_keys).transform("count")
        output[f"peer_{feature}"] = np.where(
            has_industry & (counts > 1), (totals - output[feature]) / (counts - 1), 0.0
        )
    return output


def _assign_split(panel: pd.DataFrame, splits: dict) -> pd.DataFrame:
    output = panel.copy()
    output["split"] = "excluded"
    definitions = {
        "train": (splits["train_start"], splits["train_end"]),
        "validation": (splits["validation_start"], splits["validation_end"]),
        "test": (splits["test_start"], splits["test_end"]),
    }
    for name, (start, end) in definitions.items():
        mask = output["signal_date"].between(pd.Timestamp(start), pd.Timestamp(end))
        if splits.get("purge_labels_crossing_boundaries", True):
            mask &= output["exit_date"] <= pd.Timestamp(end)
        output.loc[mask, "split"] = name
    # --- disclosed rebuild patch (preregistration CSI300_INDUSTRY_GNN_HOLDOUT_2026) ---
    # The frozen source drops every post-2025 signal date as "excluded"; the 2026 holdout
    # needs an explicit, label-purged split.  No other line of the feature pipeline changes.
    holdout_start = splits.get("holdout_start")
    holdout_end = splits.get("holdout_end")
    if holdout_start and holdout_end:
        mask = output["signal_date"].between(pd.Timestamp(holdout_start), pd.Timestamp(holdout_end))
        if splits.get("purge_labels_crossing_boundaries", True):
            mask &= output["exit_date"] <= pd.Timestamp(holdout_end)
        output.loc[mask, "split"] = "holdout"
    return output


def make_execution_prices(market: dict[str, pd.DataFrame]) -> pd.DataFrame:
    daily = _dedupe(market["daily"].copy(), ["ts_code", "trade_date"])
    adj = _dedupe(market["adj_factor"].copy(), ["ts_code", "trade_date"])
    limits = _dedupe(market["stk_limit"].copy(), ["ts_code", "trade_date"])
    suspends = _dedupe(market["suspend_d"].copy(), ["ts_code", "trade_date"])
    for frame in [daily, adj, limits, suspends]:
        if "trade_date" in frame:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    result = daily.merge(adj[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")
    limit_cols = [column for column in ["ts_code", "trade_date", "up_limit", "down_limit"] if column in limits]
    result = result.merge(limits[limit_cols], on=["ts_code", "trade_date"], how="left")
    suspended_keys = set(zip(suspends.get("ts_code", []), suspends.get("trade_date", [])))
    result["suspended"] = [
        (code, date) in suspended_keys for code, date in zip(result["ts_code"], result["trade_date"])
    ]
    for column in ["open", "adj_factor", "up_limit", "down_limit"]:
        result[column] = pd.to_numeric(result.get(column), errors="coerce")
    result["adj_open"] = result["open"] * result["adj_factor"]
    # At the opening auction, the day's eventual high/low are not observable.  In
    # the absence of order-book data, conservatively block an opening buy (sell)
    # whenever the opening price is already at the published upper (lower) limit.
    result["open_at_upper_limit"] = result["up_limit"].notna() & np.isclose(
        result["open"], result["up_limit"], rtol=0, atol=0.011
    )
    result["open_at_lower_limit"] = result["down_limit"].notna() & np.isclose(
        result["open"], result["down_limit"], rtol=0, atol=0.011
    )
    has_quote = result["adj_open"].notna() & (result["adj_open"] > 0)
    result["buyable"] = has_quote & ~result["suspended"] & ~result["open_at_upper_limit"]
    result["sellable"] = has_quote & ~result["suspended"] & ~result["open_at_lower_limit"]
    return result[[
        "ts_code", "trade_date", "adj_open", "buyable", "sellable", "suspended",
        "open_at_upper_limit", "open_at_lower_limit",
    ]].sort_values(
        ["trade_date", "ts_code"]
    )


def prepare_research_panel(
    market: dict[str, pd.DataFrame], reference: dict[str, pd.DataFrame], config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    daily_features = make_daily_feature_frame(market, reference["benchmark"], config)
    memberships = market["memberships"].copy()
    memberships["signal_date"] = pd.to_datetime(memberships["signal_date"])
    signal_dates = pd.DatetimeIndex(memberships["signal_date"].unique()).sort_values()
    trade_cal = reference["trade_cal"].copy()
    trade_cal["cal_date"] = pd.to_datetime(trade_cal["cal_date"])
    open_dates = pd.DatetimeIndex(trade_cal.loc[trade_cal["is_open"].astype(int) == 1, "cal_date"].sort_values().unique())
    schedule = _weekly_schedule(signal_dates, open_dates)

    panel = memberships.merge(
        daily_features,
        left_on=["ts_code", "signal_date"],
        right_on=["ts_code", "trade_date"],
        how="inner",
    ).drop(columns=["trade_date"])
    panel = panel.merge(schedule, on="signal_date", how="inner")
    panel = assign_industry_asof(
        panel,
        reference["industry_members"],
        lag_days=int(config["project"].get("industry_effective_lag_days", 1)),
        taxonomy_versions=config["project"].get("industry_taxonomy_versions"),
        strict_taxonomy=bool(config["project"].get("strict_industry_taxonomy", False)),
    )
    panel["industry_relative_ret_20"] = panel["ret_20"] - panel.groupby(
        ["signal_date", panel["l1_code"].fillna("UNKNOWN")]
    )["ret_20"].transform("mean")

    execution = make_execution_prices(market)
    entry = execution.rename(columns={
        "trade_date": "entry_date", "adj_open": "entry_adj_open", "buyable": "entry_buyable",
        "sellable": "entry_sellable", "suspended": "entry_suspended",
        "open_at_upper_limit": "entry_open_at_upper_limit",
        "open_at_lower_limit": "entry_open_at_lower_limit",
    })
    exit_frame = execution.rename(columns={
        "trade_date": "exit_date", "adj_open": "exit_adj_open", "buyable": "exit_buyable",
        "sellable": "exit_sellable", "suspended": "exit_suspended",
        "open_at_upper_limit": "exit_open_at_upper_limit",
        "open_at_lower_limit": "exit_open_at_lower_limit",
    })
    panel = panel.merge(entry, on=["ts_code", "entry_date"], how="left")
    panel = panel.merge(exit_frame, on=["ts_code", "exit_date"], how="left")
    # Labels are attached for supervised fitting/evaluation, but their availability must never
    # determine the signal-date prediction universe. A missing future open therefore remains a
    # row with label_available=False and is still predicted/backtested with contemporaneous rules.
    panel["target_return"] = panel["exit_adj_open"] / panel["entry_adj_open"] - 1.0

    benchmark = reference["benchmark"].copy()
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    benchmark["open"] = pd.to_numeric(benchmark["open"], errors="coerce")
    benchmark_entry = benchmark[["trade_date", "open"]].rename(columns={"trade_date": "entry_date", "open": "benchmark_entry_open"})
    benchmark_exit = benchmark[["trade_date", "open"]].rename(columns={"trade_date": "exit_date", "open": "benchmark_exit_open"})
    panel = panel.merge(benchmark_entry, on="entry_date", how="left").merge(benchmark_exit, on="exit_date", how="left")
    panel["benchmark_return"] = panel["benchmark_exit_open"] / panel["benchmark_entry_open"] - 1.0
    panel["target_excess_return"] = panel["target_return"] - panel["benchmark_return"]

    features = list(config["features"]["raw"])
    panel["entry_price_available"] = panel["entry_adj_open"].notna()
    panel["exit_price_available"] = panel["exit_adj_open"].notna()
    panel["label_available"] = panel["target_excess_return"].notna()
    # These rolling factors are observable at signal formation and enforce the configured warmup.
    # Crucially, entry/exit prices and label availability are not used here.
    panel = panel.dropna(subset=["ret_60", "beta_60"])
    panel["decision_eligible"] = True
    panel = panel.groupby("signal_date").filter(lambda section: len(section) >= config["features"]["min_cross_section"])
    panel = _cross_sectional_transform(
        panel,
        features,
        float(config["features"]["winsor_lower"]),
        float(config["features"]["winsor_upper"]),
    )
    panel = _add_peer_features(panel, features)
    panel = _assign_split(panel, config["splits"])
    panel = panel.loc[panel["split"] != "excluded"].copy()
    panel = panel.sort_values(["signal_date", "ts_code"]).reset_index(drop=True)
    panel["row_id"] = np.arange(len(panel), dtype=np.int64)

    benchmark_prices = benchmark[["trade_date", "open"]].rename(
        columns={"trade_date": "date", "open": "benchmark_open"}
    ).sort_values("date")
    metadata = {
        "created_at_utc": utc_now_iso(),
        "rows": int(len(panel)),
        "dates": int(panel["signal_date"].nunique()),
        "stocks": int(panel["ts_code"].nunique()),
        "feature_columns": features,
        "peer_feature_columns": [f"peer_{feature}" for feature in features],
        "split_rows": panel["split"].value_counts().to_dict(),
        "split_labeled_rows": panel.loc[panel["label_available"], "split"].value_counts().to_dict(),
        "label_available_fraction": float(panel["label_available"].mean()),
        "entry_price_available_fraction": float(panel["entry_price_available"].mean()),
        "exit_price_available_fraction": float(panel["exit_price_available"].mean()),
        "future_price_filter_applied": False,
        "missing_industry_fraction": float(panel["l1_code"].isna().mean()),
        "industry_taxonomy_policy": config["project"].get("industry_taxonomy_policy", "single_retrospective"),
        "industry_taxonomy_versions": config["project"].get("industry_taxonomy_versions", []),
        "industry_taxonomy_rows": panel["industry_taxonomy_source"].value_counts(dropna=False).to_dict(),
        "execution_tradeability_rule": "opening quote + suspension status + opening price versus published limit; no same-day high/low",
        "daily_basic_lag_trading_days": int(config["data"].get("daily_basic_lag_trading_days", 0)),
        "index_membership_lag_calendar_days": int(config["data"].get("index_membership_lag_calendar_days", 0)),
        "point_in_time_notes": {
            "index_membership": "latest index_weight snapshot dated on or before signal_date",
            "industry_membership": "date-applicable SW2014/SW2021 taxonomy plus in_date/out_date interval and configured lag; source revision timestamps remain unavailable",
            "features": "same-date cross-sectional imputation/winsorization/z-score after market close",
            "universe": "signal-date membership and signal-date factors only; future price/label availability never filters prediction rows",
            "label": "next-open to next-rebalance-open adjusted return; excess of benchmark open-to-open return; missing labels are masked only in fitting/evaluation",
            "split_purge": bool(config["splits"].get("purge_labels_crossing_boundaries", True)),
        },
    }
    return panel, execution, benchmark_prices, metadata


def save_prepared_checkpoint(
    panel: pd.DataFrame,
    execution: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    metadata: dict,
    project_root: str | Path,
    cache_key: str,
) -> Path:
    root = Path(project_root) / "prepared" / f"panel_{cache_key}"
    root.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(panel, root / "panel.parquet")
    _atomic_parquet(execution, root / "execution_prices.parquet")
    _atomic_parquet(benchmark_prices, root / "benchmark_prices.parquet")
    atomic_write_json(root / "metadata.json", metadata)
    atomic_write_json(root / "_SUCCESS.json", {"cache_key": cache_key, "completed_at_utc": utc_now_iso()})
    return root


def load_prepared_checkpoint(project_root: str | Path, cache_key: str):
    root = Path(project_root) / "prepared" / f"panel_{cache_key}"
    if not (root / "_SUCCESS.json").exists():
        return None
    metadata = __import__("json").loads((root / "metadata.json").read_text(encoding="utf-8"))
    return (
        pd.read_parquet(root / "panel.parquet"),
        pd.read_parquet(root / "execution_prices.parquet"),
        pd.read_parquet(root / "benchmark_prices.parquet"),
        metadata,
    )

