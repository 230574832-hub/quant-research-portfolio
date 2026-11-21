from __future__ import annotations

import calendar
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .utils import atomic_write_json, utc_now_iso


def _yyyymmdd(value) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


@dataclass
class CacheStore:
    root: Path
    refresh: bool = False

    def path(self, namespace: str, key: str) -> Path:
        safe_key = key.replace(":", "_").replace("/", "_")
        return self.root / namespace / f"{safe_key}.parquet"

    def get_or_fetch(self, namespace: str, key: str, fetcher: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        path = self.path(namespace, key)
        if path.exists() and not self.refresh:
            return pd.read_parquet(path)
        frame = fetcher()
        if frame is None:
            frame = pd.DataFrame()
        _atomic_parquet(frame, path)
        return frame


class TushareDownloader:
    def __init__(self, pro, cache: CacheStore, config: dict):
        self.pro = pro
        self.cache = cache
        self.pause = float(config["data"].get("api_pause_seconds", 0.12))
        self.max_retries = int(config["data"].get("max_retries", 6))

    def call(self, endpoint: str, **kwargs) -> pd.DataFrame:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                method = getattr(self.pro, endpoint)
                frame = method(**kwargs)
                time.sleep(self.pause)
                return frame if frame is not None else pd.DataFrame()
            except Exception as exc:  # network/API throttling is retried with bounded backoff
                last_error = exc
                time.sleep(min(30.0, 1.5**attempt))
        raise RuntimeError(f"TuShare endpoint {endpoint} failed after {self.max_retries} attempts") from last_error

    def cached_call(self, namespace: str, key: str, endpoint: str, **kwargs) -> pd.DataFrame:
        return self.cache.get_or_fetch(namespace, key, lambda: self.call(endpoint, **kwargs))


def month_ranges(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = pd.date_range(pd.Timestamp(start).replace(day=1), pd.Timestamp(end), freq="MS")
    result = []
    for first in starts:
        last = first.replace(day=calendar.monthrange(first.year, first.month)[1])
        result.append((max(first, pd.Timestamp(start)), min(last, pd.Timestamp(end))))
    return result


def download_reference_data(downloader: TushareDownloader, config: dict, project_root: str | Path) -> dict[str, pd.DataFrame]:
    start = config["data"]["start"]
    end = config["data"]["end"]
    index_code = config["project"]["index_code"]
    benchmark_code = config["project"]["benchmark_code"]
    taxonomy_versions = config["project"].get("industry_taxonomy_versions") or [
        {"source": config["project"].get("industry_source", "SW2021")}
    ]

    trade_cal = downloader.cached_call(
        "reference",
        f"trade_cal_{start}_{end}",
        "trade_cal",
        exchange="SSE",
        start_date=_yyyymmdd(start),
        end_date=_yyyymmdd(end),
        fields="exchange,cal_date,is_open,pretrade_date",
    )

    weight_frames = []
    for first, last in tqdm(month_ranges(start, end), desc="index_weight months"):
        key = f"index_weight_{index_code}_{first:%Y%m}"
        weight_frames.append(
            downloader.cached_call(
                "reference",
                key,
                "index_weight",
                index_code=index_code,
                start_date=_yyyymmdd(first),
                end_date=_yyyymmdd(last),
            )
        )
    index_weight = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    if index_weight.empty:
        raise RuntimeError(f"No index_weight data returned for {index_code}; verify the code and TuShare permission.")

    industry_class_frames = []
    industry_frames = []
    for version in taxonomy_versions:
        src = str(version["source"])
        classification = downloader.cached_call(
            "reference", f"index_classify_L1_{src}", "index_classify", level="L1", src=src
        ).copy()
        if classification.empty or "index_code" not in classification:
            raise RuntimeError(f"No Shenwan L1 industry classification returned for {src}.")
        classification["taxonomy_source"] = src
        classification["taxonomy_applicable_start"] = version.get("applicable_start")
        classification["taxonomy_applicable_end"] = version.get("applicable_end")
        industry_class_frames.append(classification)
        for code in tqdm(classification["index_code"].dropna().unique(), desc=f"{src} industry members"):
            for is_new, suffix in [("N", "historical"), ("Y", "current")]:
                members = downloader.cached_call(
                    "reference",
                    f"index_member_all_{src}_{code}_{suffix}",
                    "index_member_all",
                    l1_code=code,
                    is_new=is_new,
                ).copy()
                members["taxonomy_source"] = src
                members["taxonomy_applicable_start"] = version.get("applicable_start")
                members["taxonomy_applicable_end"] = version.get("applicable_end")
                industry_frames.append(members)
    industry_class = pd.concat(industry_class_frames, ignore_index=True) if industry_class_frames else pd.DataFrame()
    industry_members = pd.concat(industry_frames, ignore_index=True) if industry_frames else pd.DataFrame()
    if industry_members.empty:
        raise RuntimeError("No industry membership data returned; index_member_all access is required.")
    industry_members = industry_members.drop_duplicates(
        subset=[column for column in ["taxonomy_source", "l1_code", "ts_code", "in_date", "out_date"] if column in industry_members]
    )

    benchmark = downloader.cached_call(
        "reference",
        f"index_daily_{benchmark_code}_{start}_{end}",
        "index_daily",
        ts_code=benchmark_code,
        start_date=_yyyymmdd(start),
        end_date=_yyyymmdd(end),
        fields="ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount",
    )
    if benchmark.empty:
        raise RuntimeError(f"No benchmark history returned for {benchmark_code}.")

    reference = {
        "trade_cal": trade_cal,
        "index_weight": index_weight,
        "industry_class": industry_class,
        "industry_members": industry_members,
        "benchmark": benchmark,
    }
    audit = {name: {"rows": int(len(frame)), "columns": list(frame.columns)} for name, frame in reference.items()}
    audit["captured_at_utc"] = utc_now_iso()
    atomic_write_json(Path(project_root) / "tables" / "reference_data_audit.json", audit)
    return reference


def open_trade_dates(trade_cal: pd.DataFrame, start: str, end: str) -> pd.DatetimeIndex:
    frame = trade_cal.copy()
    frame["cal_date"] = pd.to_datetime(frame["cal_date"])
    mask = (frame["is_open"].astype(int) == 1) & frame["cal_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    return pd.DatetimeIndex(frame.loc[mask, "cal_date"].sort_values().unique())


def signal_dates_from_calendar(open_dates: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(open_dates).sort_values()
    series = pd.Series(dates, index=dates)
    # Last open day in each ISO week; this naturally handles holidays.
    return pd.DatetimeIndex(series.groupby(dates.to_period("W-FRI")).max().values)


def build_membership_table(
    index_weight: pd.DataFrame,
    signal_dates: Iterable[pd.Timestamp],
    max_stocks: int | None,
    lag_calendar_days: int = 0,
) -> pd.DataFrame:
    weights = index_weight.rename(columns={"con_code": "ts_code"}).copy()
    weights["snapshot_date"] = pd.to_datetime(weights["trade_date"])
    weights["weight"] = pd.to_numeric(weights.get("weight"), errors="coerce")
    rows = []
    snapshots = sorted(weights["snapshot_date"].dropna().unique())
    for signal_date in pd.DatetimeIndex(signal_dates):
        cutoff = pd.Timestamp(signal_date) - pd.Timedelta(days=int(lag_calendar_days))
        eligible = [date for date in snapshots if pd.Timestamp(date) <= cutoff]
        if not eligible:
            continue
        snapshot = pd.Timestamp(eligible[-1])
        active = weights.loc[weights["snapshot_date"] == snapshot, ["ts_code", "weight"]].copy()
        active = active.sort_values(["weight", "ts_code"], ascending=[False, True])
        if max_stocks:
            active = active.head(int(max_stocks))
        active["signal_date"] = signal_date
        active["index_snapshot_date"] = snapshot
        rows.append(active)
    if not rows:
        raise RuntimeError("No point-in-time index membership could be matched to signal dates.")
    return pd.concat(rows, ignore_index=True)


def _filter_codes(frame: pd.DataFrame, codes: set[str]) -> pd.DataFrame:
    if frame.empty or "ts_code" not in frame.columns:
        return frame
    return frame.loc[frame["ts_code"].isin(codes)].reset_index(drop=True)


def download_market_data(
    downloader: TushareDownloader,
    config: dict,
    reference: dict[str, pd.DataFrame],
    project_root: str | Path,
) -> dict[str, pd.DataFrame]:
    dates = open_trade_dates(reference["trade_cal"], config["data"]["start"], config["data"]["end"])
    signals = signal_dates_from_calendar(dates)
    memberships = build_membership_table(
        reference["index_weight"], signals, config["data"].get("max_stocks_per_snapshot"),
        lag_calendar_days=int(config["data"].get("index_membership_lag_calendar_days", 0)),
    )
    universe_codes = set(memberships["ts_code"].unique())
    fields_map = {
        "daily": config["data"]["daily_fields"],
        "adj_factor": config["data"]["adj_factor_fields"],
        "daily_basic": config["data"]["daily_basic_fields"],
        "stk_limit": config["data"]["limit_fields"],
        "suspend_d": config["data"]["suspend_fields"],
    }
    collected: dict[str, list[pd.DataFrame]] = {endpoint: [] for endpoint in config["data"]["endpoints"]}
    for date in tqdm(dates, desc="market dates"):
        date_key = date.strftime("%Y%m%d")
        for endpoint in config["data"]["endpoints"]:
            kwargs = {"trade_date": date_key, "fields": fields_map[endpoint]}
            if endpoint == "suspend_d":
                kwargs["suspend_type"] = "S"
            frame = downloader.cached_call("raw" + "/" + endpoint, date_key, endpoint, **kwargs)
            collected[endpoint].append(_filter_codes(frame, universe_codes))

    market = {
        endpoint: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for endpoint, frames in collected.items()
    }
    market["memberships"] = memberships
    audit = data_quality_audit({**reference, **market})
    atomic_write_json(Path(project_root) / "tables" / "download_data_quality.json", audit)
    return market


def data_quality_audit(frames: dict[str, pd.DataFrame]) -> dict:
    audit = {"captured_at_utc": utc_now_iso(), "datasets": {}}
    primary_keys = {
        "daily": ["ts_code", "trade_date"],
        "adj_factor": ["ts_code", "trade_date"],
        "daily_basic": ["ts_code", "trade_date"],
        "stk_limit": ["ts_code", "trade_date"],
        "suspend_d": ["ts_code", "trade_date", "suspend_type", "suspend_timing"],
        "benchmark": ["ts_code", "trade_date"],
        "trade_cal": ["exchange", "cal_date"],
        "index_weight": ["index_code", "con_code", "trade_date"],
        "industry_class": ["taxonomy_source", "index_code"],
        "industry_members": ["taxonomy_source", "ts_code", "l1_code", "in_date", "out_date"],
        "memberships": ["signal_date", "ts_code"],
    }
    for name, frame in frames.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        expected_key = primary_keys.get(name, [])
        key_cols = [column for column in expected_key if column in frame.columns]
        complete_key = bool(expected_key) and len(key_cols) == len(expected_key)
        duplicated = frame.duplicated(key_cols, keep=False) if key_cols and complete_key else pd.Series(False, index=frame.index)
        duplicate_groups = int(frame.loc[duplicated].groupby(key_cols, dropna=False).ngroups) if duplicated.any() else 0
        audit["datasets"][name] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "key_columns": key_cols,
            "key_definition_complete": complete_key,
            "duplicate_key_rows": int(duplicated.sum()),
            "duplicate_key_groups": duplicate_groups,
            "exact_duplicate_rows": int(frame.duplicated(keep=False).sum()),
            "null_fraction": {column: round(float(frame[column].isna().mean()), 6) for column in frame.columns},
        }
    return audit


def save_download_checkpoint(
    market: dict[str, pd.DataFrame], reference: dict[str, pd.DataFrame], project_root: str | Path, cache_key: str
) -> Path:
    root = Path(project_root) / "prepared" / f"download_{cache_key}"
    root.mkdir(parents=True, exist_ok=True)
    for name, frame in {**reference, **market}.items():
        _atomic_parquet(frame, root / f"{name}.parquet")
    atomic_write_json(root / "_SUCCESS.json", {"completed_at_utc": utc_now_iso(), "cache_key": cache_key})
    return root


def load_download_checkpoint(project_root: str | Path, cache_key: str) -> dict[str, pd.DataFrame] | None:
    root = Path(project_root) / "prepared" / f"download_{cache_key}"
    if not (root / "_SUCCESS.json").exists():
        return None
    return {path.stem: pd.read_parquet(path) for path in root.glob("*.parquet")}

