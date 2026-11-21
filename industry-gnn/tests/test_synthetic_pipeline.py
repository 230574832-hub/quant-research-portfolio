from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from industry_gnn.data import build_membership_table, signal_dates_from_calendar
from industry_gnn.features import prepare_research_panel
from industry_gnn.training import train_ridge
from industry_gnn.utils import load_config


def test_synthetic_feature_label_split_and_ridge(tmp_path):
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-09-01", "2023-07-14")
    codes = [f"{idx:06d}.SZ" for idx in range(12)]
    trade_cal = pd.DataFrame({
        "exchange": "SSE",
        "cal_date": dates.strftime("%Y%m%d"),
        "is_open": 1,
        "pretrade_date": pd.Series(dates).shift(1).dt.strftime("%Y%m%d"),
    })
    signals = signal_dates_from_calendar(dates)
    month_starts = pd.date_range(dates.min(), dates.max(), freq="MS")
    index_weight = pd.DataFrame([
        {"index_code": "000300.SH", "con_code": code, "trade_date": month.strftime("%Y%m%d"), "weight": 100 / len(codes)}
        for month in month_starts for code in codes
    ])
    memberships = build_membership_table(index_weight, signals, max_stocks=None)

    daily_rows = []
    basic_rows = []
    adj_rows = []
    limit_rows = []
    for stock_idx, code in enumerate(codes):
        returns = rng.normal(0.0003 + stock_idx * 0.00001, 0.012, len(dates))
        close = (10 + stock_idx) * np.cumprod(1 + returns)
        open_price = close * (1 + rng.normal(0, 0.002, len(dates)))
        for idx, date in enumerate(dates):
            date_text = date.strftime("%Y%m%d")
            daily_rows.append({
                "ts_code": code, "trade_date": date_text, "open": open_price[idx],
                "high": max(open_price[idx], close[idx]) * 1.01, "low": min(open_price[idx], close[idx]) * 0.99,
                "close": close[idx], "pre_close": close[idx - 1] if idx else close[idx],
                "vol": 1_000_000 * (1 + stock_idx / 10) + idx * 100, "amount": close[idx] * 1_000_000,
            })
            basic_rows.append({
                "ts_code": code, "trade_date": date_text, "turnover_rate": 1 + stock_idx / 10,
                "volume_ratio": 1.0, "pb": 1.2 + stock_idx / 20,
                "total_mv": 1e6 * (stock_idx + 1), "circ_mv": 8e5 * (stock_idx + 1),
            })
            adj_rows.append({"ts_code": code, "trade_date": date_text, "adj_factor": 1.0})
            limit_rows.append({
                "ts_code": code, "trade_date": date_text,
                "up_limit": open_price[idx] * 1.2, "down_limit": open_price[idx] * 0.8,
            })
    benchmark_close = 4000 * np.cumprod(1 + rng.normal(0.0002, 0.008, len(dates)))
    benchmark = pd.DataFrame({
        "ts_code": "000300.SH", "trade_date": dates.strftime("%Y%m%d"),
        "open": benchmark_close * 0.999, "high": benchmark_close * 1.01,
        "low": benchmark_close * 0.99, "close": benchmark_close,
        "pre_close": np.r_[benchmark_close[0], benchmark_close[:-1]], "pct_chg": 0.0, "vol": 1.0, "amount": 1.0,
    })
    industry_members_base = pd.DataFrame({
        "l1_code": ["I1"] * 6 + ["I2"] * 6,
        "l1_name": ["Industry 1"] * 6 + ["Industry 2"] * 6,
        "ts_code": codes,
        "in_date": "20100101",
        "out_date": None,
    })
    industry_members = pd.concat([
        industry_members_base.assign(taxonomy_source="SW2014"),
        industry_members_base.assign(taxonomy_source="SW2021"),
    ], ignore_index=True)
    market = {
        "daily": pd.DataFrame(daily_rows),
        "daily_basic": pd.DataFrame(basic_rows),
        "adj_factor": pd.DataFrame(adj_rows),
        "stk_limit": pd.DataFrame(limit_rows),
        "suspend_d": pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type", "suspend_timing"]),
        "memberships": memberships,
    }
    # Remove one future exit-date quote. The earlier signal-date row must remain in the
    # decision universe, but its supervised label must be masked.
    target_signal = signals[signals >= pd.Timestamp("2022-07-01")][0]
    next_signal = signals[signals.get_loc(target_signal) + 1]
    missing_exit_date = dates[dates > next_signal][0].strftime("%Y%m%d")
    market["daily"] = market["daily"].loc[
        ~((market["daily"]["ts_code"] == codes[0]) & (market["daily"]["trade_date"] == missing_exit_date))
    ].copy()
    reference = {
        "trade_cal": trade_cal,
        "index_weight": index_weight,
        "industry_class": pd.DataFrame(),
        "industry_members": industry_members,
        "benchmark": benchmark,
    }
    config = load_config(Path(__file__).parents[1] / "config.yaml", "SMOKE_TEST")
    panel, execution, benchmark_prices, metadata = prepare_research_panel(market, reference, config)
    assert set(panel["split"]) == {"train", "validation", "test"}
    assert panel[config["features"]["raw"]].notna().all().all()
    assert panel.groupby("signal_date").size().min() == len(codes)
    assert metadata["missing_industry_fraction"] == 0.0
    missing_label_row = panel[(panel["signal_date"] == target_signal) & (panel["ts_code"] == codes[0])]
    assert len(missing_label_row) == 1
    assert not bool(missing_label_row.iloc[0]["label_available"])
    assert bool(missing_label_row.iloc[0]["decision_eligible"])
    predictions = train_ridge(panel, config["features"]["raw"], config, tmp_path)
    assert len(predictions) == len(panel)
    assert predictions["y_pred"].notna().all()

