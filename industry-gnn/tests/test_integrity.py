from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

from industry_gnn.backtest import CostScenario, run_execution_backtest
from industry_gnn.data import build_membership_table, data_quality_audit
from industry_gnn.evaluation import industry_neutral_metrics, paired_rankic_tests
from industry_gnn.features import _assign_split, _cross_sectional_transform, assign_industry_asof, make_execution_prices
from industry_gnn.graphs import (
    degree_preserving_rewire, industry_edges, label_permuted_industry_edges,
    permute_industry_labels, self_loop_edges,
)
from industry_gnn.utils import load_config, validate_formal_protocol
from industry_gnn.training import build_replicate_specs


def test_mode_overrides_and_no_secret_field():
    config = load_config(Path(__file__).parents[1] / "config.yaml", "SMOKE_TEST")
    assert config["models"]["max_epochs"] == 30
    assert config["models"]["training_seeds"] == [42]
    assert config["models"]["random_graph_seeds"] == [1042]
    assert config["data"]["max_stocks_per_snapshot"] == 40
    assert "token" not in str(config).lower()


def test_full_run_protocol_is_frozen_and_fail_closed():
    config = load_config(Path(__file__).parents[1] / "config.yaml", "FULL_RUN")
    validate_formal_protocol(config)
    assert config["splits"]["validation_end"] == "2023-12-31"
    assert config["splits"]["test_start"] == "2024-01-01"
    assert config["splits"]["test_end"] == "2025-12-31"
    assert len(config["models"]["training_seeds"]) == 3
    assert len(config["models"]["random_graph_seeds"]) == 3
    names = [item["name"] for item in config["evaluation"]["paired_rankic_comparisons"]]
    assert "GAT minus Industry MLP" in names
    assert config["models"]["random_graph_replication"] == "anchored_factorial"

    changed = deepcopy(config)
    changed["splits"]["test_start"] = "2022-01-01"
    try:
        validate_formal_protocol(changed)
    except RuntimeError as exc:
        assert "frozen" in str(exc)
    else:
        raise AssertionError("A changed FULL_RUN test interval must fail closed.")


def test_random_graph_replication_separates_training_and_graph_seeds():
    config = load_config(Path(__file__).parents[1] / "config.yaml", "FULL_RUN")
    specs = build_replicate_specs(config, "gat_random")
    main = [item for item in specs if item[2] == "main_training_seed"]
    topology = [item for item in specs if item[2] == "graph_randomization"]
    assert main == [(42, 1042, "main_training_seed"), (3407, 1042, "main_training_seed"), (2026, 1042, "main_training_seed")]
    assert topology == [(42, 13407, "graph_randomization"), (42, 12026, "graph_randomization")]


def test_cross_sectional_transform_is_date_local():
    panel = pd.DataFrame({
        "signal_date": [pd.Timestamp("2022-01-07")] * 3 + [pd.Timestamp("2022-01-14")] * 3,
        "factor": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
    })
    transformed = _cross_sectional_transform(panel, ["factor"], 0.0, 1.0)
    means = transformed.groupby("signal_date")["factor"].mean()
    assert np.allclose(means.to_numpy(), 0.0)
    assert np.isclose(transformed.iloc[0]["factor"], transformed.iloc[3]["factor"])


def test_split_purges_boundary_crossing_label():
    panel = pd.DataFrame({
        "signal_date": pd.to_datetime(["2021-12-24", "2021-12-31", "2022-01-07"]),
        "exit_date": pd.to_datetime(["2021-12-31", "2022-01-10", "2022-01-17"]),
    })
    splits = {
        "train_start": "2021-01-01", "train_end": "2021-12-31",
        "validation_start": "2022-01-01", "validation_end": "2022-12-31",
        "test_start": "2023-01-01", "test_end": "2023-12-31",
        "purge_labels_crossing_boundaries": True,
    }
    assigned = _assign_split(panel, splits)
    assert assigned.loc[0, "split"] == "train"
    assert assigned.loc[1, "split"] == "excluded"
    assert assigned.loc[2, "split"] == "validation"


def test_industry_interval_respects_lag_and_out_date():
    rows = pd.DataFrame({
        "signal_date": pd.to_datetime(["2022-01-10", "2022-01-12"]),
        "ts_code": ["A", "A"],
    })
    members = pd.DataFrame({
        "ts_code": ["A"], "l1_code": ["I1"], "l1_name": ["Industry"],
        "in_date": ["20220110"], "out_date": ["20220112"],
    })
    assigned = assign_industry_asof(rows, members, lag_days=1)
    assert pd.isna(assigned.iloc[0]["l1_code"])
    assert pd.isna(assigned.iloc[1]["l1_code"])


def test_random_graph_preserves_degree_sequence():
    edges = industry_edges(["A", "A", "A", "B", "B", "B"], include_self_loops=True)
    rewired = degree_preserving_rewire(edges, 6, seed=7, swaps_per_edge=1)
    original_degree = sorted(np.bincount(edges[0][edges[0] != edges[1]], minlength=6))
    rewired_degree = sorted(np.bincount(rewired[0][rewired[0] != rewired[1]], minlength=6))
    assert original_degree == rewired_degree


def test_label_permutation_preserves_industry_clique_sizes():
    labels = ["A", "A", "A", "B", "B", "C", None]
    permuted = permute_industry_labels(labels, seed=17)
    before = sorted(pd.Series(labels).value_counts(dropna=False).tolist())
    after = sorted(pd.Series(permuted).value_counts(dropna=False).tolist())
    assert before == after
    assert industry_edges(labels).shape == label_permuted_industry_edges(labels, seed=17).shape


def test_self_loop_graph_has_no_cross_stock_edges():
    edges = self_loop_edges(4)
    assert edges.shape == (2, 4)
    assert np.array_equal(edges[0], edges[1])


def test_historical_stamp_duty_changes_on_2023_08_28():
    costs = CostScenario.from_dict({
        "commission_bps": 2, "slippage_bps": 5, "stamp_duty_policy": "historical"
    })
    schedule = [
        {"effective_start": "2008-09-19", "effective_end": "2023-08-27", "stamp_duty_sell_bps": 10},
        {"effective_start": "2023-08-28", "effective_end": None, "stamp_duty_sell_bps": 5},
    ]
    assert costs.stamp_duty_bps("2023-08-27", schedule) == 10
    assert costs.stamp_duty_bps("2023-08-28", schedule) == 5


def test_data_quality_uses_dataset_specific_primary_keys():
    index_weight = pd.DataFrame({
        "index_code": ["000300.SH", "000905.SH"],
        "con_code": ["A", "A"],
        "trade_date": ["20230131", "20230131"],
    })
    audit = data_quality_audit({"index_weight": index_weight})["datasets"]["index_weight"]
    assert audit["key_columns"] == ["index_code", "con_code", "trade_date"]
    assert audit["duplicate_key_rows"] == 0


def test_paired_block_bootstrap_and_holm_are_reported():
    dates = pd.date_range("2023-01-01", periods=20, freq="W")
    ic = pd.DataFrame({
        "model": ["a"] * 20 + ["b"] * 20,
        "signal_date": list(dates) * 2,
        "rank_ic": [0.2] * 20 + [0.0] * 20,
    })
    result = paired_rankic_tests(
        ic, [{"name": "a-b", "model_a": "a", "model_b": "b", "primary": True}],
        repetitions=200, block_length=4, seed=7,
    )
    assert result.iloc[0]["mean_rank_ic_difference"] > 0
    assert result.iloc[0]["bootstrap_95_low"] > 0
    assert bool(result.iloc[0]["significant_after_holm_5pct"])


def test_backtest_carries_blocked_sale():
    predictions = pd.DataFrame({
        "model": ["gcn", "gcn", "gcn", "gcn"],
        "split": ["test"] * 4,
        "signal_date": pd.to_datetime(["2023-01-06", "2023-01-06", "2023-01-13", "2023-01-13"]),
        "entry_date": pd.to_datetime(["2023-01-09", "2023-01-09", "2023-01-16", "2023-01-16"]),
        "exit_date": pd.to_datetime(["2023-01-16", "2023-01-16", "2023-01-23", "2023-01-23"]),
        "ts_code": ["A", "B", "A", "B"],
        "y_pred": [1.0, 0.0, 0.0, 1.0],
    })
    execution = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-09", "2023-01-09", "2023-01-16", "2023-01-16", "2023-01-23", "2023-01-23"]),
        "ts_code": ["A", "B", "A", "B", "A", "B"],
        "adj_open": [10.0, 10.0, 9.0, 11.0, 9.5, 12.0],
        "buyable": [True, True, False, True, True, True],
        "sellable": [True, True, False, True, True, True],
        "suspended": [False, False, True, False, False, False],
    })
    benchmark = pd.DataFrame({"date": execution["trade_date"].unique(), "benchmark_open": [100.0, 101.0, 102.0]})
    returns, trades = run_execution_backtest(
        predictions,
        execution,
        benchmark,
        top_k=1,
        scenario_name="zero",
        scenario={"commission_bps": 0, "stamp_duty_sell_bps": 0, "slippage_bps": 0},
    )
    blocked = trades[(trades["date"] == pd.Timestamp("2023-01-16")) & (trades["ts_code"] == "A")]
    assert (blocked["status"] == "BLOCKED").any()
    assert returns.iloc[-1]["portfolio_value"] > 0


def test_open_limit_blocks_without_using_later_high_or_low():
    dates = ["20240108", "20240109"]
    daily = pd.DataFrame({
        "ts_code": ["UP", "DOWN"], "trade_date": dates,
        # Intraday prices deliberately show that both limits opened later.  The
        # conservative opening-auction decision must remain blocked.
        "open": [11.0, 9.0], "high": [11.0, 10.5], "low": [10.2, 9.0], "close": [10.5, 10.0],
    })
    market = {
        "daily": daily,
        "adj_factor": pd.DataFrame({"ts_code": ["UP", "DOWN"], "trade_date": dates, "adj_factor": [1.0, 1.0]}),
        "stk_limit": pd.DataFrame({
            "ts_code": ["UP", "DOWN"], "trade_date": dates,
            "up_limit": [11.0, 11.0], "down_limit": [9.0, 9.0],
        }),
        "suspend_d": pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type", "suspend_timing"]),
    }
    prices = make_execution_prices(market).set_index("ts_code")
    assert not bool(prices.loc["UP", "buyable"])
    assert bool(prices.loc["UP", "sellable"])
    assert bool(prices.loc["DOWN", "buyable"])
    assert not bool(prices.loc["DOWN", "sellable"])


def test_index_snapshot_requires_configured_lag():
    weights = pd.DataFrame({
        "index_code": ["000300.SH", "000300.SH"],
        "con_code": ["A", "B"], "trade_date": ["20240105", "20240104"], "weight": [1.0, 1.0],
    })
    table = build_membership_table(weights, [pd.Timestamp("2024-01-05")], None, lag_calendar_days=1)
    assert table.iloc[0]["ts_code"] == "B"
    assert table.iloc[0]["index_snapshot_date"] == pd.Timestamp("2024-01-04")


def test_benchmark_starts_when_strategy_starts():
    predictions = pd.DataFrame({
        "model": ["ridge"], "split": ["test"], "signal_date": pd.to_datetime(["2024-01-05"]),
        "entry_date": pd.to_datetime(["2024-01-08"]), "exit_date": pd.to_datetime(["2024-01-10"]),
        "ts_code": ["A"], "y_pred": [1.0],
    })
    execution = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10"]),
        "ts_code": ["A", "A", "A"], "adj_open": [10.0, 11.0, 12.0],
        "buyable": [True] * 3, "sellable": [True] * 3, "suspended": [False] * 3,
    })
    benchmark = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]),
        "benchmark_open": [90.0, 100.0, 101.0, 102.0],
    })
    returns, _ = run_execution_backtest(
        predictions, execution, benchmark, 1, "zero",
        {"commission_bps": 0, "stamp_duty_sell_bps": 0, "slippage_bps": 0},
    )
    assert returns.iloc[0]["benchmark_return"] == 0.0
    assert np.isclose(returns.iloc[1]["benchmark_return"], 0.01)


def test_tiny_industry_neutral_float_noise_is_undefined():
    predictions = pd.DataFrame({
        "model": ["gcn"] * 6, "split": ["test"] * 6,
        "signal_date": pd.to_datetime(["2024-01-05"] * 6),
        "l1_code": ["A"] * 3 + ["B"] * 3,
        "y_pred": [0.01, 0.01 + 1e-12, 0.01 - 1e-12, -0.02, -0.02 + 1e-12, -0.02 - 1e-12],
        "target_excess_return": [0.1, 0.0, -0.1, 0.2, 0.0, -0.2],
        "label_available": [True] * 6,
    })
    result = industry_neutral_metrics(predictions, neutral_prediction_atol=1e-10, neutral_prediction_rtol=1e-5)
    assert np.isnan(result.iloc[0]["industry_neutral_rank_ic_mean"])
    assert result.iloc[0]["industry_neutral_rank_ic_defined_dates"] == 0

