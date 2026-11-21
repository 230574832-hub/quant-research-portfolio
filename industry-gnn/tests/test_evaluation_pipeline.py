from __future__ import annotations

import pandas as pd

from industry_gnn.evaluation import evaluate_all


def test_evaluation_exports_paired_tests_historical_costs_and_audits(tmp_path):
    (tmp_path / "tables").mkdir()
    (tmp_path / "predictions").mkdir()
    signals = pd.to_datetime(["2023-01-06", "2023-01-13"])
    codes = ["A", "B", "C", "D"]
    targets = [[0.04, 0.03, -0.01, -0.02], [-0.02, -0.01, 0.03, 0.04]]
    rows = []
    row_id = 0
    for model in ["a", "b"]:
        for date_idx, signal in enumerate(signals):
            for stock_idx, code in enumerate(codes):
                target = targets[date_idx][stock_idx]
                rows.append({
                    "row_id": row_id, "model": model, "split": "test", "signal_date": signal,
                    "entry_date": pd.Timestamp("2023-01-09") if date_idx == 0 else pd.Timestamp("2023-01-16"),
                    "exit_date": pd.Timestamp("2023-01-16") if date_idx == 0 else pd.Timestamp("2023-01-23"),
                    "ts_code": code, "l1_code": "I1" if stock_idx < 2 else "I2",
                    "target_excess_return": target, "target_return": target, "benchmark_return": 0.0,
                    "label_available": True, "y_pred": target if model == "a" else -target,
                })
                row_id += 1
    predictions = pd.DataFrame(rows)
    execution = pd.DataFrame([
        {"trade_date": date, "ts_code": code, "adj_open": 10.0 + idx, "buyable": True, "sellable": True, "suspended": False}
        for date in pd.to_datetime(["2023-01-09", "2023-01-16", "2023-01-23"])
        for idx, code in enumerate(codes)
    ])
    benchmark = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-09", "2023-01-16", "2023-01-23"]),
        "benchmark_open": [100.0, 101.0, 102.0],
    })
    config = {
        "project": {"seed": 7},
        "evaluation": {
            "bootstrap_repetitions": 100, "bootstrap_block_length": 2, "top_k": [2], "primary_top_k": 2,
            "stamp_duty_schedule": [
                {"effective_start": "2008-09-19", "effective_end": "2023-08-27", "stamp_duty_sell_bps": 10},
                {"effective_start": "2023-08-28", "effective_end": None, "stamp_duty_sell_bps": 5},
            ],
            "cost_scenarios": {
                "base": {"commission_bps": 2, "slippage_bps": 5, "stamp_duty_policy": "historical"}
            },
            "paired_rankic_comparisons": [
                {"name": "a-b", "model_a": "a", "model_b": "b", "primary": True}
            ],
        },
    }
    results = evaluate_all(predictions, execution, benchmark, config, tmp_path)
    assert not results["paired_rankic_tests"].empty
    assert set(results["execution_constraint_audit"]["blocked"]) == {0}
    sells = results["trade_logs"].query("side == 'SELL' and status == 'EXECUTED'")
    assert not sells.empty
    assert set(sells["applied_stamp_duty_bps"]) == {10.0}
    for name in [
        "metrics.csv", "paired_rankic_tests.csv", "ablation_results.csv", "portfolio_returns.csv",
        "trade_logs.csv", "training_seed_results.csv", "industry_neutral_metrics.csv",
    ]:
        assert (tmp_path / name).exists()
