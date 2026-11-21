from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .backtest import run_execution_backtest
from .utils import config_hash


def _has_meaningful_variation(
    values: pd.Series,
    *,
    reference_scale: float | None = None,
    atol: float = 1e-12,
    rtol: float = 1e-8,
) -> bool:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(finite) < 2:
        return False
    scale = float(reference_scale) if reference_scale is not None else float(np.max(np.abs(finite)))
    threshold = float(atol) + float(rtol) * max(abs(scale), float(atol))
    return bool(np.ptp(finite) > threshold)


def _safe_corr(
    x: pd.Series,
    y: pd.Series,
    rank: bool = False,
    *,
    x_reference_scale: float | None = None,
    atol: float = 1e-12,
    rtol: float = 1e-8,
) -> float:
    mask = x.notna() & y.notna()
    if (
        mask.sum() < 3
        or not _has_meaningful_variation(
            x[mask], reference_scale=x_reference_scale, atol=atol, rtol=rtol
        )
        or not _has_meaningful_variation(y[mask], atol=atol, rtol=rtol)
    ):
        return np.nan
    statistic = spearmanr(x[mask], y[mask]).statistic if rank else pearsonr(x[mask], y[mask]).statistic
    return float(statistic)


def daily_ic(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test = predictions.loc[predictions["split"] == "test"].copy()
    if "label_available" in test:
        test["label_available"] = test["label_available"].fillna(False).astype(bool)
    else:
        test["label_available"] = test["target_excess_return"].notna()
    for (model, date), section in test.groupby(["model", "signal_date"]):
        labeled = section.loc[section["label_available"] & section["target_excess_return"].notna()]
        rows.append({
            "model": model, "signal_date": date,
            "ic": _safe_corr(labeled["y_pred"], labeled["target_excess_return"], rank=False),
            "rank_ic": _safe_corr(labeled["y_pred"], labeled["target_excess_return"], rank=True),
            "n_decision": int(len(section)), "n_labeled": int(len(labeled)),
        })
    return pd.DataFrame(rows)


def _moving_block_samples(values: pd.Series | np.ndarray, repetitions: int, block_length: int, seed: int) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = len(clean)
    if n < 2:
        return np.asarray([], dtype=float)
    block = max(1, min(int(block_length), n))
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(int(repetitions), block_count))
    samples = np.empty(int(repetitions), dtype=float)
    offsets = np.arange(block)
    for idx, row in enumerate(starts):
        positions = ((row[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        samples[idx] = clean[positions].mean()
    return samples


def _block_bootstrap_mean(values, repetitions: int, block_length: int, seed: int) -> tuple[float, float]:
    samples = _moving_block_samples(values, repetitions, block_length, seed)
    if not len(samples):
        return np.nan, np.nan
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def prediction_metrics(predictions: pd.DataFrame, repetitions: int, block_length: int, seed: int):
    ic_frame = daily_ic(predictions)
    rows = []
    test = predictions.loc[predictions["split"] == "test"].copy()
    test["label_available"] = test.get("label_available", test["target_excess_return"].notna()).fillna(False).astype(bool)
    for model, decision_rows in test.groupby("model"):
        section = decision_rows.loc[decision_rows["label_available"] & decision_rows["target_excess_return"].notna()]
        dates = ic_frame[ic_frame["model"] == model].sort_values("signal_date")
        rank_lo, rank_hi = _block_bootstrap_mean(dates["rank_ic"], repetitions, block_length, seed)
        ic_lo, ic_hi = _block_bootstrap_mean(dates["ic"], repetitions, block_length, seed + 1)
        error = section["y_pred"] - section["target_excess_return"]
        rows.append({
            "model": model, "mse": float(np.mean(error**2)) if len(error) else np.nan,
            "ic_mean": float(dates["ic"].mean()), "ic_std": float(dates["ic"].std()),
            "ic_bootstrap_95_low": ic_lo, "ic_bootstrap_95_high": ic_hi,
            "rank_ic_mean": float(dates["rank_ic"].mean()), "rank_ic_std": float(dates["rank_ic"].std()),
            "rank_ic_bootstrap_95_low": rank_lo, "rank_ic_bootstrap_95_high": rank_hi,
            "test_decision_observations": int(len(decision_rows)), "test_labeled_observations": int(len(section)),
            "test_label_coverage": float(len(section) / len(decision_rows)) if len(decision_rows) else np.nan,
            "test_dates": int(decision_rows["signal_date"].nunique()), "bootstrap_block_length": int(block_length),
        })
    return pd.DataFrame(rows), ic_frame


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [np.nan] * len(p_values)
    valid = [(idx, value) for idx, value in enumerate(p_values) if np.isfinite(value)]
    ordered = sorted(valid, key=lambda item: item[1])
    running = 0.0
    m = len(ordered)
    for rank, (idx, value) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * value))
        adjusted[idx] = running
    return adjusted


def paired_rankic_tests(ic_frame: pd.DataFrame, comparisons: list[dict], repetitions: int, block_length: int, seed: int):
    pivot = ic_frame.pivot(index="signal_date", columns="model", values="rank_ic").sort_index()
    rows = []
    for idx, comparison in enumerate(comparisons):
        model_a, model_b = comparison["model_a"], comparison["model_b"]
        if model_a not in pivot or model_b not in pivot:
            continue
        paired = (pivot[model_a] - pivot[model_b]).dropna()
        samples = _moving_block_samples(paired.to_numpy(), repetitions, block_length, seed + idx * 97)
        if len(samples):
            low, high = np.quantile(samples, [0.025, 0.975])
            p_value = min(1.0, 2.0 * min(float(np.mean(samples <= 0)), float(np.mean(samples >= 0))))
        else:
            low = high = p_value = np.nan
        rows.append({
            "comparison": comparison["name"], "model_a": model_a, "model_b": model_b,
            "hypothesis": comparison.get("hypothesis", "unspecified"),
            "primary_comparison": bool(comparison.get("primary", False)),
            "paired_dates": int(len(paired)), "mean_rank_ic_difference": float(paired.mean()) if len(paired) else np.nan,
            "bootstrap_95_low": float(low), "bootstrap_95_high": float(high),
            "bootstrap_p_value_two_sided": float(p_value), "bootstrap_block_length": int(block_length),
            "bootstrap_repetitions": int(repetitions),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["multiplicity_family"] = np.where(result["primary_comparison"], "preregistered_primary", "secondary")
        result["holm_adjusted_p_value"] = np.nan
        for family, indexes in result.groupby("multiplicity_family").groups.items():
            adjusted = _holm_adjust(result.loc[indexes, "bootstrap_p_value_two_sided"].tolist())
            result.loc[indexes, "holm_adjusted_p_value"] = adjusted
        result["significant_after_holm_5pct"] = result["holm_adjusted_p_value"] < 0.05
    return result


def investment_metrics(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model", "top_k", "cost_scenario"]
    for key, section in returns.groupby(keys, sort=True):
        section = section.sort_values("date")
        net = section["net_return"].fillna(0.0)
        benchmark = section["benchmark_return"].fillna(0.0)
        years = len(net) / 252.0
        cumulative = float((1.0 + net).prod())
        benchmark_cumulative = float((1.0 + benchmark).prod())
        annualized = cumulative ** (1.0 / years) - 1.0 if years > 0 and cumulative > 0 else np.nan
        benchmark_annualized = benchmark_cumulative ** (1.0 / years) - 1.0 if years > 0 and benchmark_cumulative > 0 else np.nan
        volatility = float(net.std(ddof=1) * np.sqrt(252))
        sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(252)) if net.std(ddof=1) > 0 else np.nan
        equity_curve = (1.0 + net).cumprod()
        drawdown = equity_curve / equity_curve.cummax() - 1.0
        rows.append({
            "model": key[0], "top_k": int(key[1]), "cost_scenario": key[2],
            "cumulative_net_return": cumulative - 1.0,
            "benchmark_cumulative_return": benchmark_cumulative - 1.0,
            "annualized_return": annualized, "benchmark_annualized_return": benchmark_annualized,
            "annualized_excess_return": annualized - benchmark_annualized,
            "annualized_volatility": volatility, "sharpe_ratio": sharpe,
            "maximum_drawdown": float(drawdown.min()), "average_daily_turnover": float(section["turnover"].mean()),
            "total_turnover": float(section["turnover"].sum()),
            "total_transaction_cost": float(section["transaction_cost"].sum()),
            "ending_value": float(section["portfolio_value"].iloc[-1]),
        })
    return pd.DataFrame(rows)


def build_ablation_results(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    pred_metrics = metrics.drop_duplicates("model").set_index("model")
    rows = []
    comparisons = [
        ("A", "GCN vs Shared MLP", "gcn", "shared_mlp", "graph message passing over stock-only features"),
        ("B", "GCN vs Industry Aggregation", "gcn", "industry_mlp", "incremental value beyond explicit peer averages"),
        ("C", "GCN industry vs degree-preserving random", "gcn", "gcn_random", "industry topology beyond graph degree"),
        ("D", "GAT vs GCN", "gat", "gcn", "learned neighbor weights over normalized averaging"),
        ("F", "GAT industry vs degree-preserving random", "gat", "gat_random", "GAT benefit attributable to real industry edges"),
        ("G", "GAT industry vs self-loops only", "gat", "gat_self_only", "GAT benefit attributable to cross-stock messages"),
    ]
    for code, name, left, right, hypothesis in comparisons:
        if left in pred_metrics.index and right in pred_metrics.index:
            rows.append({
                "ablation": code, "comparison": name, "model_a": left, "model_b": right,
                "delta_mse_a_minus_b": pred_metrics.loc[left, "mse"] - pred_metrics.loc[right, "mse"],
                "delta_rank_ic_a_minus_b": pred_metrics.loc[left, "rank_ic_mean"] - pred_metrics.loc[right, "rank_ic_mean"],
                "hypothesis_tested": hypothesis,
            })

    return pd.DataFrame(rows)


def industry_neutral_metrics(
    predictions: pd.DataFrame,
    neutral_prediction_atol: float = 1e-10,
    neutral_prediction_rtol: float = 1e-5,
    return_by_date: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Explicit industry-neutral and within-industry diagnostics, averaged over test dates.

    neutral_rank_ic removes the same-date industry mean from prediction and target before
    cross-sectional Spearman correlation. within_industry_rank_ic first computes Spearman
    within each industry (n>=3), then equally averages valid industries. Exposure is the
    cross-sectional prediction variance explained by the stock-weighted industry mean.
    """
    test = predictions.loc[predictions["split"] == "test"].copy()
    test = test.loc[test.get("label_available", test["target_excess_return"].notna()).fillna(False)]
    test["industry_key"] = test["l1_code"].fillna("UNKNOWN")
    rows = []
    for (model, date), day in test.groupby(["model", "signal_date"], sort=True):
        industry_groups = day.groupby("industry_key")
        pred_mean = industry_groups["y_pred"].transform("mean")
        target_mean = industry_groups["target_excess_return"].transform("mean")
        neutral_prediction = day["y_pred"] - pred_mean
        neutral_target = day["target_excess_return"] - target_mean
        original_prediction_scale = float(np.max(np.abs(day["y_pred"].to_numpy(dtype=float))))
        neutral_prediction_defined = _has_meaningful_variation(
            neutral_prediction,
            reference_scale=original_prediction_scale,
            atol=neutral_prediction_atol,
            rtol=neutral_prediction_rtol,
        )
        within = []
        for _, industry in industry_groups:
            if len(industry) >= 3:
                within.append(_safe_corr(
                    industry["y_pred"], industry["target_excess_return"], rank=True,
                    x_reference_scale=original_prediction_scale,
                    atol=neutral_prediction_atol, rtol=neutral_prediction_rtol,
                ))
        valid_within = [value for value in within if np.isfinite(value)]
        total_variance = float(day["y_pred"].var(ddof=0))
        explained = float(pred_mean.var(ddof=0))
        neutral_variance = float(neutral_prediction.var(ddof=0))
        rows.append({
            "model": model, "signal_date": date,
            "industry_neutral_rank_ic": (
                _safe_corr(
                    neutral_prediction, neutral_target, rank=True,
                    x_reference_scale=original_prediction_scale,
                    atol=neutral_prediction_atol, rtol=neutral_prediction_rtol,
                ) if neutral_prediction_defined else np.nan
            ),
            "industry_neutral_rank_ic_defined": bool(neutral_prediction_defined),
            "equal_weighted_within_industry_rank_ic": float(np.mean(valid_within)) if valid_within else np.nan,
            "industry_exposure_variance_share": (
                float(np.clip(explained / total_variance, 0.0, 1.0)) if total_variance > 0 else np.nan
            ),
            "total_prediction_variance": total_variance,
            "neutral_prediction_variance": neutral_variance,
            "mean_absolute_industry_prediction": float(pred_mean.abs().mean()),
            "valid_industries_for_within_rank_ic": int(len(valid_within)),
            "labeled_stocks": int(len(day)),
        })
    by_date = pd.DataFrame(rows)
    if by_date.empty:
        return (by_date, by_date) if return_by_date else by_date
    summary = by_date.groupby("model", as_index=False).agg(
        industry_neutral_rank_ic_mean=("industry_neutral_rank_ic", "mean"),
        industry_neutral_rank_ic_std=("industry_neutral_rank_ic", "std"),
        equal_weighted_within_industry_rank_ic_mean=("equal_weighted_within_industry_rank_ic", "mean"),
        industry_exposure_variance_share_mean=("industry_exposure_variance_share", "mean"),
        mean_absolute_industry_prediction_mean=("mean_absolute_industry_prediction", "mean"),
        total_prediction_variance_mean=("total_prediction_variance", "mean"),
        neutral_prediction_variance_mean=("neutral_prediction_variance", "mean"),
        industry_neutral_rank_ic_defined_dates=("industry_neutral_rank_ic_defined", "sum"),
        test_dates=("signal_date", "nunique"),
    )
    summary["industry_neutral_rank_ic_undefined_dates"] = (
        summary["test_dates"] - summary["industry_neutral_rank_ic_defined_dates"]
    )
    return (summary, by_date) if return_by_date else summary


def seed_stability_metrics(root: Path, predictions: pd.DataFrame, config: dict) -> pd.DataFrame:
    run_hash = config_hash(config)
    metadata = predictions.drop_duplicates("row_id")[["row_id", "split", "signal_date", "target_excess_return", "label_available"]]
    rows = []
    for path in sorted((root / "predictions").glob(f"*_{run_hash}_by_seed.parquet")):
        success_path = Path(str(path).replace("_by_seed.parquet", ".json"))
        run_lookup = {}
        if success_path.exists():
            payload = json.loads(success_path.read_text(encoding="utf-8"))
            for item in payload.get("seed_runs", []):
                key = (int(item["training_seed"]), int(item.get("graph_seed", -1)), item.get("replicate_role", "legacy_unspecified"))
                run_lookup[key] = item
        frame = pd.read_parquet(path).merge(metadata, on="row_id", how="left", validate="many_to_one")
        frame = frame.loc[(frame["split_y"] == "test") & frame["label_available"].fillna(False)]
        if "graph_seed" not in frame:
            frame["graph_seed"] = -1
        if "replicate_role" not in frame:
            frame["replicate_role"] = "legacy_unspecified"
        for (model, seed, graph_seed, role), section in frame.groupby(
            ["model", "training_seed", "graph_seed", "replicate_role"]
        ):
            run = run_lookup.get((int(seed), int(graph_seed), role), {})
            rank_values = [_safe_corr(day["y_pred"], day["target_excess_return"], rank=True) for _, day in section.groupby("signal_date")]
            error = section["y_pred"] - section["target_excess_return"]
            rows.append({
                "model": model, "training_seed": int(seed),
                "graph_seed": (np.nan if int(graph_seed) < 0 else int(graph_seed)),
                "random_graph_replication": bool(int(graph_seed) >= 0),
                "replicate_role": role,
                "epochs": run.get("epochs"),
                "training_elapsed_seconds": run.get("training_elapsed_seconds"),
                "peak_gpu_memory_mb": run.get("peak_gpu_memory_mb"),
                "amp_enabled": run.get("amp_enabled"),
                "parameter_count": run.get("parameter_count"),
                "mse": float(np.mean(error**2)),
                "rank_ic_mean": float(np.nanmean(rank_values)), "labeled_observations": int(len(section)),
                "test_dates": int(section["signal_date"].nunique()),
            })
    ridge_success = root / "predictions" / f"ridge_{run_hash}.json"
    if ridge_success.exists():
        payload = json.loads(ridge_success.read_text(encoding="utf-8"))
        ridge = predictions.loc[
            (predictions["model"] == "ridge") & (predictions["split"] == "test")
            & predictions.get("label_available", predictions["target_excess_return"].notna()).fillna(False)
        ]
        values = [_safe_corr(day["y_pred"], day["target_excess_return"], rank=True) for _, day in ridge.groupby("signal_date")]
        error = ridge["y_pred"] - ridge["target_excess_return"]
        rows.append({
            "model": "ridge", "training_seed": np.nan, "graph_seed": np.nan,
            "random_graph_replication": False, "replicate_role": "deterministic_baseline",
            "epochs": np.nan, "training_elapsed_seconds": payload.get("training_elapsed_seconds"),
            "peak_gpu_memory_mb": 0.0, "amp_enabled": False, "parameter_count": np.nan,
            "mse": float(np.mean(error**2)) if len(error) else np.nan,
            "rank_ic_mean": float(np.nanmean(values)) if len(values) else np.nan,
            "labeled_observations": int(len(ridge)), "test_dates": int(ridge["signal_date"].nunique()),
        })
    return pd.DataFrame(rows)


def execution_constraint_audit(trade_logs: pd.DataFrame) -> pd.DataFrame:
    if trade_logs.empty:
        return pd.DataFrame(columns=["model", "top_k", "cost_scenario", "trades", "executed", "blocked", "blocked_fraction", "blocked_reasons"])
    rows = []
    for key, section in trade_logs.groupby(["model", "top_k", "cost_scenario"], sort=True):
        blocked = section.loc[section["status"] == "BLOCKED"]
        reasons = blocked["reason"].value_counts().to_dict()
        rows.append({
            "model": key[0], "top_k": int(key[1]), "cost_scenario": key[2], "trades": int(len(section)),
            "executed": int((section["status"] == "EXECUTED").sum()), "blocked": int(len(blocked)),
            "blocked_fraction": float(len(blocked) / len(section)),
            "blocked_reasons": "; ".join(f"{name}:{count}" for name, count in sorted(reasons.items())),
        })
    return pd.DataFrame(rows)


def evaluate_all(predictions, execution_prices, benchmark_prices, config: dict, project_root: str | Path):
    root = Path(project_root)
    repetitions = int(config["evaluation"]["bootstrap_repetitions"])
    block_length = int(config["evaluation"].get("bootstrap_block_length", 1))
    prediction_table, ic_frame = prediction_metrics(predictions, repetitions, block_length, int(config["project"]["seed"]))
    paired = paired_rankic_tests(
        ic_frame, config["evaluation"].get("paired_rankic_comparisons", []), repetitions, block_length,
        int(config["project"]["seed"]) + 1000,
    )
    all_returns, all_trades = [], []
    for model, model_predictions in predictions.groupby("model"):
        for top_k in config["evaluation"]["top_k"]:
            for scenario_name, scenario in config["evaluation"]["cost_scenarios"].items():
                returns, trades = run_execution_backtest(
                    model_predictions, execution_prices, benchmark_prices, int(top_k), scenario_name, scenario,
                    stamp_duty_schedule=config["evaluation"].get("stamp_duty_schedule", []),
                )
                all_returns.append(returns)
                all_trades.append(trades)
    portfolio_returns = pd.concat(all_returns, ignore_index=True)
    trade_logs = pd.concat(all_trades, ignore_index=True) if any(not frame.empty for frame in all_trades) else pd.DataFrame()
    investments = investment_metrics(portfolio_returns)
    metrics = investments.merge(prediction_table, on="model", how="left")
    ablations = build_ablation_results(predictions, prediction_table)
    seed_metrics = seed_stability_metrics(root, predictions, config)
    neutral_metrics, neutral_by_date = industry_neutral_metrics(
        predictions,
        neutral_prediction_atol=float(config["evaluation"].get("neutral_prediction_atol", 1e-10)),
        neutral_prediction_rtol=float(config["evaluation"].get("neutral_prediction_rtol", 1e-5)),
        return_by_date=True,
    )
    execution_audit = execution_constraint_audit(trade_logs)

    metrics.to_csv(root / "metrics.csv", index=False)
    ablations.to_csv(root / "ablation_results.csv", index=False)
    portfolio_returns.to_csv(root / "portfolio_returns.csv", index=False)
    trade_logs.to_csv(root / "trade_logs.csv", index=False)
    paired.to_csv(root / "paired_rankic_tests.csv", index=False)
    seed_metrics.to_csv(root / "training_seed_results.csv", index=False)
    neutral_metrics.to_csv(root / "industry_neutral_metrics.csv", index=False)
    neutral_by_date.to_csv(root / "tables" / "industry_neutral_by_date.csv", index=False)
    ic_frame.to_csv(root / "tables" / "daily_ic.csv", index=False)
    paired.to_csv(root / "tables" / "paired_rankic_tests.csv", index=False)
    seed_metrics.to_csv(root / "tables" / "seed_metrics.csv", index=False)
    execution_audit.to_csv(root / "tables" / "execution_constraint_audit.csv", index=False)
    return {
        "metrics": metrics, "ablation_results": ablations, "portfolio_returns": portfolio_returns,
        "trade_logs": trade_logs, "daily_ic": ic_frame, "paired_rankic_tests": paired,
        "seed_metrics": seed_metrics, "industry_neutral_metrics": neutral_metrics,
        "industry_neutral_by_date": neutral_by_date,
        "execution_constraint_audit": execution_audit,
    }
