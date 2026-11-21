from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import sha256_file, utc_now_iso


PALETTE = {
    "ridge": "#5B6770",
    "shared_mlp": "#2F6B9A",
    "industry_mlp": "#C49A3A",
    "gcn": "#D46A3A",
    "gat": "#7A8F3D",
    "gcn_random": "#B56B8A",
    "gat_random": "#8B5E83",
    "gcn_label_random": "#6C8EAD",
    "gat_label_random": "#A26D5A",
    "gat_self_only": "#7C7C7C",
}


def set_research_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def create_figures(results: dict[str, pd.DataFrame], config: dict, project_root: str | Path) -> list[Path]:
    root = Path(project_root)
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    set_research_style()
    created = []

    metrics = results["metrics"]
    base = metrics[
        (metrics["cost_scenario"] == config["evaluation"]["base_cost_scenario"])
        & (metrics["top_k"] == int(config["evaluation"].get("primary_top_k", 10)))
    ].drop_duplicates("model")
    order = base.sort_values("rank_ic_mean")["model"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.barh(order, base.set_index("model").loc[order, "rank_ic_mean"], color=[PALETTE.get(item, "#2F6B9A") for item in order])
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set(title="Out-of-Sample Rank IC by Model", xlabel="Mean weekly Rank IC", ylabel="")
    fig.tight_layout()
    path = figure_dir / "rank_ic_by_model.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    neutral = results.get("industry_neutral_metrics", pd.DataFrame())
    if not neutral.empty:
        neutral = neutral.sort_values("industry_neutral_rank_ic_mean")
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.barh(
            neutral["model"], neutral["industry_neutral_rank_ic_mean"],
            color=[PALETTE.get(item, "#2F6B9A") for item in neutral["model"]],
        )
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set(
            title="Industry-Neutral Out-of-Sample Rank IC",
            xlabel="Mean weekly Rank IC after same-date industry demeaning", ylabel="",
        )
        fig.tight_layout()
        path = figure_dir / "industry_neutral_rank_ic.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    returns = results["portfolio_returns"]
    selected = returns[
        (returns["cost_scenario"] == config["evaluation"]["base_cost_scenario"])
        & (returns["top_k"] == int(config["evaluation"].get("primary_top_k", 10)))
    ].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    for model, section in selected.groupby("model"):
        section = section.sort_values("date")
        curve = (1.0 + section["net_return"].fillna(0)).cumprod()
        ax.plot(section["date"], curve, label=model, color=PALETTE.get(model), linewidth=1.4)
    if not selected.empty:
        benchmark_section = selected[selected["model"] == selected["model"].iloc[0]].sort_values("date")
        benchmark_curve = (1.0 + benchmark_section["benchmark_return"].fillna(0)).cumprod()
        ax.plot(benchmark_section["date"], benchmark_curve, label="CSI 300 benchmark", color="#222222", linestyle="--")
    selected_k = int(config["evaluation"].get("primary_top_k", 10))
    selected_cost = config["evaluation"]["base_cost_scenario"]
    ax.set(
        title=f"Execution-Aware Top-{selected_k} Portfolio Growth — {selected_cost} costs",
        xlabel="Date", ylabel="Growth of 1.0",
    )
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    path = figure_dir / "cumulative_portfolio_returns.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    paired = results.get("paired_rankic_tests", pd.DataFrame()).dropna(subset=["mean_rank_ic_difference"])
    if not paired.empty:
        paired = paired.sort_values("mean_rank_ic_difference")
        center = paired["mean_rank_ic_difference"].to_numpy()
        lower = np.maximum(0.0, center - paired["bootstrap_95_low"].to_numpy())
        upper = np.maximum(0.0, paired["bootstrap_95_high"].to_numpy() - center)
        fig, ax = plt.subplots(figsize=(8.0, max(3.2, 0.55 * len(paired))))
        ax.errorbar(center, paired["comparison"], xerr=np.vstack([lower, upper]), fmt="o", color="#2F6B9A", capsize=3)
        ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set(
            title=f"Paired Rank IC Differences — moving-block bootstrap (block={int(paired['bootstrap_block_length'].iloc[0])})",
            xlabel="Mean Rank IC difference (model A − model B)", ylabel="",
        )
        fig.tight_layout()
        path = figure_dir / "paired_rankic_differences.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    graph_path = root / "tables" / "graph_audit_by_date.csv"
    if graph_path.exists():
        graph = pd.read_csv(graph_path, parse_dates=["signal_date"])
        if not graph.empty:
            fig, left = plt.subplots(figsize=(8.0, 4.1))
            left.plot(graph["signal_date"], graph["mean_peer_degree"], color="#D46A3A", label="Mean peer degree")
            left.set(xlabel="Signal date", ylabel="Mean peer degree", title="Historical Industry-Graph Structure")
            right = left.twinx()
            right.plot(graph["signal_date"], graph["industries"], color="#2F6B9A", alpha=0.75, label="Industries")
            right.set_ylabel("Industry count")
            handles = left.get_lines() + right.get_lines()
            left.legend(handles, [line.get_label() for line in handles], loc="best")
            fig.tight_layout()
            path = figure_dir / "graph_structure_over_time.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            created.append(path)

    ic = results["daily_ic"].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    for model, section in ic.groupby("model"):
        section = section.sort_values("signal_date")
        rolling = section["rank_ic"].rolling(13, min_periods=6).mean()
        ax.plot(section["signal_date"], rolling, label=model, color=PALETTE.get(model), linewidth=1.2)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set(title="Thirteen-Week Rolling Mean Rank IC", xlabel="Signal date", ylabel="Rank IC")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    path = figure_dir / "rolling_rank_ic.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    return created


def write_research_summary(
    results: dict[str, pd.DataFrame], metadata: dict, config: dict, project_root: str | Path
) -> Path:
    root = Path(project_root)
    metrics = results["metrics"]
    base = metrics[
        (metrics["cost_scenario"] == config["evaluation"]["base_cost_scenario"])
        & (metrics["top_k"] == int(config["evaluation"].get("primary_top_k", 10)))
    ].drop_duplicates("model")
    best_predictive = base.sort_values("rank_ic_mean", ascending=False).iloc[0] if not base.empty else None
    best_investment = base.sort_values("annualized_return", ascending=False).iloc[0] if not base.empty else None
    status = "FULL_RUN completed" if config["run_mode"] == "FULL_RUN" else "SMOKE_TEST completed; FULL_RUN not yet executed"
    lines = [
        "# Research Summary",
        "",
        f"Generated: {utc_now_iso()}",
        f"Status: **{status}**",
        "",
        "## Completed work",
        "",
        f"- Prepared {metadata.get('rows', 0):,} stock-week observations across {metadata.get('dates', 0)} signal dates.",
        f"- Trained and evaluated: {', '.join(sorted(metrics['model'].unique()))}.",
        f"- Formal protocol: `{config.get('formal_protocol', {}).get('protocol_id', 'not declared')}`; independent test interval {config['splits']['test_start']} through {config['splits']['test_end']}.",
        "- Kept the signal-date prediction universe separate from future label availability; missing future prices are masked only for fitting/evaluation.",
        "- Used one chronological train/validation/test protocol, point-in-time index membership, date-applicable SW2014/SW2021 taxonomies, next-open execution, tradeability constraints, date-specific historical stamp duty, and explicit cost scenarios.",
        "- Opening tradeability uses only the opening quote, suspension flag and published price limit; same-day high/low are never consulted.",
        "- GCN/GAT preserve the stock-specific encoder path and fuse own embedding, graph message and their difference.",
        "- Ran degree-preserving rewiring, fixed-clique-size industry-label permutation, GAT self-loop-only controls, industry-neutral diagnostics with numerical-degeneracy guards, paired moving-block bootstrap tests, and configured training-seed replications.",
        "",
        "## Preliminary observed results",
        "",
    ]
    if best_predictive is not None:
        lines.append(
            f"- Highest mean test Rank IC in the base comparison: `{best_predictive['model']}` "
            f"({best_predictive['rank_ic_mean']:.4f}; moving-block bootstrap 95% interval "
            f"[{best_predictive['rank_ic_bootstrap_95_low']:.4f}, {best_predictive['rank_ic_bootstrap_95_high']:.4f}])."
        )
    if best_investment is not None:
        lines.append(
            f"- Highest base-cost annualized Top-{int(best_investment['top_k'])} return: `{best_investment['model']}` "
            f"({best_investment['annualized_return']:.2%}, Sharpe {best_investment['sharpe_ratio']:.2f})."
        )
    lines.extend([
        "- These are realized pipeline outputs, not claims of statistical or economic superiority. Interpret them jointly with costs, drawdowns, ablations, and the FULL_RUN status.",
        "",
        "## Material limitations",
        "",
        "- The 2016–2021-07-29 sample uses SW2014 and later dates use SW2021. `index_member_all` still lacks a separate historical announcement/revision timestamp, so residual vendor-backfill risk remains and is explicitly audited.",
        "- TuShare historical records may be revised after the fact; the cache timestamp and file manifest should be retained for reproducibility.",
        "- Fractional adjusted shares are used; lot sizes, bid-ask depth, and market impact beyond configured slippage are not modeled.",
        "- Statistical superiority requires checking `paired_rankic_tests.csv`, Holm-adjusted p-values, and seed stability; a higher point estimate alone is insufficient.",
        "- If the execution audit has zero blocked trades, the constraint code was not empirically exercised in that run even though synthetic tests cover it.",
        "- A smoke-test result is a pipeline check, not the course-project conclusion.",
        "",
        "## Files to inspect",
        "",
        "- `metrics.csv`, `paired_rankic_tests.csv`, `ablation_results.csv`, `portfolio_returns.csv`, `trade_logs.csv`, `predictions.parquet`",
        "- `training_seed_results.csv`, `industry_neutral_metrics.csv`, `graph_audit_summary.json`",
        "- `tables/execution_constraint_audit.csv`, `tables/graph_audit_by_date.csv`, `tables/industry_neutral_by_date.csv`",
        "- `tables/universe_timing_audit.json`, `tables/industry_taxonomy_audit.json`, `FORMAL_EXPERIMENT_PROTOCOL.md`",
        "- `figures/`, `training_logs/`, `experiment_config.yaml`, `environment.json`, `file_manifest.csv`",
    ])
    path = root / "research_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _looks_sensitive(text: str) -> bool:
    patterns = [
        r"(?i)(?:token|api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]{16,}['\"]",
        r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._-]{16,}",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def create_results_bundle(project_root: str | Path) -> Path:
    root = Path(project_root)
    bundle_path = root / "research_results_bundle.zip"
    # Per-model/per-seed prediction checkpoints may include stale hashes from prior revisions.
    # The current aggregate `predictions.parquet` and seed summary table remain in the bundle.
    excluded_roots = {"cache", "prepared", "checkpoints", "predictions", "outputs", ".venv"}
    candidates = [path for path in root.rglob("*") if path.is_file() and path != bundle_path]
    manifest_rows = []
    included = []
    for path in candidates:
        relative = path.relative_to(root)
        omit = (
            relative.parts[0] in excluded_roots
            or any(part in {"cache", "prepared", "checkpoints", "predictions", ".venv"} for part in relative.parts)
            or path.stat().st_size > 50 * 1024 * 1024
        )
        sensitive = False
        if not omit and path.suffix.lower() in {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".ipynb"}:
            sensitive = _looks_sensitive(path.read_text(encoding="utf-8", errors="ignore"))
            if sensitive:
                omit = True
        # Large caches/checkpoints are already content-fingerprinted by the data/model stages.
        # Avoid reading gigabytes merely to build the portable result bundle manifest.
        digest = "" if omit else sha256_file(path)
        manifest_rows.append({
            "path": str(relative).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
            "included_in_bundle": not omit,
            "excluded_reason": "credential_pattern" if sensitive else ("large_or_cache_or_checkpoint" if omit else ""),
        })
        if not omit:
            included.append((path, relative))

    manifest_path = root / "file_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]) if manifest_rows else ["path"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    included = [(path, relative) for path, relative in included if path != manifest_path]
    included.append((manifest_path, Path("file_manifest.csv")))
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, relative in included:
            archive.write(path, arcname=str(relative).replace("\\", "/"))
    return bundle_path

