from __future__ import annotations

import json
import shutil
import hashlib
from pathlib import Path

import pandas as pd

from .data import (
    CacheStore,
    TushareDownloader,
    download_market_data,
    download_reference_data,
    data_quality_audit,
    load_download_checkpoint,
    save_download_checkpoint,
)
from .evaluation import evaluate_all
from .features import load_prepared_checkpoint, prepare_research_panel, save_prepared_checkpoint
from .graphs import graph_audit, random_graph_audit
from .reporting import create_figures, create_results_bundle, write_research_summary
from .training import combine_predictions, train_neural_model, train_ridge
from .utils import (
    atomic_write_json,
    atomic_write_yaml,
    collect_environment,
    config_hash,
    experiment_id,
    ensure_project_dirs,
    load_config,
    require_full_run_gpu,
    set_seed,
    sha256_file,
    tree_fingerprint,
    utc_now_iso,
    validate_formal_protocol,
)


REFERENCE_NAMES = {"trade_cal", "index_weight", "industry_class", "industry_members", "benchmark"}


def initialize(project_root: str | Path, config_path: str | Path, mode: str) -> dict:
    root = Path(project_root)
    ensure_project_dirs(root)
    config = load_config(config_path, mode)
    source_root = Path(__file__).resolve().parents[2]
    config.setdefault("_runtime", {})["source_hash"] = tree_fingerprint(source_root / "src", ("*.py",))
    config["_runtime"]["configuration_file_hash"] = sha256_file(config_path)
    validate_formal_protocol(config)
    set_seed(int(config["project"]["seed"]))
    require_full_run_gpu(config)
    atomic_write_yaml(root / "experiment_config.yaml", config)
    environment = collect_environment()
    atomic_write_json(root / "environment.json", environment)
    return config


def data_stage(pro, config: dict, project_root: str | Path):
    root = Path(project_root)
    data_key = config_hash({"project": config["project"], "data": config["data"]})
    cached = load_download_checkpoint(root, data_key)
    if cached:
        reference = {key: cached[key] for key in REFERENCE_NAMES}
        market = {key: value for key, value in cached.items() if key not in REFERENCE_NAMES}
        # Audit logic may evolve independently of the immutable raw-data cache.
        atomic_write_json(root / "tables" / "download_data_quality.json", data_quality_audit({**reference, **market}))
        _record_data_identity(config, root, data_key)
        return reference, market, data_key, True
    cache = CacheStore(root / "cache", refresh=bool(config["data"].get("refresh_cache", False)))
    downloader = TushareDownloader(pro, cache, config)
    reference = download_reference_data(downloader, config, root)
    market = download_market_data(downloader, config, reference, root)
    save_download_checkpoint(market, reference, root, data_key)
    _record_data_identity(config, root, data_key)
    return reference, market, data_key, False


def _record_data_identity(config: dict, root: Path, data_key: str) -> None:
    checkpoint = root / "prepared" / f"download_{data_key}"
    digest = hashlib.sha256()
    files = []
    for path in sorted(checkpoint.glob("*.parquet")):
        file_hash = sha256_file(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
        files.append({
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": file_hash,
        })
    if not files:
        raise FileNotFoundError(f"No parquet files found under {checkpoint}")
    fingerprint = digest.hexdigest()
    runtime = config.setdefault("_runtime", {})
    runtime["data_fingerprint"] = fingerprint
    runtime.pop("experiment_id", None)
    runtime["experiment_id"] = experiment_id(config)
    atomic_write_json(root / "tables" / "data_file_fingerprints.json", {
        "data_key": data_key, "data_fingerprint": fingerprint, "files": files,
    })
    atomic_write_yaml(root / "experiment_config.yaml", config)


def feature_stage(reference: dict, market: dict, config: dict, project_root: str | Path, data_key: str):
    root = Path(project_root)
    prepared_key = config_hash(
        {"data_key": data_key, "features": config["features"], "splits": config["splits"], "target": config["project"]["target"]}
    )
    cached = load_prepared_checkpoint(root, prepared_key)
    if cached:
        panel, execution, benchmark, metadata = cached
        _write_universe_timing_audit(panel, metadata, root)
        return panel, execution, benchmark, metadata, prepared_key, True
    panel, execution, benchmark, metadata = prepare_research_panel(market, reference, config)
    save_prepared_checkpoint(panel, execution, benchmark, metadata, root, prepared_key)
    _write_universe_timing_audit(panel, metadata, root)
    return panel, execution, benchmark, metadata, prepared_key, False


def _write_universe_timing_audit(panel: pd.DataFrame, metadata: dict, root: Path) -> None:
    if not panel.get("decision_eligible", pd.Series(False, index=panel.index)).fillna(False).all():
        raise RuntimeError("Decision-universe audit failed: a retained row is not signal-date eligible.")
    by_split = {}
    for split, section in panel.groupby("split", sort=True):
        labeled = section.get("label_available", section["target_excess_return"].notna()).fillna(False)
        by_split[str(split)] = {
            "decision_rows": int(len(section)),
            "labeled_rows": int(labeled.sum()),
            "unlabeled_decision_rows": int((~labeled).sum()),
            "entry_price_missing_rows": int((~section.get("entry_price_available", pd.Series(False, index=section.index))).sum()),
            "exit_price_missing_rows": int((~section.get("exit_price_available", pd.Series(False, index=section.index))).sum()),
        }
    atomic_write_json(root / "tables" / "universe_timing_audit.json", {
        "captured_at_utc": utc_now_iso(),
        "future_price_filter_applied": False,
        "future_price_columns_used_for_decision_filter": [],
        "decision_rule": "signal-date point-in-time membership plus signal-date ret_60 and beta_60 availability",
        "label_rule": "future entry/exit prices determine label_available only; unlabeled rows remain prediction/backtest candidates",
        "by_split": by_split,
        "metadata_label_available_fraction": metadata.get("label_available_fraction"),
    })
    taxonomy_by_split = {}
    if "industry_taxonomy_source" in panel:
        for split, section in panel.groupby("split", sort=True):
            taxonomy_by_split[str(split)] = {
                str(key): int(value)
                for key, value in section["industry_taxonomy_source"].value_counts(dropna=False).items()
            }
    atomic_write_json(root / "tables" / "industry_taxonomy_audit.json", {
        "captured_at_utc": utc_now_iso(),
        "policy": metadata.get("industry_taxonomy_policy"),
        "configured_versions": metadata.get("industry_taxonomy_versions", []),
        "rows_by_split_and_taxonomy": taxonomy_by_split,
        "missing_industry_fraction": metadata.get("missing_industry_fraction"),
        "claim_boundary": (
            "Version-applicable taxonomy and effective membership intervals are enforced. "
            "TuShare revision/announcement timestamps are not available, so this is a best-effort "
            "point-in-time classification rather than formal proof against all vendor backfills."
        ),
    })


def graph_audit_stage(panel: pd.DataFrame, project_root: str | Path, config: dict | None = None):
    root = Path(project_root)
    by_date, summary = graph_audit(panel)
    by_date.to_csv(root / "tables" / "graph_audit_by_date.csv", index=False)
    if config is not None:
        summary["random_graph"] = random_graph_audit(
            panel,
            [int(seed) for seed in config["models"].get("random_graph_seeds", [])],
            int(config["models"].get("random_graph_swaps_per_edge", 3)),
        )
        if summary["random_graph"]["degree_preserving_rewire"].get("degree_sequence_mismatches", 0):
            raise RuntimeError("Random-graph audit failed: degree sequence changed.")
        label_audit = summary["random_graph"]["label_permutation"]
        if any(label_audit.get(key, 0) for key in [
            "clique_size_mismatches", "directed_edge_count_mismatches", "connected_component_count_mismatches"
        ]):
            raise RuntimeError("Label-permutation audit failed: clique-size topology was not preserved.")
    atomic_write_json(root / "tables" / "graph_audit_summary.json", summary)
    atomic_write_json(root / "graph_audit_summary.json", summary)
    return by_date, summary


def train_model_stage(model_name: str, panel: pd.DataFrame, config: dict, project_root: str | Path):
    if model_name not in config["models"].get("enabled", []):
        raise ValueError(f"Model {model_name!r} is not enabled by the frozen experiment configuration.")
    features = list(config["features"]["raw"])
    if model_name == "ridge":
        return train_ridge(panel, features, config, project_root)
    return train_neural_model(model_name, panel, features, config, project_root)


def combine_prediction_stage(panel: pd.DataFrame, prediction_frames: list[pd.DataFrame], project_root: str | Path):
    return combine_predictions(panel, prediction_frames, project_root)


def evaluation_stage(predictions, execution, benchmark, config, project_root):
    return evaluate_all(predictions, execution, benchmark, config, project_root)


def finalize_stage(results: dict, metadata: dict, config: dict, project_root: str | Path):
    figures = create_figures(results, config, project_root)
    summary = write_research_summary(results, metadata, config, project_root)
    bundle = create_results_bundle(project_root)
    experiment = config.get("_runtime", {}).get("experiment_id", experiment_id(config))
    snapshot = _snapshot_experiment_outputs(Path(project_root), experiment)
    return {"figures": figures, "summary": summary, "bundle": bundle, "experiment_output": snapshot}


def _snapshot_experiment_outputs(root: Path, experiment: str) -> Path:
    """Keep result-bearing artifacts under a unique experiment ID without duplicating raw caches."""
    destination = root / "outputs" / experiment
    destination.mkdir(parents=True, exist_ok=True)
    files = [
        "metrics.csv", "paired_rankic_tests.csv", "ablation_results.csv", "portfolio_returns.csv",
        "trade_logs.csv", "training_seed_results.csv", "industry_neutral_metrics.csv",
        "predictions.parquet", "graph_audit_summary.json", "experiment_config.yaml", "environment.json",
        "research_summary.md", "research_results_bundle.zip", "file_manifest.csv",
    ]
    for name in files:
        source = root / name
        if source.exists():
            shutil.copy2(source, destination / name)
    for folder in ["figures", "tables", "training_logs"]:
        source = root / folder
        target = destination / folder
        if source.exists():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
    atomic_write_json(destination / "experiment_identity.json", {"experiment_id": experiment})
    return destination

