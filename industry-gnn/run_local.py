from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_gnn.pipeline import (  # noqa: E402
    combine_prediction_stage,
    data_stage,
    evaluation_stage,
    feature_stage,
    finalize_stage,
    graph_audit_stage,
    initialize,
    train_model_stage,
)
from industry_gnn.security import create_tushare_client, get_tushare_token  # noqa: E402
from industry_gnn.utils import collect_environment, config_hash  # noqa: E402


def _require_cuda() -> None:
    environment = collect_environment()
    if not environment.get("cuda_available"):
        raise RuntimeError(
            "CUDA preflight failed. FULL_RUN was not started. Run scripts/check_gpu.py after installing "
            "the NVIDIA driver and CUDA-enabled PyTorch."
        )


def _prepare(mode: str):
    config = initialize(PROJECT_DIR, PROJECT_DIR / "config.yaml", mode)
    token = get_tushare_token()
    pro = create_tushare_client(token)
    reference, market, data_key, data_cached = data_stage(pro, config, PROJECT_DIR)
    panel, execution, benchmark, metadata, prepared_key, prepared_cached = feature_stage(
        reference, market, config, PROJECT_DIR, data_key
    )
    graph_audit_stage(panel, PROJECT_DIR, config)
    print(json.dumps({
        "mode": mode, "data_cache_hit": data_cached, "feature_cache_hit": prepared_cached,
        "data_key": data_key, "prepared_key": prepared_key,
        "experiment_id": config.get("_runtime", {}).get("experiment_id"),
        "panel_rows": len(panel), "signal_dates": int(panel["signal_date"].nunique()),
    }, indent=2, default=str))
    return config, panel, execution, benchmark, metadata


def _train(config: dict, panel, models: list[str]) -> list:
    frames = []
    for model in models:
        print(f"[train] {model}", flush=True)
        frames.append(train_model_stage(model, panel, config, PROJECT_DIR))
    return frames


def _load_completed_predictions(config: dict) -> list:
    run_hash = config_hash(config)
    frames = []
    missing = []
    import pandas as pd

    for model in config["models"]["enabled"]:
        path = PROJECT_DIR / "predictions" / f"{model}_{run_hash}.parquet"
        success = path.with_suffix(".json")
        if path.exists() and success.exists():
            frames.append(pd.read_parquet(path))
        else:
            missing.append(model)
    if missing:
        raise RuntimeError(
            "Evaluation stopped because these frozen-protocol models are incomplete: " + ", ".join(missing)
        )
    return frames


def run_end_to_end(mode: str, confirm_full_run: bool, allow_cpu_smoke: bool = False) -> None:
    if mode == "FULL_RUN":
        if not confirm_full_run:
            raise RuntimeError("FULL_RUN requires the explicit flag --confirm-full-run.")
        _require_cuda()
    elif not allow_cpu_smoke:
        _require_cuda()
    config, panel, execution, benchmark, metadata = _prepare(mode)
    frames = _train(config, panel, list(config["models"]["enabled"]))
    predictions = combine_prediction_stage(panel, frames, PROJECT_DIR)
    results = evaluation_stage(predictions, execution, benchmark, config, PROJECT_DIR)
    artifacts = finalize_stage(results, metadata, config, PROJECT_DIR)
    print(json.dumps({key: str(value) for key, value in artifacts.items() if key != "figures"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local runner for the frozen Industry-GNN experiment.")
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="Write and print an environment snapshot.")
    env.add_argument("--require-cuda", action="store_true")

    data = sub.add_parser("data", help="Download/cache data and build audited features.")
    data.add_argument("--mode", choices=["SMOKE_TEST", "FULL_RUN"], default="SMOKE_TEST")

    train = sub.add_parser("train", help="Train selected models; completed hashes resume automatically.")
    train.add_argument("--mode", choices=["SMOKE_TEST", "FULL_RUN"], default="SMOKE_TEST")
    train.add_argument("--models", nargs="+", required=True)
    train.add_argument("--confirm-full-run", action="store_true")

    evaluate = sub.add_parser("evaluate", help="Evaluate all completed frozen-protocol models and package results.")
    evaluate.add_argument("--mode", choices=["SMOKE_TEST", "FULL_RUN"], default="SMOKE_TEST")
    evaluate.add_argument("--confirm-full-run", action="store_true")

    preflight = sub.add_parser("preflight", help="GPU-gated end-to-end SMOKE_TEST.")
    preflight.add_argument("--allow-cpu-smoke", action="store_true", help="CI/debug only; never enables CPU FULL_RUN.")

    full = sub.add_parser("full", help="Resume or start the complete frozen FULL_RUN.")
    full.add_argument("--confirm-full-run", action="store_true")

    args = parser.parse_args()
    if args.command == "env":
        if args.require_cuda:
            _require_cuda()
        payload = collect_environment()
        (PROJECT_DIR / "environment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
    elif args.command == "data":
        if args.mode == "FULL_RUN":
            _require_cuda()
        _prepare(args.mode)
    elif args.command == "train":
        if args.mode == "FULL_RUN" and not args.confirm_full_run:
            raise RuntimeError("FULL_RUN training requires --confirm-full-run.")
        if args.mode == "FULL_RUN":
            _require_cuda()
        config, panel, *_ = _prepare(args.mode)
        unknown = sorted(set(args.models) - set(config["models"]["enabled"]))
        if unknown:
            raise ValueError("Models outside the frozen protocol: " + ", ".join(unknown))
        _train(config, panel, args.models)
    elif args.command == "evaluate":
        if args.mode == "FULL_RUN" and not args.confirm_full_run:
            raise RuntimeError("FULL_RUN evaluation requires --confirm-full-run.")
        config, panel, execution, benchmark, metadata = _prepare(args.mode)
        frames = _load_completed_predictions(config)
        predictions = combine_prediction_stage(panel, frames, PROJECT_DIR)
        results = evaluation_stage(predictions, execution, benchmark, config, PROJECT_DIR)
        finalize_stage(results, metadata, config, PROJECT_DIR)
    elif args.command == "preflight":
        run_end_to_end("SMOKE_TEST", False, args.allow_cpu_smoke)
    elif args.command == "full":
        run_end_to_end("FULL_RUN", args.confirm_full_run)


if __name__ == "__main__":
    main()
