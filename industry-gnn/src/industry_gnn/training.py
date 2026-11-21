from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from .graphs import build_graph_snapshots
from .utils import atomic_write_json, config_hash, set_seed, utc_now_iso


NEURAL_MODELS = [
    "shared_mlp", "industry_mlp", "gcn", "gat", "gcn_random", "gat_random",
    "gcn_label_random", "gat_label_random", "gat_self_only"
]


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temp)
    os.replace(temp, path)


def _prediction_path(root: Path, model_name: str, run_hash: str) -> Path:
    return root / "predictions" / f"{model_name}_{run_hash}.parquet"


def _labeled(frame: pd.DataFrame) -> pd.DataFrame:
    mask = frame.get("label_available", frame["target_excess_return"].notna()).fillna(False)
    return frame.loc[mask & frame["target_excess_return"].notna()]


def train_ridge(panel: pd.DataFrame, feature_columns: list[str], config: dict, project_root: str | Path) -> pd.DataFrame:
    import joblib

    root = Path(project_root)
    run_hash = config_hash(config)
    prediction_path = _prediction_path(root, "ridge", run_hash)
    success_path = prediction_path.with_suffix(".json")
    if prediction_path.exists() and success_path.exists():
        return pd.read_parquet(prediction_path)

    fit_start = time.perf_counter()

    train = _labeled(panel[panel["split"] == "train"])
    validation = _labeled(panel[panel["split"] == "validation"])
    if train.empty or validation.empty:
        raise RuntimeError("Ridge requires labeled rows in both training and validation splits.")
    best = None
    trials = []
    for alpha in config["models"]["ridge_alphas"]:
        model = Ridge(alpha=float(alpha))
        model.fit(train[feature_columns], train["target_excess_return"])
        validation_prediction = model.predict(validation[feature_columns])
        validation_mse = mean_squared_error(validation["target_excess_return"], validation_prediction)
        trials.append({"alpha": float(alpha), "validation_mse": float(validation_mse)})
        if best is None or validation_mse < best[0]:
            best = (validation_mse, float(alpha), model)
    _, alpha, model = best
    # Predict every signal-date row; future label availability is not an eligibility rule.
    predictions = pd.DataFrame({
        "row_id": panel["row_id"].astype(np.int64),
        "model": "ridge",
        "y_pred": model.predict(panel[feature_columns]),
    })
    _atomic_parquet(predictions, prediction_path)
    checkpoint_path = root / "checkpoints" / f"ridge_{run_hash}.joblib"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_temp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".part")
    joblib.dump({"model": model, "features": feature_columns, "alpha": alpha}, checkpoint_temp)
    os.replace(checkpoint_temp, checkpoint_path)
    atomic_write_json(success_path, {
        "completed_at_utc": utc_now_iso(), "selected_alpha": alpha, "trials": trials,
        "fit_rows": int(len(train)), "validation_rows": int(len(validation)),
        "predicted_rows": int(len(panel)), "checkpoint": str(checkpoint_path),
        "training_elapsed_seconds": float(time.perf_counter() - fit_start),
        "peak_gpu_memory_mb": 0.0, "amp_enabled": False,
    })
    return predictions


def _loader(snapshots, batch_size: int, shuffle: bool):
    from torch_geometric.loader import DataLoader
    return DataLoader(snapshots, batch_size=batch_size, shuffle=shuffle)


def _evaluate_mse(model, snapshots, device, batch_size: int, amp_enabled: bool = False) -> float:
    import torch

    model.eval()
    squared_error = 0.0
    observations = 0
    with torch.no_grad():
        for batch in _loader(snapshots, batch_size, False):
            batch = batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                pred = model(batch)
            mask = batch.label_mask.bool()
            if mask.any():
                squared_error += torch.sum((pred[mask] - batch.y[mask]) ** 2).item()
                observations += int(mask.sum().item())
    return squared_error / observations if observations else float("inf")


def _fit_neural_trial(model, train_snapshots, val_snapshots, config: dict, device, log_path: Path):
    import torch

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["models"]["learning_rate"]),
        weight_decay=float(config["models"]["weight_decay"]),
    )
    batch_size = int(config["models"]["batch_graphs"])
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    start = time.perf_counter()
    amp_enabled = bool(config["models"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    for epoch in range(1, int(config["models"]["max_epochs"]) + 1):
        model.train()
        train_losses = []
        for batch in _loader(train_snapshots, batch_size, True):
            batch = batch.to(device)
            mask = batch.label_mask.bool()
            if not mask.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                prediction = model(batch)
                loss = torch.mean((prediction[mask] - batch.y[mask]) ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        validation_loss = _evaluate_mse(model, val_snapshots, device, batch_size, amp_enabled)
        row = {
            "epoch": epoch,
            "train_mse": float(np.mean(train_losses)) if train_losses else None,
            "validation_mse": validation_loss,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }
        history.append(row)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if validation_loss < best_loss - float(config["models"]["min_delta"]):
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(config["models"]["patience"]):
            break
    if best_state is None:
        raise RuntimeError("No valid checkpoint; verify labeled rows in train and validation.")
    model.load_state_dict(best_state)
    return model, best_loss, history


def _predict_neural(
    model, snapshots_by_split: dict, device, batch_size: int, model_name: str, seed: int, graph_seed: int,
    replicate_role: str, amp_enabled: bool,
) -> pd.DataFrame:
    import torch

    rows = []
    model.eval()
    with torch.no_grad():
        for split, snapshots in snapshots_by_split.items():
            for batch in _loader(snapshots, batch_size, False):
                batch = batch.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    prediction = model(batch)
                rows.append(pd.DataFrame({
                    "row_id": batch.row_id.detach().cpu().numpy(),
                    "model": model_name,
                    "y_pred": prediction.detach().cpu().float().numpy(),
                    "split": split,
                    "training_seed": int(seed),
                    "graph_seed": int(graph_seed),
                    "replicate_role": replicate_role,
                }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["row_id", "model", "y_pred", "split", "training_seed", "graph_seed", "replicate_role"]
    )


def _graph_mode(model_name: str) -> str:
    if model_name in {"gcn_random", "gat_random"}:
        return "random"
    if model_name in {"gcn_label_random", "gat_label_random"}:
        return "label_permutation"
    if model_name == "gat_self_only":
        return "self"
    return "industry"


def build_replicate_specs(config: dict, model_name: str) -> list[tuple[int, int, str]]:
    """Separate parameter-seed replication from random-topology sensitivity."""
    selection_seed = int(config["models"].get("hyperparameter_selection_seed", config["project"]["seed"]))
    seeds = [int(value) for value in config["models"].get("training_seeds", [selection_seed])]
    if _graph_mode(model_name) not in {"random", "label_permutation"}:
        return [(seed, -1, "main_training_seed") for seed in seeds]
    graph_seeds = [int(value) for value in config["models"].get("random_graph_seeds", [])]
    if not graph_seeds:
        raise RuntimeError(f"{model_name}: at least one random_graph_seed is required.")
    anchor_seed = int(config["models"].get("random_graph_anchor_training_seed", selection_seed))
    if anchor_seed not in seeds:
        raise RuntimeError("random_graph_anchor_training_seed must be included in training_seeds.")
    specifications = [(seed, graph_seeds[0], "main_training_seed") for seed in seeds]
    specifications.extend((anchor_seed, graph_seed, "graph_randomization") for graph_seed in graph_seeds[1:])
    return specifications


def _snapshots(panel, feature_columns, config, graph_seed, model_name, root: Path, run_hash: str):
    mode = _graph_mode(model_name)
    graph_tag = int(graph_seed) if mode in {"random", "label_permutation"} else -1
    cache = root / "prepared" / "graph_snapshots" / run_hash / f"{mode}_g{graph_tag}.pt"
    if cache.exists():
        import torch
        return torch.load(cache, map_location="cpu", weights_only=False)
    snapshots = {
        split: build_graph_snapshots(
            panel, feature_columns, split, graph_mode=mode, seed=int(graph_seed),
            swaps_per_edge=int(config["models"]["random_graph_swaps_per_edge"]),
        )
        for split in ["train", "validation", "test"]
    }
    _atomic_torch_save(snapshots, cache)
    return snapshots


def _model_kwargs(config: dict, feature_columns: list[str], hidden_dim: int) -> dict:
    return {
        "input_dim": len(feature_columns), "hidden_dim": int(hidden_dim),
        "layers": int(config["models"]["mlp_layers"]),
        "graph_layers": int(config["models"]["graph_layers"]),
        "gat_heads": int(config["models"]["gat_heads"]),
        "dropout": float(config["models"]["dropout"]),
    }


def _fit_or_load(
    model_name, kwargs, feature_columns, snapshots, config, device, checkpoint: Path, log_path: Path, seed: int
):
    import torch
    from .models import build_model

    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = build_model(model_name, **payload["model_kwargs"]).to(device)
        model.load_state_dict(payload["state_dict"])
        return model, float(payload["validation_mse"]), int(payload.get("epochs", 0)), payload

    set_seed(int(seed))
    model = build_model(model_name, **kwargs).to(device)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fit_start = time.perf_counter()
    model, validation_mse, history = _fit_neural_trial(
        model, snapshots["train"], snapshots["validation"], config, device, log_path
    )
    elapsed_seconds = time.perf_counter() - fit_start
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    )
    payload = {
        "state_dict": model.state_dict(), "model_name": model_name, "model_kwargs": kwargs,
        "feature_columns": feature_columns, "validation_mse": validation_mse,
        "epochs": len(history), "training_seed": int(seed), "config_hash": config_hash(config),
        "training_elapsed_seconds": float(elapsed_seconds),
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
        "amp_enabled": bool(config["models"].get("amp", False)) and device.type == "cuda",
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(payload, checkpoint)
    return model, validation_mse, len(history), payload


def train_neural_model(
    model_name: str, panel: pd.DataFrame, feature_columns: list[str], config: dict, project_root: str | Path
) -> pd.DataFrame:
    import torch

    if model_name not in NEURAL_MODELS:
        raise ValueError(model_name)
    root = Path(project_root)
    run_hash = config_hash(config)
    prediction_path = _prediction_path(root, model_name, run_hash)
    by_seed_path = root / "predictions" / f"{model_name}_{run_hash}_by_seed.parquet"
    success_path = prediction_path.with_suffix(".json")
    if prediction_path.exists() and by_seed_path.exists() and success_path.exists():
        return pd.read_parquet(prediction_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selection_seed = int(config["models"].get("hyperparameter_selection_seed", config["project"]["seed"]))
    seeds = [int(value) for value in config["models"].get("training_seeds", [selection_seed])]
    if selection_seed not in seeds:
        raise RuntimeError("hyperparameter_selection_seed must be included in training_seeds.")
    random_graph_model = _graph_mode(model_name) in {"random", "label_permutation"}
    graph_seeds = [int(value) for value in config["models"].get("random_graph_seeds", [])]
    if random_graph_model and not graph_seeds:
        raise RuntimeError(f"{model_name}: at least one random_graph_seed is required.")
    anchor_seed = int(config["models"].get("random_graph_anchor_training_seed", selection_seed))
    if anchor_seed not in seeds:
        raise RuntimeError("random_graph_anchor_training_seed must be included in training_seeds.")
    selection_graph_seed = graph_seeds[0] if random_graph_model else -1
    selection_snapshots = _snapshots(
        panel, feature_columns, config, selection_graph_seed, model_name, root, run_hash
    )
    if any(not selection_snapshots[split] for split in ["train", "validation", "test"]):
        raise RuntimeError(f"{model_name}: at least one split has no graph snapshots.")

    log_dir = root / "training_logs"
    checkpoint_dir = root / "checkpoints"
    trial_results = []
    selected = None
    for hidden_dim in config["models"]["hidden_dims"]:
        hidden_dim = int(hidden_dim)
        kwargs = _model_kwargs(config, feature_columns, hidden_dim)
        graph_tag = f"_g{selection_graph_seed}" if random_graph_model else ""
        checkpoint = checkpoint_dir / f"{model_name}_{run_hash}_selection_s{selection_seed}{graph_tag}_h{hidden_dim}.pt"
        log_path = log_dir / f"{model_name}_{run_hash}_selection_s{selection_seed}{graph_tag}_h{hidden_dim}.jsonl"
        model, validation_mse, epochs, payload = _fit_or_load(
            model_name, kwargs, feature_columns, selection_snapshots, config, device, checkpoint, log_path,
            selection_seed,
        )
        payload["selection_seed"] = selection_seed
        payload["graph_seed"] = selection_graph_seed
        _atomic_torch_save(payload, checkpoint)
        trial = {"hidden_dim": hidden_dim, "validation_mse": validation_mse, "checkpoint": str(checkpoint), "epochs": epochs}
        trial_results.append(trial)
        if selected is None or validation_mse < selected[0]:
            selected = (validation_mse, hidden_dim, checkpoint, kwargs)

    _, selected_hidden, selected_checkpoint, kwargs = selected
    seed_predictions = []
    seed_runs = []
    replicate_specs = build_replicate_specs(config, model_name)
    for seed, graph_seed, replicate_role in replicate_specs:
        snapshots = (
            selection_snapshots
            if seed == selection_seed and graph_seed == selection_graph_seed
            else _snapshots(panel, feature_columns, config, graph_seed, model_name, root, run_hash)
        )
        graph_tag = f"_g{graph_seed}" if random_graph_model else ""
        checkpoint = selected_checkpoint if seed == selection_seed and graph_seed == selection_graph_seed else (
            checkpoint_dir / f"{model_name}_{run_hash}_final_s{seed}{graph_tag}_h{selected_hidden}.pt"
        )
        log_path = log_dir / f"{model_name}_{run_hash}_final_s{seed}{graph_tag}_h{selected_hidden}.jsonl"
        model, validation_mse, epochs, payload = _fit_or_load(
            model_name, kwargs, feature_columns, snapshots, config, device, checkpoint, log_path, seed
        )
        payload["selected_hidden_dim"] = selected_hidden
        payload["training_seed"] = seed
        payload["graph_seed"] = graph_seed
        payload["replicate_role"] = replicate_role
        _atomic_torch_save(payload, checkpoint)
        seed_predictions.append(_predict_neural(
            model, snapshots, device, int(config["models"]["batch_graphs"]), model_name, seed, graph_seed,
            replicate_role, bool(config["models"].get("amp", False)) and device.type == "cuda",
        ))
        seed_runs.append({
            "training_seed": seed, "graph_seed": graph_seed, "validation_mse": validation_mse,
            "replicate_role": replicate_role, "epochs": epochs, "checkpoint": str(checkpoint),
            "training_elapsed_seconds": payload.get("training_elapsed_seconds"),
            "peak_gpu_memory_mb": payload.get("peak_gpu_memory_mb"),
            "amp_enabled": payload.get("amp_enabled", False),
            "parameter_count": payload.get("parameter_count"),
        })

    predictions_by_seed = pd.concat(seed_predictions, ignore_index=True)
    _atomic_parquet(predictions_by_seed, by_seed_path)
    main_predictions = predictions_by_seed.loc[predictions_by_seed["replicate_role"] == "main_training_seed"]
    predictions = main_predictions.groupby(["row_id", "model", "split"], as_index=False)["y_pred"].mean()
    _atomic_parquet(predictions, prediction_path)
    atomic_write_json(success_path, {
        "completed_at_utc": utc_now_iso(), "device": str(device),
        "selected_hidden_dim": selected_hidden, "selection_seed": selection_seed,
        "training_seeds": seeds, "random_graph_seeds": graph_seeds if random_graph_model else [],
        "random_graph_replication": "anchored_factorial" if random_graph_model else "not_applicable",
        "random_graph_anchor_training_seed": anchor_seed if random_graph_model else None,
        "seed_runs": seed_runs, "trials": trial_results,
        "aggregate": "arithmetic mean over main_training_seed replicas only; graph_randomization replicas are sensitivity diagnostics",
        "by_seed_predictions": str(by_seed_path),
    })
    return predictions


def combine_predictions(panel: pd.DataFrame, predictions: list[pd.DataFrame], project_root: str | Path) -> pd.DataFrame:
    combined = pd.concat(predictions, ignore_index=True)
    if combined.duplicated(["row_id", "model"]).any():
        raise ValueError("Duplicate predictions detected.")
    columns = [
        "row_id", "signal_date", "entry_date", "exit_date", "ts_code", "l1_code",
        "industry_taxonomy_source", "index_snapshot_date", "daily_basic_source_date", "total_mv", "split",
        "target_return", "benchmark_return", "target_excess_return", "decision_eligible", "label_available",
        "entry_price_available", "exit_price_available", "entry_buyable", "exit_sellable",
    ]
    available = [column for column in columns if column in panel.columns]
    output = combined.drop(columns=["split"], errors="ignore").merge(
        panel[available], on="row_id", how="left", validate="many_to_one"
    )
    output = output.sort_values(["model", "signal_date", "ts_code"])
    _atomic_parquet(output, Path(project_root) / "predictions.parquet")
    return output
