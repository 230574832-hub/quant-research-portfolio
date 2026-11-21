from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SECRET_KEYS = {"token", "tushare_token", "api_key", "secret", "password"}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, mode: str) -> dict[str, Any]:
    mode = mode.upper()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if mode not in raw.get("modes", {}):
        raise ValueError(f"Unknown run mode: {mode}")
    mode_cfg = raw["modes"][mode]
    base = {key: value for key, value in raw.items() if key != "modes"}
    merged = deep_merge(base, mode_cfg)
    merged["run_mode"] = mode
    return merged


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("***REDACTED***" if key.lower() in SECRET_KEYS else redact_secrets(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def config_hash(config: dict[str, Any], length: int = 12) -> str:
    payload = json.dumps(redact_secrets(config), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)


def atomic_write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(yaml.safe_dump(redact_secrets(payload), allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(temp, path)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint(root: str | Path, patterns: tuple[str, ...] = ("*.py", "*.yaml")) -> str:
    """Content fingerprint for code/config identity; paths are relative and sorted."""
    root = Path(root)
    digest = hashlib.sha256()
    paths = sorted({path for pattern in patterns for path in root.rglob(pattern) if path.is_file()})
    for path in paths:
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def parquet_directory_fingerprint(path: str | Path) -> str:
    """Stable content fingerprint for an immutable parquet checkpoint directory."""
    path = Path(path)
    digest = hashlib.sha256()
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(sha256_file(file).encode("ascii"))
    return digest.hexdigest()


def experiment_id(config: dict[str, Any]) -> str:
    protocol = config.get("formal_protocol", {}).get("protocol_id", "UNFROZEN")
    mode = config.get("run_mode", "UNKNOWN")
    return f"{protocol}_{mode}_{config_hash(config, 16)}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_project_dirs(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    names = [
        "cache/raw",
        "cache/reference",
        "prepared",
        "checkpoints",
        "predictions",
        "training_logs",
        "figures",
        "tables",
        "outputs",
        "source_snapshot",
    ]
    paths = {name.replace("/", "_"): root / name for name in names}
    root.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_environment() -> dict[str, Any]:
    import platform

    env: dict[str, Any] = {
        "captured_at_utc": utc_now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable_name": Path(sys.executable).name,
        "colab": "google.colab" in sys.modules,
    }
    try:
        import psutil

        env["cpu_count_logical"] = psutil.cpu_count(logical=True)
        env["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
    except Exception as exc:  # pragma: no cover - optional diagnostics
        env["system_probe_error"] = type(exc).__name__
    packages = {}
    for name in ["torch", "torch_geometric", "pandas", "numpy", "sklearn", "pyarrow", "tushare"]:
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception:
            packages[name] = None
    env["packages"] = packages
    try:
        import torch

        env["cuda_available"] = bool(torch.cuda.is_available())
        env["torch_cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu_memory_gb"] = round(props.total_memory / 2**30, 2)
            env["gpu_compute_capability"] = f"{props.major}.{props.minor}"
            env["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:
        env["cuda_available"] = False
        env["torch_probe_error"] = type(exc).__name__
    return env


def require_full_run_gpu(config: dict[str, Any]) -> None:
    if config["run_mode"] != "FULL_RUN" or not config["project"].get("require_cuda_for_full_run", True):
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("FULL_RUN requires a CUDA-enabled PyTorch installation.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "FULL_RUN stopped: CUDA GPU is unavailable. Local execution requires a working NVIDIA driver and "
            "CUDA-enabled PyTorch; Colab remains a backup runtime."
        )


def validate_formal_protocol(config: dict[str, Any]) -> None:
    """Fail closed if a FULL_RUN departs from the preregistered protocol."""
    protocol = config.get("formal_protocol", {})
    if config.get("run_mode") != "FULL_RUN" or not protocol.get("frozen", False):
        return
    mismatches = []
    expected_splits = protocol.get("expected_splits", {})
    actual_splits = {key: config["splits"].get(key) for key in expected_splits}
    if actual_splits != expected_splits:
        mismatches.append(f"splits={actual_splits!r}")
    checks = [
        ("models", list(config["models"].get("enabled", [])), list(protocol.get("expected_models", []))),
        ("training_seeds", list(config["models"].get("training_seeds", [])), list(protocol.get("expected_training_seeds", []))),
        ("random_graph_seeds", list(config["models"].get("random_graph_seeds", [])), list(protocol.get("expected_random_graph_seeds", []))),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            mismatches.append(f"{name}={actual!r}")
    expected_model_config = protocol.get("expected_model_config", {})
    actual_model_config = {key: config["models"].get(key) for key in expected_model_config}
    if actual_model_config != expected_model_config:
        mismatches.append(f"model_config={actual_model_config!r}")
    actual_primary = [
        item.get("name") for item in config.get("evaluation", {}).get("paired_rankic_comparisons", [])
        if item.get("primary", False)
    ]
    if actual_primary != list(protocol.get("expected_primary_comparisons", [])):
        mismatches.append(f"primary_comparisons={actual_primary!r}")
    actual_replication = config["models"].get("random_graph_replication")
    if actual_replication != protocol.get("random_graph_replication"):
        mismatches.append(f"random_graph_replication={actual_replication!r}")
    if config["models"].get("random_graph_anchor_training_seed") not in config["models"].get("training_seeds", []):
        mismatches.append("random_graph_anchor_training_seed must be one of training_seeds")
    expected_project = protocol.get("expected_project_config", {})
    actual_project = {key: config["project"].get(key) for key in expected_project}
    if actual_project != expected_project:
        mismatches.append(f"project_config={actual_project!r}")
    if mismatches:
        protocol_id = protocol.get("protocol_id", "unnamed")
        raise RuntimeError(f"FULL_RUN protocol {protocol_id} is frozen; mismatches: " + "; ".join(mismatches))

