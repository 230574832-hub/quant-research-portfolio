from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="environment_preflight.json")
    args = parser.parse_args()
    result = {
        "python": sys.version,
        "python_executable_name": Path(sys.executable).name,
        "disk_free_gb": round(shutil.disk_usage(Path.cwd()).free / 2**30, 2),
    }
    try:
        import psutil
        result["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
        result["ram_available_gb"] = round(psutil.virtual_memory().available / 2**30, 2)
    except Exception as exc:
        result["ram_probe_error"] = type(exc).__name__
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        result["nvidia_smi"] = probe.stdout.strip()
    except Exception as exc:
        result["nvidia_smi_error"] = type(exc).__name__
    try:
        import torch
        import torch_geometric
        from torch_geometric.data import Data
        from torch_geometric.nn import GATConv, GCNConv

        result.update({
            "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
            "torch_geometric": torch_geometric.__version__, "cuda_available": torch.cuda.is_available(),
        })
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(0)
        result.update({
            "gpu_name": properties.name,
            "gpu_memory_gb": round(properties.total_memory / 2**30, 2),
            "compute_capability": f"{properties.major}.{properties.minor}",
        })
        x = torch.randn(12, 8, device=device, requires_grad=True)
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9], [1, 0, 2, 1, 3, 2, 5, 4, 7, 6, 9, 8]],
            dtype=torch.long, device=device,
        )
        gcn = GCNConv(8, 16).to(device)
        gat = GATConv(8, 8, heads=2, concat=False).to(device)
        loss = (gcn(x, edge_index).square().mean() + gat(x, edge_index).square().mean())
        loss.backward()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            amp_value = gat(x.detach(), edge_index).float().mean()
        torch.cuda.synchronize()
        result["pyg_cuda_forward_backward"] = True
        result["amp_smoke_finite"] = bool(torch.isfinite(amp_value).item())
        result["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 2)
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["gpu_stack_error"] = f"{type(exc).__name__}: {exc}"
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
