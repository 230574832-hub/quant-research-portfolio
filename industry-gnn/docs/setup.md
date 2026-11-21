# Industry-Aware Stock Return Prediction with GNNs — PIT/HIST V4

## Current entry point

Use **`Industry_GNN_Research_PIT_HIST_V4_From_Zero.ipynb`** for the current study. It is a self-contained Python 3.11 / Miniconda / VS Code Notebook: it restores its own source tree, defines `CONFIG` immediately, can reuse an existing project cache, and only requests a TuShare token when required cache files are absent.

The V4 Notebook adds three research-critical changes:

- a field-by-field Point-in-Time evidence table that keeps unavailable vendor timestamps as `PARTIAL` or `UNVERIFIED`;
- conservative opening execution, explicit blocked-order tests, historical stamp duty, and CSI 300 benchmark returns aligned to the portfolio's holding interval;
- a common repaired Top-10 comparison for Ridge, Shared MLP, Industry MLP, Residual-GCN, Residual-GAT Complete, Residual-GAT Sparse, and `HIST-Factor`.

`HIST-Factor` adapts HIST's predefined-concept, hidden-concept, individual-information and doubly residual structure to this project's 12 weekly factors and industry concepts. It is deliberately not described as an exact reproduction of the paper's Alpha360/GRU and industry-plus-business-concept input.

Run `SMOKE_TEST` first. Only after its PIT table and execution tests pass should you change `RUN_MODE` to `FULL_RUN` and set `CONFIRM_FULL_RUN=True`. No model performance result is included in this source package.

This is the local-first, reproducible implementation of **Beyond Individual Stock Factors: Graph Neural Networks for Industry-Aware Stock Return Prediction**. The primary target is a Windows or Linux laptop with an NVIDIA RTX 4070 Laptop GPU. Google Colab remains a backup path through `Industry_GNN_Research_Colab.ipynb`.

No old Smoke Test number is treated as a new result. FULL_RUN is gated, uses an untouched 2024–2025 holdout, and stops if the frozen protocol changes.

### Truly from-zero single-Notebook option

The V4 Notebook named above supersedes `Industry_GNN_Research_Local_From_Zero.ipynb`. It embeds the complete source and frozen `config.yaml`; its first executable sections install/verify every dependency, create the project directory, initialize `CONFIG`, securely request the TuShare Token, and define/download `REFERENCE` and `MARKET`. It does not assume a cloud Bundle, cache, or adjacent source folder. The only prerequisite is selecting a Python 3.11 Miniconda kernel in VS Code.

## 0. Miniconda + VS Code graphical workflow

This is the recommended workflow if you do not want to type training commands manually:

1. Install Miniconda, VS Code, and the Microsoft **Python**, **Python Environments**, and **Jupyter** extensions. Opening this project makes VS Code recommend them automatically.
2. In VS Code choose **File → Open Folder** and open the extracted `industry_gnn_project` folder itself, not its parent folder.
3. Open the Python sidebar, expand **Environment Managers**, click `+`, choose **Conda**, Python 3.11, and name it `industry-gnn`. Alternatively use `Ctrl+Shift+P → Python: Create Environment → Conda` and select `environment.yml` when offered.
4. Click the Python interpreter shown on the VS Code status bar and select `industry-gnn`.
5. Open `scripts/setup_conda_vscode.py` and click the triangular **Run Python File** button in the upper-right corner. It installs the pinned CUDA/PyG stack into the selected environment and runs the GPU test. It refuses to modify Conda `base`.
6. Open `Industry_GNN_Research_PIT_HIST_V4_From_Zero.ipynb`, click **Select Kernel** in the upper-right, choose **Python Environments → industry-gnn**, and run cells from top to bottom. Leave `RUN_MODE="SMOKE_TEST"` initially. The TuShare token is entered through a hidden notebook input widget only if the matching cache is absent.
7. After Smoke passes, change the two mode values in the notebook to `RUN_MODE="FULL_RUN"` and `CONFIRM_FULL_RUN=True`, then click **Run All** again.

If you prefer running `run_local.py` instead of the Notebook, open the VS Code **Run and Debug** sidebar and choose `0 - Check RTX/CUDA`, `1 - Smoke Preflight`, or `3 - FULL RUN (confirmed)`. VS Code will show a program-output terminal because hidden Token input and long-running logs need an interactive stream, but you do not need to enter any shell command yourself.

## 1. Install on Windows

Prerequisites: current NVIDIA driver, 64-bit Python 3.11, at least 25 GB free disk, and preferably 16 GB system RAM.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

The script creates `.venv`, installs `torch==2.12.1` from the official CUDA 12.6 wheel index, installs `torch-geometric==2.8.0` plus pinned dependencies, then performs a CUDA GCN/GAT forward-and-backward test. It does **not** install a separate CUDA Toolkit.

Linux: `bash scripts/setup_linux.sh`.

If GPU detection fails, stop there. Do not launch FULL_RUN on CPU.

## 2. TuShare credential

The safest default is to run without setting an environment variable. The program requests the token in a hidden prompt and keeps it in memory only. Do not paste it into a notebook cell, YAML file, command line, or log. If a token has already appeared in chat, rotate it before the formal run.

For a temporary shell-only variable, clear it after the run:

```powershell
$env:TUSHARE_TOKEN = Read-Host -AsSecureString | ConvertFrom-SecureString -AsPlainText
# run the command, then:
Remove-Item Env:TUSHARE_TOKEN
```

## 3. Stage A–C local preflight

```powershell
.\scripts\run_preflight.ps1
```

This runs the real-data `SMOKE_TEST` end to end on CUDA, including all eight model variants, checkpoint creation, backtesting, statistics, figures, and ZIP packaging. Re-run the same command once: completed artifacts with the same source/data/config fingerprint must be reused.

The current controlled build environment could not access the laptop GPU or TuShare, so a successful `environment_preflight.json` and real-data Smoke output must be produced on the laptop before FULL_RUN.

## 4. Full data and staged training

```powershell
.\scripts\run_data.ps1
.\scripts\run_train.ps1 ridge,shared_mlp,industry_mlp
.\scripts\run_train.ps1 gcn,gat
.\scripts\run_train.ps1 gcn_random,gat_random,gat_self_only
```

Or, after Stage A–C pass, run/resume everything:

```powershell
.\scripts\run_full.ps1 -ConfirmFullRun
```

Direct equivalent: `.\.venv\Scripts\python.exe run_local.py full --confirm-full-run`. The explicit confirmation flag prevents an accidental hours-long FULL_RUN.

## 5. Resume after interruption

Run the same command again. TuShare calls are cached per request using atomic Parquet writes; prepared panels, graph snapshots, per-hyperparameter checkpoints, per-seed checkpoints, and predictions are content-keyed. A file is only considered complete when its success sidecar exists. Partial `.part` files are ignored and regenerated.

Cache/checkpoint identity includes source hash, configuration-file hash, data-file SHA-256 fingerprint, feature schema, universe/date rules, hyperparameters, training seeds, and graph seeds. Old Smoke checkpoints therefore cannot silently satisfy FULL_RUN. Do not delete `cache/`, `prepared/`, `checkpoints/`, or `predictions/` when resuming.

## 6. Frozen experiment

- Training: 2016-01-01 through 2021-12-31.
- Validation/development: 2022-01-01 through 2023-12-31.
- Final holdout: 2024-01-01 through 2025-12-31.
- No train+validation refit: validation is used consistently for hyperparameter selection and early stopping; the test period never influences weights or settings.
- V4 common-backtest models: Ridge, Shared MLP, Industry MLP, Residual-GCN Complete, Residual-GAT Complete, Residual-GAT Sparse, and HIST-Factor.
- Random Sparse and Self-only residual GAT are still trained as mechanism diagnostics but are not substituted for the fixed seven-model V4 Top-10 table.
- Neural training seeds: 42, 3407, 2026.
- Random graph seeds: 1042, 13407, 12026, using the anchored design in `experiment_protocol.md`.
- Primary strategy: weekly Top-10 equal weight, historical Base costs. Top-5/Top-20 and alternative costs are sensitivity analyses.
- Maximum 150 epochs, patience 20, identical neural search budget, AMP disabled in the frozen protocol because it was not validated on the target laptop before freezing.

See `experiment_protocol.md`, `research_design.md`, `data_leakage_audit.md`, and `backtest_audit.md`.

## 7. Output contract

Successful completion creates these root artifacts and an immutable mirror under `outputs/<experiment_id>/`:

- `metrics.csv`, `paired_rankic_tests.csv`, `ablation_results.csv`
- `portfolio_returns.csv`, `trade_logs.csv`, `predictions.parquet`
- `training_seed_results.csv`, `industry_neutral_metrics.csv`
- `graph_audit_summary.json`, `experiment_config.yaml`, `environment.json`
- `research_summary.md`, `research_results_bundle.zip`

Raw cache, graph cache, and model weights remain outside the ZIP; `file_manifest.csv` records exclusions. Return `research_results_bundle.zip` to ChatGPT for academic review.

## 8. Expected resources

The models are small and graphs are weekly CSI 300 cross-sections. Expected peak model-training VRAM is roughly 1–3 GB; allow 4 GB headroom. On an RTX 4070 Laptop GPU, a cached FULL_RUN is estimated at about 5–13 hours including the HIST-Factor adaptation; first-time TuShare download/feature construction may add 1–4 hours depending on API limits and network speed. These are planning ranges, not measured results. HIST contains all-stock similarity operations and may be the slowest group. Actual elapsed time and peak allocated GPU memory must be taken from the target run.

## 9. Project status

Completed here: source/design audit, V4 protocol/configuration, PIT evidence reporting, conservative execution and benchmark fixes, HIST-Factor implementation, local setup scripts, checkpoint/randomness design, static compilation, 23 synthetic/contract tests, and Notebook structural validation.

Still requires the laptop: actual RTX 4070/driver/VRAM verification, PyTorch/PyG CUDA test, TuShare real-data preflight, resume timing check, and FULL_RUN. No formal outcome or model superiority is claimed before those artifacts exist.
