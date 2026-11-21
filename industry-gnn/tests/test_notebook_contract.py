from __future__ import annotations

import json
import re
from pathlib import Path


def test_notebook_contains_required_stages():
    notebook_path = Path(__file__).parents[2] / "Industry_GNN_Research_Colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required = [
        "SMOKE_TEST", "FULL_RUN", "TUSHARE_TOKEN", "drive.mount", "data_stage", "feature_stage",
        "ridge", "shared_mlp", "industry_mlp", "gcn", "gat", "gcn_random", "gat_random", "gat_self_only",
        "label_available", "paired_rankic_tests", "seed_metrics", "execution_constraint_audit",
        "universe_timing_audit", "GAT − Industry MLP", "random_graph_seeds",
        "2024–2025", "historical", "evaluation_stage", "research_results_bundle.zip",
    ]
    for marker in required:
        assert marker in source
    assert not re.search(r"(?i)tushare_token\s*=\s*['\"][A-Za-z0-9_-]{20,}['\"]", source)


def test_local_notebook_teaches_and_runs_frozen_protocol():
    notebook_path = Path(__file__).parents[1] / "Industry_GNN_Research_Local.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required = [
        "SMOKE_TEST", "FULL_RUN", "CONFIRM_FULL_RUN", "torch.cuda.is_available", "get_tushare_token",
        "data_stage", "feature_stage", "graph_audit_stage", "ridge", "shared_mlp", "industry_mlp",
        "gcn_random", "gat_random", "gat_self_only", "GAT vs Industry MLP", "RankIC",
        "Transformer", "Holm", "industry_neutral_metrics", "training_seed_results.csv",
        "research_results_bundle.zip", "2024–2025",
    ]
    for marker in required:
        assert marker in source
    assert not re.search(r"(?i)tushare_token\s*=\s*['\"][A-Za-z0-9_-]{20,}['\"]", source)


def test_from_zero_notebook_embeds_source_dependencies_config_and_data_stage():
    notebook_path = Path(__file__).parents[1] / "Industry_GNN_Research_Local_From_Zero.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required = [
        "torch==2.12.1", "torch-geometric==2.8.0", "tushare==1.4.24",
        "PROJECT_ARCHIVE_B64", "config.yaml exists", "CONFIG = initialize",
        "REFERENCE, MARKET, DATA_KEY", "data_stage(PRO, CONFIG, PROJECT_ROOT)",
        "feature_stage", "graph_audit_stage", "train_model_stage", "evaluation_stage",
        "research_results_bundle.zip",
    ]
    for marker in required:
        assert marker in source
    assert not re.search(r"(?i)tushare_token\s*=\s*['\"][A-Za-z0-9_-]{20,}['\"]", source)


def test_pit_hist_v4_notebook_is_standalone_and_honest():
    notebook_path = Path(__file__).parents[1] / "Industry_GNN_Research_PIT_HIST_V4_From_Zero.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required = [
        "PROJECT_ARCHIVE_B64", "CONFIG = initialize", "build_point_in_time_audit",
        "PARTIAL/UNVERIFIED", "conservative opening-limit execution", "backtest_carries_blocked_sale",
        "res_gcn_complete", "res_gat_complete", "res_gat_sparse", "hist_factor",
        "primary_portfolio_top_k", "paired_rankic_tests.csv", "ablation_results.csv",
        "training_seed_results.csv", "execution_constraint_audit.csv",
        "research_results_bundle.zip",
    ]
    for marker in required:
        assert marker in source
    assert not re.search(r"(?i)tushare_token\s*=\s*['\"][A-Za-z0-9_-]{20,}['\"]", source)

