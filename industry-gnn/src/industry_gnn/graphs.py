from __future__ import annotations

from collections import Counter
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd


def industry_edges(industry_codes: Iterable[str | None], include_self_loops: bool = True) -> np.ndarray:
    codes = list(industry_codes)
    groups: dict[str, list[int]] = {}
    for idx, code in enumerate(codes):
        if code is None or pd.isna(code):
            continue
        groups.setdefault(str(code), []).append(idx)
    edges: list[tuple[int, int]] = []
    for nodes in groups.values():
        edges.extend((source, target) for source in nodes for target in nodes if source != target)
    if include_self_loops:
        edges.extend((idx, idx) for idx in range(len(codes)))
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def self_loop_edges(node_count: int) -> np.ndarray:
    nodes = np.arange(int(node_count), dtype=np.int64)
    return np.vstack([nodes, nodes])


def permute_industry_labels(industry_codes: Iterable[str | None], seed: int) -> list[str | None]:
    """Randomly reassign the observed labels while preserving every clique size exactly."""
    codes = np.asarray(list(industry_codes), dtype=object)
    if not len(codes):
        return []
    permutation = np.random.default_rng(int(seed)).permutation(len(codes))
    return codes[permutation].tolist()


def label_permuted_industry_edges(
    industry_codes: Iterable[str | None], seed: int, include_self_loops: bool = True
) -> np.ndarray:
    return industry_edges(permute_industry_labels(industry_codes, seed), include_self_loops=include_self_loops)


def degree_preserving_rewire(edge_index: np.ndarray, node_count: int, seed: int, swaps_per_edge: int = 3) -> np.ndarray:
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    undirected = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in edge_index.T if a != b}
    graph.add_edges_from(undirected)
    edge_count = graph.number_of_edges()
    if edge_count >= 2:
        requested = max(1, int(swaps_per_edge) * edge_count)
        try:
            nx.double_edge_swap(graph, nswap=requested, max_tries=max(requested * 20, 100), seed=seed)
        except nx.NetworkXAlgorithmError:
            # Small/saturated graphs may not admit the requested number of swaps; preserve the valid partial graph.
            pass
    directed = []
    for source, target in graph.edges():
        directed.extend([(source, target), (target, source)])
    directed.extend((idx, idx) for idx in range(node_count))
    return np.asarray(directed, dtype=np.int64).T if directed else np.empty((2, 0), dtype=np.int64)


def build_graph_snapshots(
    panel: pd.DataFrame,
    feature_columns: list[str],
    split: str,
    target_column: str = "target_excess_return",
    graph_mode: str = "industry",
    seed: int = 0,
    swaps_per_edge: int = 3,
):
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - exercised in Colab
        raise RuntimeError("PyTorch and torch-geometric are required to build graph snapshots.") from exc

    snapshots = []
    selected = panel.loc[panel["split"] == split]
    for date_idx, (signal_date, section) in enumerate(selected.groupby("signal_date", sort=True)):
        section = section.sort_values("ts_code")
        edge_index = industry_edges(section["l1_code"], include_self_loops=True)
        if graph_mode == "random":
            edge_index = degree_preserving_rewire(
                edge_index,
                len(section),
                seed=seed + date_idx,
                swaps_per_edge=swaps_per_edge,
            )
        elif graph_mode == "label_permutation":
            edge_index = label_permuted_industry_edges(
                section["l1_code"], seed=seed + date_idx, include_self_loops=True
            )
        elif graph_mode == "self":
            edge_index = self_loop_edges(len(section))
        elif graph_mode != "industry":
            raise ValueError(f"Unknown graph_mode: {graph_mode}")
        x = torch.tensor(section[feature_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
        agg = torch.tensor(
            section[[f"peer_{feature}" for feature in feature_columns]].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        label_mask = section.get("label_available", section[target_column].notna()).fillna(False).to_numpy(dtype=bool)
        # Keep finite tensor values for unlabeled decision rows; label_mask excludes them from loss/metrics.
        target = section[target_column].fillna(0.0).to_numpy(dtype=np.float32)
        y = torch.tensor(target, dtype=torch.float32)
        data = Data(
            x=x,
            agg_x=agg,
            y=y,
            label_mask=torch.tensor(label_mask, dtype=torch.bool),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            row_id=torch.tensor(section["row_id"].to_numpy(dtype=np.int64), dtype=torch.long),
            num_nodes=len(section),
        )
        data.signal_ordinal = torch.full((len(section),), int(pd.Timestamp(signal_date).toordinal()), dtype=torch.long)
        snapshots.append(data)
    return snapshots


def graph_audit(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    all_degrees = []
    for signal_date, section in panel.groupby("signal_date", sort=True):
        edges = industry_edges(section["l1_code"], include_self_loops=True)
        non_self = edges[:, edges[0] != edges[1]] if edges.size else edges
        degrees = Counter(non_self[0].tolist())
        degree_values = [degrees.get(idx, 0) for idx in range(len(section))]
        all_degrees.extend(degree_values)
        industry_sizes = section.loc[section["l1_code"].notna()].groupby("l1_code").size()
        undirected = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in non_self.T}
        graph = nx.Graph()
        graph.add_nodes_from(range(len(section)))
        graph.add_edges_from(undirected)
        possible_directed = len(section) * max(len(section) - 1, 1)
        rows.append(
            {
                "signal_date": signal_date,
                "nodes": len(section),
                "directed_edges_including_self": int(edges.shape[1]),
                "directed_peer_edges": int(non_self.shape[1]),
                "peer_edge_density": float(non_self.shape[1] / possible_directed) if len(section) > 1 else 0.0,
                "mean_peer_degree": float(np.mean(degree_values)) if degree_values else 0.0,
                "max_peer_degree": int(max(degree_values)) if degree_values else 0,
                "isolated_nodes": int(sum(value == 0 for value in degree_values)),
                "connected_components": int(nx.number_connected_components(graph)) if len(section) else 0,
                "industries": int(section["l1_code"].nunique(dropna=True)),
                "mean_industry_size": float(industry_sizes.mean()) if len(industry_sizes) else 0.0,
                "max_industry_size": int(industry_sizes.max()) if len(industry_sizes) else 0,
                "missing_industry": int(section["l1_code"].isna().sum()),
            }
        )
    audit = {
        "dates": len(rows),
        "mean_degree": float(np.mean(all_degrees)) if all_degrees else 0.0,
        "median_degree": float(np.median(all_degrees)) if all_degrees else 0.0,
        "p95_degree": float(np.quantile(all_degrees, 0.95)) if all_degrees else 0.0,
        "mean_nodes_per_date": float(np.mean([row["nodes"] for row in rows])) if rows else 0.0,
        "mean_peer_edge_density": float(np.mean([row["peer_edge_density"] for row in rows])) if rows else 0.0,
        "mean_isolated_nodes": float(np.mean([row["isolated_nodes"] for row in rows])) if rows else 0.0,
        "self_loops": True,
        "edge_semantics": "directed representation of same-industry undirected cliques plus explicit self-loops",
    }
    return pd.DataFrame(rows), audit


def random_graph_audit(
    panel: pd.DataFrame, seeds: list[int], swaps_per_edge: int, max_dates: int = 12
) -> dict:
    """Audit both topology rewiring and fixed-clique-size label randomization."""
    dates = pd.DatetimeIndex(sorted(panel["signal_date"].unique()))
    if not len(dates) or not seeds:
        return {
            "degree_preserving_rewire": {"sampled_dates": 0, "graph_seeds": seeds, "degree_sequence_mismatches": 0},
            "label_permutation": {"sampled_dates": 0, "graph_seeds": seeds, "clique_size_mismatches": 0},
        }
    indexes = np.unique(np.linspace(0, len(dates) - 1, min(max_dates, len(dates)), dtype=int))
    sampled_dates = dates[indexes]
    mismatches = 0
    rows = 0
    density_changes = []
    component_changes = []
    label_clique_mismatches = 0
    label_edge_mismatches = 0
    label_component_mismatches = 0
    label_assignment_changes = []
    for seed in seeds:
        for date_idx, date in enumerate(sampled_dates):
            section = panel.loc[panel["signal_date"] == date].sort_values("ts_code")
            original_labels = section["l1_code"].tolist()
            real = industry_edges(original_labels, include_self_loops=True)
            random_edges = degree_preserving_rewire(
                real, len(section), int(seed) + date_idx, swaps_per_edge=int(swaps_per_edge)
            )
            real_peer = real[:, real[0] != real[1]]
            random_peer = random_edges[:, random_edges[0] != random_edges[1]]
            real_degree = np.bincount(real_peer[0], minlength=len(section))
            random_degree = np.bincount(random_peer[0], minlength=len(section))
            mismatches += int(not np.array_equal(real_degree, random_degree))
            denominator = len(section) * max(len(section) - 1, 1)
            density_changes.append((random_peer.shape[1] - real_peer.shape[1]) / denominator)
            real_graph = nx.Graph()
            random_graph = nx.Graph()
            real_graph.add_nodes_from(range(len(section)))
            random_graph.add_nodes_from(range(len(section)))
            real_graph.add_edges_from(real_peer.T.tolist())
            random_graph.add_edges_from(random_peer.T.tolist())
            component_changes.append(
                nx.number_connected_components(random_graph) - nx.number_connected_components(real_graph)
            )

            permuted_labels = permute_industry_labels(original_labels, int(seed) + date_idx)
            label_edges = industry_edges(permuted_labels, include_self_loops=True)
            original_sizes = sorted(Counter(str(value) for value in original_labels if not pd.isna(value)).values())
            permuted_sizes = sorted(Counter(str(value) for value in permuted_labels if not pd.isna(value)).values())
            label_clique_mismatches += int(original_sizes != permuted_sizes)
            label_edge_mismatches += int(real.shape[1] != label_edges.shape[1])
            label_peer = label_edges[:, label_edges[0] != label_edges[1]]
            label_graph = nx.Graph()
            label_graph.add_nodes_from(range(len(section)))
            label_graph.add_edges_from(label_peer.T.tolist())
            label_component_mismatches += int(
                nx.number_connected_components(label_graph) != nx.number_connected_components(real_graph)
            )
            if len(original_labels):
                label_assignment_changes.append(float(np.mean([
                    (str(a) != str(b)) if not (pd.isna(a) and pd.isna(b)) else False
                    for a, b in zip(original_labels, permuted_labels)
                ])))
            rows += 1
    common = {
        "sampled_dates": int(len(sampled_dates)), "graph_seeds": [int(seed) for seed in seeds],
        "seed_date_checks": int(rows),
        "audit_scope": "up to 12 evenly spaced dates per graph seed; all dates are randomized during model training",
    }
    return {
        "degree_preserving_rewire": {
            **common,
            "degree_sequence_mismatches": int(mismatches),
            "maximum_absolute_density_change": float(np.max(np.abs(density_changes))) if density_changes else 0.0,
            "mean_component_count_change": float(np.mean(component_changes)) if component_changes else 0.0,
        },
        "label_permutation": {
            **common,
            "clique_size_mismatches": int(label_clique_mismatches),
            "directed_edge_count_mismatches": int(label_edge_mismatches),
            "connected_component_count_mismatches": int(label_component_mismatches),
            "mean_node_label_change_fraction": float(np.mean(label_assignment_changes)) if label_assignment_changes else 0.0,
            "semantics": "industry labels are permuted across stocks; clique-size multiset and missing-label count are fixed",
        },
    }

