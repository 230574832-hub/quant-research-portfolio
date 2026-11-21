from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn


def seed_all(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def prepare_hist_samples(samples: list[dict], panel: pd.DataFrame) -> list[dict]:
    """Attach contemporaneous industry concepts and market-cap weights to V3 samples."""
    prepared = []
    dates = {
        pd.Timestamp(date): section.sort_values("ts_code")
        for date, section in panel.groupby("signal_date", sort=True)
    }
    for sample in samples:
        section = dates[pd.Timestamp(sample["date"])]
        if list(section["ts_code"].astype(str)) != list(map(str, sample["ts_code"])):
            raise ValueError("HIST sample order does not match the research panel.")
        labels = section["l1_code"].fillna("UNKNOWN").astype(str)
        known = sorted(value for value in labels.unique() if value != "UNKNOWN")
        lookup = {value: idx for idx, value in enumerate(known)}
        concept_id = labels.map(lookup).fillna(-1).to_numpy(dtype=np.int64)
        if "total_mv" in section:
            market_cap = pd.to_numeric(section["total_mv"], errors="coerce").to_numpy(dtype=float)
            positive = market_cap[np.isfinite(market_cap) & (market_cap > 0)]
            fallback = float(np.median(positive)) if len(positive) else 1.0
            market_cap = np.where(np.isfinite(market_cap) & (market_cap > 0), market_cap, fallback)
        else:
            market_cap = np.ones(len(section), dtype=float)
        item = dict(sample)
        item["concept_id"] = torch.tensor(concept_id, dtype=torch.long)
        item["market_cap"] = torch.tensor(market_cap, dtype=torch.float32)
        prepared.append(item)
    return prepared


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    if logits.numel() == 0:
        return logits
    masked = logits.masked_fill(~mask, -1e9)
    weights = torch.softmax(masked, dim=dim) * mask.to(logits.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp(min=1e-12)


class HISTFactorRegressor(nn.Module):
    """HIST-inspired factor model for the project's weekly 12-factor panel.

    It retains the paper's predefined-concept, hidden-concept, individual-information
    and doubly-residual mechanism.  It is deliberately named HIST-Factor rather than
    an exact HIST reproduction because this project uses an MLP factor encoder and
    industry concepts only, not Alpha360's two-layer GRU plus industry/business concepts.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.predefined_correct = nn.Linear(hidden_dim, hidden_dim)
        self.predefined_shared = nn.Linear(hidden_dim, hidden_dim)
        self.predefined_backcast = nn.Linear(hidden_dim, hidden_dim)
        self.predefined_forecast = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_correct = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_shared = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_backcast = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_forecast = nn.Linear(hidden_dim, hidden_dim)
        self.individual_forecast = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.LeakyReLU(0.1)

    @staticmethod
    def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = torch.nn.functional.normalize(left, dim=-1, eps=1e-12)
        right = torch.nn.functional.normalize(right, dim=-1, eps=1e-12)
        return left @ right.T

    def _predefined_module(
        self, h0: torch.Tensor, concept_id: torch.Tensor, market_cap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = concept_id >= 0
        concept_count = int(concept_id[valid].max().item()) + 1 if valid.any() else 0
        if concept_count == 0:
            shared = torch.zeros_like(h0)
        else:
            caps = market_cap.clamp(min=0)
            denominators = h0.new_zeros(concept_count).index_add_(0, concept_id[valid], caps[valid])
            weights = caps[valid] / denominators[concept_id[valid]].clamp(min=1e-12)
            initial = h0.new_zeros((concept_count, h0.shape[1])).index_add_(
                0, concept_id[valid], weights[:, None] * h0[valid]
            )
            # Paper Sec. 4.2.2: all-stock similarity corrects incomplete/noisy
            # predefined memberships before concepts return information to stocks.
            similarity = self._cosine(h0, initial)
            correction_weights = torch.softmax(similarity, dim=0)
            corrected = self.activation(self.predefined_correct(correction_weights.T @ h0))
            stock_weights = torch.softmax(self._cosine(h0, corrected), dim=1)
            shared = self.activation(self.predefined_shared(stock_weights @ corrected))
        backcast = self.activation(self.predefined_backcast(shared))
        forecast = self.activation(self.predefined_forecast(shared))
        return backcast, forecast

    def _hidden_module(self, h1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(h1)
        if count < 2:
            shared = torch.zeros_like(h1)
        else:
            # Paper Sec. 4.3: each stock initializes a hidden concept, then every
            # stock links to its most similar non-self concept; retained prototypes
            # also keep their originating stock.
            similarity = self._cosine(h1, h1)
            nonself = similarity.masked_fill(torch.eye(count, dtype=torch.bool, device=h1.device), -1e9)
            selected = nonself.argmax(dim=1)
            connections = torch.zeros((count, count), dtype=torch.bool, device=h1.device)
            rows = torch.arange(count, device=h1.device)
            connections[rows, selected] = True
            used = torch.unique(selected)
            connections[used, used] = True
            to_concept = _masked_softmax(similarity, connections, dim=0)
            concepts = self.activation(self.hidden_correct(to_concept.T @ h1))
            back_similarity = self._cosine(h1, concepts)
            to_stock = _masked_softmax(back_similarity, connections, dim=1)
            shared = self.activation(self.hidden_shared(to_stock @ concepts))
        backcast = self.activation(self.hidden_backcast(shared))
        forecast = self.activation(self.hidden_forecast(shared))
        return backcast, forecast

    def forward(self, sample: dict) -> torch.Tensor:
        h0 = self.encoder(sample["x"])
        back0, forecast0 = self._predefined_module(
            h0, sample["concept_id"], sample["market_cap"]
        )
        h1 = h0 - back0
        back1, forecast1 = self._hidden_module(h1)
        h2 = h1 - back1
        forecast2 = self.activation(self.individual_forecast(h2))
        return self.output(forecast0 + forecast1 + forecast2).squeeze(-1)


def _to_device(sample: dict, device) -> dict:
    return {
        **sample,
        **{key: value.to(device) for key, value in sample.items() if isinstance(value, torch.Tensor)},
    }


def _mse(model, samples: list[dict], device) -> float:
    model.eval()
    total = 0.0
    observations = 0
    with torch.no_grad():
        for sample in samples:
            batch = _to_device(sample, device)
            mask = batch["mask"].bool()
            if mask.any():
                error = model(batch)[mask] - batch["y"][mask]
                total += float(torch.sum(error.square()))
                observations += int(mask.sum())
    return total / observations if observations else float("inf")


def train_hist_factor(
    samples: list[dict],
    input_dim: int,
    seed: int,
    hidden_dim: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    batch_dates: int,
    device,
) -> tuple[HISTFactorRegressor, list[dict]]:
    seed_all(seed)
    model = HISTFactorRegressor(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    training = [sample for sample in samples if sample["period"] == "train"]
    validation = [sample for sample in samples if sample["period"] == "validation"]
    if not training or not validation:
        raise ValueError("HIST-Factor requires training and validation dates.")
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = np.random.default_rng(int(seed) + epoch).permutation(len(training))
        for chunk in np.array_split(order, max(1, int(np.ceil(len(order) / int(batch_dates))))):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = None
            observations = 0
            for index in chunk:
                batch = _to_device(training[int(index)], device)
                mask = batch["mask"].bool()
                if not mask.any():
                    continue
                error = model(batch)[mask] - batch["y"][mask]
                value = torch.sum(error.square())
                loss_sum = value if loss_sum is None else loss_sum + value
                observations += int(mask.sum())
            if observations:
                (loss_sum / observations).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        validation_mse = _mse(model, validation, device)
        history.append({"epoch": epoch, "validation_mse": validation_mse})
        if validation_mse < best_loss - 1e-8:
            best_loss = validation_mse
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            break
    if best_state is None:
        raise RuntimeError("HIST-Factor did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    return model, history


def predict_hist_factor(
    model: HISTFactorRegressor,
    samples: list[dict],
    seed: int,
    device,
    include_holdout: bool = False,
) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            if sample["period"] == "holdout" and not include_holdout:
                continue
            prediction = model(_to_device(sample, device)).detach().cpu().numpy()
            rows.append(pd.DataFrame({
                "row_id": sample["row_id"], "model": "hist_factor", "y_pred": prediction,
                "seed": int(seed),
            }))
    return pd.concat(rows, ignore_index=True)

