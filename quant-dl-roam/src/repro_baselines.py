"""复现论文 Table 2 里可以忠实重建的基线: LSTM 与 MLP-Linear。

论文 Table 2 的 9 行里:
  * StockMixer        —— 官方代码, 已复现
  * LSTM              —— "applies vanilla LSTM on temporal price data for ranking"
  * MLP Linear        —— "uses simple fully connected layers to predict the final price"
  * ALSTM / RGCN / GAT / RSR-I / STHAN-SR / ESTIMATE
      —— 各自论文的代码不在本仓库(关系数据 relation.tar.gz 有, 但模型代码没有),
         且原实现基于 TensorFlow 1.x; 忠实重建需要逐篇重实现, 本轮不做,
         在报告里如实标注为"未复现"而不是给一个不可信的数。

本脚本用**与官方完全相同的协议**跑 LSTM / Linear: 每天一个 batch、损失作用在
预测价格上、按验证损失选轮次、用作者的 evaluator 打分。

    ./.venv/bin/python src/repro_baselines.py --models lstm,linear --seeds 0,1,2
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "vendor/StockMixer/src")
from model import get_loss as official_loss       # noqa: E402
from evaluator import evaluate as official_eval   # noqa: E402

OUT = "out"
VENDOR = "vendor/StockMixer"


class LSTMBaseline(nn.Module):
    """vanilla LSTM over the temporal window -> one number per stock."""

    def __init__(self, channels, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(channels, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                      # x: (N, T, C) -> (N,1)
        h, _ = self.lstm(x)
        return self.head(h[:, -1])


class LinearBaseline(nn.Module):
    """flatten the window -> fully connected -> one number per stock."""

    def __init__(self, T, C, hidden=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(T * C, hidden), nn.ReLU(),
                                nn.Linear(hidden, 1))

    def forward(self, x):                      # x: (N, T, C) -> (N,1)
        return self.fc(x.flatten(1))


def load_market(market):
    dp = os.path.join(VENDOR, "dataset", market)
    eod = pickle.load(open(f"{dp}/eod_data.pkl", "rb"))
    mask = pickle.load(open(f"{dp}/mask_data.pkl", "rb"))
    gt = pickle.load(open(f"{dp}/gt_data.pkl", "rb"))
    price = pickle.load(open(f"{dp}/price_data.pkl", "rb"))
    return eod, mask, gt, price


def run_one(model_name, seed, market, eod, mask, gt, price, epochs,
            valid_index, test_index, lookback=16, alpha=0.1, lr=1e-3):
    np.random.seed(123456789 + 1000 * seed)
    torch.manual_seed(12345678 + seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    stock_num, trade_dates = eod.shape[0], mask.shape[1]

    if model_name == "lstm":
        model = LSTMBaseline(eod.shape[2]).to(device)
    else:
        model = LinearBaseline(lookback, eod.shape[2]).to(device)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    offsets = np.arange(0, valid_index, dtype=int)

    def get_batch(offset):
        seq = lookback
        mb = np.min(mask[:, offset:offset + seq + 1], axis=1)
        return (eod[:, offset:offset + seq, :],
                np.expand_dims(mb, axis=1),
                np.expand_dims(price[:, offset + seq - 1], axis=1),
                np.expand_dims(gt[:, offset + seq], axis=1))

    def run_range(start, end):
        P = np.zeros([stock_num, end - start], dtype=float)
        G = np.zeros([stock_num, end - start], dtype=float)
        M = np.zeros([stock_num, end - start], dtype=float)
        base = start - lookback
        with torch.no_grad():
            for off in range(base, end - lookback):
                db, mb, pb, gb = map(lambda x: torch.Tensor(x).to(device), get_batch(off))
                pred = model(db)
                _, _, _, rr = official_loss(pred, gb, pb, mb, stock_num, alpha)
                k = off - base
                P[:, k] = rr[:, 0].cpu()
                G[:, k] = gb[:, 0].cpu()
                M[:, k] = mb[:, 0].cpu()
        return P, G, M

    best_val, best_test, best_ep = np.inf, None, -1
    for ep in range(epochs):
        np.random.shuffle(offsets)
        for j in range(valid_index - lookback):
            db, mb, pb, gb = map(lambda x: torch.Tensor(x).to(device),
                                 get_batch(offsets[j]))
            opt.zero_grad()
            loss, _, _, _ = official_loss(model(db), gb, pb, mb, stock_num, alpha)
            loss.backward()
            opt.step()
        Pv, Gv, Mv = run_range(valid_index, test_index)
        vloss = float(np.linalg.norm((Pv - Gv) * Mv) ** 2 / np.sum(Mv))
        if vloss < best_val:
            best_val, best_ep = vloss, ep + 1
            best_test = run_range(test_index, trade_dates)
    Pt, Gt, Mt = best_test
    perf = official_eval(Pt, Gt, Mt)
    return dict(model=model_name, market=market, seed=seed, epoch=best_ep,
                params=npar, IC=perf["IC"], ICIR=perf["RIC"],
                prec_10=perf["prec_10"], SR=perf["sharpe5"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lstm,linear")
    ap.add_argument("--market", default="NASDAQ")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--valid-index", type=int, default=756)
    ap.add_argument("--test-index", type=int, default=1008)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    eod, mask, gt, price = load_market(args.market)
    print(f"{args.market}: {eod.shape[0]} 只 × {eod.shape[1]} 日")
    rows = []
    for mn in args.models.split(","):
        for sd in [int(x) for x in args.seeds.split(",")]:
            r = run_one(mn, sd, args.market, eod, mask, gt, price, args.epochs,
                        args.valid_index, args.test_index)
            rows.append(r)
            print(f"  [{mn} seed{sd}] 选模轮次 {r['epoch']}: IC={r['IC']:.4f} "
                  f"ICIR={r['ICIR']:.4f} prec@10={r['prec_10']:.4f} "
                  f"SR={r['SR']:.4f}", flush=True)
    out = args.out or f"{OUT}/baselines_{args.market.lower()}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
