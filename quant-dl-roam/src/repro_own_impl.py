"""交叉验证:把**本项目自己的** StockMixer 实现,放到**作者的** NASDAQ 数据与协议上跑。

目的:本项目关于 StockMixer 的所有 A 股结论都建立在自己写的 src/models.py 上。
逐行核对官方 model.py 只是"看着像",不是"验证过"。本脚本用同一份官方数据、
同一套官方协议(每天一个 batch、损失作用在预测价格上、按验证损失选轮次、官方
evaluator 打分),跑本项目的实现,看能不能落到作者的数字附近。

若本实现能复现出作者的量级, 则 A 股结论建立在"同一套机制"上, 只是换了市场与特征;
若不能, 则 A 股结论可能只是本实现的 bug。

    ./.venv/bin/python src/repro_own_impl.py --seeds 0,1,2 --epochs 100
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, "vendor/StockMixer/src")

from models import StockMixer                     # 本项目的实现  # noqa: E402
from model import get_loss as official_loss      # 作者的损失函数  # noqa: E402
from evaluator import evaluate as official_eval  # 作者的评估器  # noqa: E402

OUT = "out"
VENDOR = "vendor/StockMixer"
PAPER = dict(IC=0.043, RIC=0.501, prec_10=0.545, sharpe5=1.465)


def load_nasdaq():
    dp = os.path.join(VENDOR, "dataset", "NASDAQ")
    eod = pickle.load(open(f"{dp}/eod_data.pkl", "rb"))
    mask = pickle.load(open(f"{dp}/mask_data.pkl", "rb"))
    gt = pickle.load(open(f"{dp}/gt_data.pkl", "rb"))
    price = pickle.load(open(f"{dp}/price_data.pkl", "rb"))
    return eod, mask, gt, price


def run_seed(seed, eod, mask, gt, price, epochs, valid_index, test_index,
             stock_num, lookback=16, alpha=0.1, lr=1e-3, market=20, lr_tag=""):
    np.random.seed(123456789 + 1000 * seed)
    torch.manual_seed(12345678 + seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = StockMixer(stocks=stock_num, T=lookback, C=eod.shape[2],
                       market=market, mix="graph_nomask").to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    offsets = np.arange(0, valid_index, dtype=int)
    trade_dates = mask.shape[1]

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
                # 本项目实现的 forward 期望 (B, N, T, C) 与 (B, N) 掩码
                pred = model(db.unsqueeze(0), torch.ones(1, db.shape[0], device=device))
                _, _, _, rr = official_loss(pred.transpose(0, 1), gb, pb, mb, stock_num, alpha)
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
            pred = model(db.unsqueeze(0), torch.ones(1, db.shape[0], device=device))
            loss, _, _, _ = official_loss(pred.transpose(0, 1), gb, pb, mb, stock_num, alpha)
            loss.backward()
            opt.step()
        Pv, Gv, Mv = run_range(valid_index, test_index)
        vloss = float(np.linalg.norm((Pv - Gv) * Mv) ** 2 / np.sum(Mv))
        if vloss < best_val:
            best_val, best_ep = vloss, ep + 1
            best_test = run_range(test_index, trade_dates)
        if (ep + 1) % 10 == 0:
            print(f"  [seed{seed}] epoch{ep+1:3d} valid_loss={vloss:.3e} "
                  f"best={best_val:.3e} (ep{best_ep})", flush=True)

    Pt, Gt, Mt = best_test
    perf = official_eval(Pt, Gt, Mt)
    # 补算真正的 Spearman 秩相关
    ric = []
    for t in range(Pt.shape[1]):
        m = Mt[:, t] > 0.5
        if m.sum() < 20:
            continue
        p, g = Pt[m, t], Gt[m, t]
        if np.std(p) < 1e-12 or np.std(g) < 1e-12:
            continue
        ric.append(np.corrcoef(pd.Series(p).rank(), pd.Series(g).rank())[0, 1])
    ric = np.array(ric)
    res = dict(seed=seed, epoch=best_ep, params=n_par,
               IC=perf["IC"], ICIR=perf["RIC"], prec_10=perf["prec_10"],
               SR=perf["sharpe5"], mse=perf["mse"],
               rankic_spearman=float(ric.mean()),
               rankicir=float(ric.mean() / ric.std(ddof=1)) if ric.std() > 0 else np.nan)
    print(f"  [seed{seed}] 选模轮次 {best_ep}: IC={res['IC']:.4f} ICIR={res['ICIR']:.4f} "
          f"prec@10={res['prec_10']:.4f} SR={res['SR']:.4f} "
          f"RankIC={res['rankic_spearman']:.4f} (params={n_par})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", default=f"{OUT}/own_impl_nasdaq.csv")
    args = ap.parse_args()

    eod, mask, gt, price = load_nasdaq()
    stock_num, trade_dates = eod.shape[0], mask.shape[1]
    print(f"NASDAQ 官方数据: {stock_num} 只 × {trade_dates} 日 × {eod.shape[2]} 特征")

    rows = []
    for sd in [int(x) for x in args.seeds.split(",")]:
        rows.append(run_seed(sd, eod, mask, gt, price, args.epochs,
                             valid_index=756, test_index=1008, stock_num=stock_num))
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print()
    print("### 本项目实现 vs 官方代码 vs 论文 (NASDAQ)\n")
    off = pd.read_csv(f"{OUT}/paper_repro_nasdaq_best.csv")
    print("| 指标 | 论文 Table 2 | 官方代码(3 种子) | **本项目实现(3 种子)** |")
    print("|---|---:|---:|---:|")
    for c, lab, pv, oc_col in [
            ("IC", "IC", PAPER["IC"], "test_IC"),
            ("ICIR", "\"RIC\"(=ICIR)", PAPER["RIC"], "test_RIC"),
            ("prec_10", "prec@10", PAPER["prec_10"], "test_prec10"),
            ("SR", "SR", PAPER["sharpe5"], "test_SR")]:
        oc = off[oc_col].mean()
        mine = df[c].mean()
        print(f"| {lab} | {pv:.4f} | {oc:.4f} | **{mine:.4f}** |")
    print()
    print(f"本实现的参数量: {df['params'].iloc[0]}（官方实现见另一次运行）")
    print(f"跨种子标准差: IC {df['IC'].std():.4f}, ICIR {df['ICIR'].std():.4f}")
    print(f"真正 Spearman RankIC: {df['rankic_spearman'].mean():.4f} "
          f"(论文的 \"RIC\" {PAPER['RIC']:.3f} 实为 ICIR)")
    print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
