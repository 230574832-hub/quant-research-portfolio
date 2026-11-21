"""官方 StockMixer 代码的复现评测器。

做两件事:
  1) 解析作者原始 train.py 的运行日志, 抽出逐轮的 train/valid/test loss 与
     五个指标, 按**验证损失最低**的轮次挑出该轮的测试表现 —— 这正是作者代码
     的选模规则 (`if val_loss < best_valid_loss`)。
  2) 用作者自己的 model.py + 作者自己的数据集, 复跑官方协议并把测试期预测
     落盘, 以便补算**真正的 Spearman 秩相关** —— 因为官方 evaluator 里的
     `RIC = mean(IC)/std(IC)` 实为 ICIR, 与论文正文对 "RIC" 的表述不符
     (见 reports/lit_spec.md §1.4)。

用法:
    ./.venv/bin/python src/repro_paper.py parse            # 解析日志并对拍论文表格
    ./.venv/bin/python src/repro_paper.py rankic            # 补算 Spearman 秩相关
"""
import argparse
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

OUT = "out"
VENDOR = "vendor/StockMixer"
LOGS = "logs"

# 论文 Table 2 (AAAI-24, NASDAQ)
PAPER = {
    "NASDAQ": dict(IC=0.043, RIC=0.501, prec_10=0.545, sharpe5=1.465),
    "NYSE":   dict(IC=0.029, RIC=0.351, prec_10=0.539, sharpe5=1.454),
    "SP500":  dict(IC=0.041, RIC=0.262, prec_10=0.551, sharpe5=1.586),
}
PAPER_SPLIT = dict(NASDAQ="2010/01/04~2017/12/08", NYSE="2010/01/04~2016/12/30",
                   SP500="2014/01/02~2020/12/24")

RE_EPOCH = re.compile(r"^epoch(\d+)#+$")
RE_TRAIN = re.compile(r"^Train : loss:([-\deE.+]+)\s+=\s+([-\deE.+]+)\s*\+\s*alpha\*([-\deE.+]+)")
RE_VALID = re.compile(r"^Valid : loss:([-\deE.+]+)\s+=\s+([-\deE.+]+)\s*\+\s*alpha\*([-\deE.+]+)")
RE_TEST = re.compile(r"^Test: loss:([-\deE.+]+)\s+=\s+([-\deE.+]+)\s*\+\s*alpha\*([-\deE.+]+)")
RE_PERF = re.compile(r"mse:([-\deE.+]+),\s*IC:([-\deE.+]+),\s*RIC:([-\deE.+]+),\s*"
                     r"prec@10:([-\deE.+]+),\s*SR:([-\deE.+]+)")


def parse_log(path):
    """把作者的日志解析成逐轮记录。"""
    rows = []
    cur = {}
    which = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = RE_EPOCH.match(line)
            if m:
                if cur:
                    rows.append(cur)
                cur = {"epoch": int(m.group(1))}
                which = None
                continue
            if not cur:
                continue
            m = RE_TRAIN.match(line)
            if m:
                cur["train_loss"] = float(m.group(1))
                cur["train_reg"] = float(m.group(2))
                cur["train_rank"] = float(m.group(3))
                continue
            m = RE_VALID.match(line)
            if m:
                cur["valid_loss"] = float(m.group(1))
                cur["valid_reg"] = float(m.group(2))
                cur["valid_rank"] = float(m.group(3))
                continue
            m = RE_TEST.match(line)
            if m:
                cur["test_loss"] = float(m.group(1))
                cur["test_reg"] = float(m.group(2))
                cur["test_rank"] = float(m.group(3))
                continue
            if line.startswith("Valid performance"):
                which = "valid"
                continue
            if line.startswith("Test performance"):
                which = "test"
                continue
            m = RE_PERF.search(line)
            if m and which:
                for k, v in zip(["mse", "IC", "RIC", "prec_10", "sharpe5"], m.groups()):
                    cur[f"{which}_{k}"] = float(v)
                which = None
                continue
    if cur:
        rows.append(cur)
    return pd.DataFrame(rows)


def best_epoch(df):
    """作者的选模规则: 验证损失最低的那一轮。"""
    return df.loc[df["valid_loss"].idxmin()]


def cmd_parse(args):
    frames = []
    for tag, path in args.logs:
        if not os.path.exists(path):
            print(f"  [warn] 缺日志 {path}")
            continue
        d = parse_log(path)
        market, _, seed = tag.partition("/")
        d["market"] = market or "NASDAQ"
        d["seed"] = seed or tag
        d["tag"] = tag
        frames.append(d)
        print(f"  {path}: {len(d)} 轮 ({d['market'].iloc[0]}/{d['seed'].iloc[0]})")
    if not frames:
        raise SystemExit("没有可用日志")
    alld = pd.concat(frames, ignore_index=True)
    alld.to_csv(f"{OUT}/paper_repro_epochs.csv", index=False)
    nk = alld[alld.market == "NASDAQ"]
    if len(nk):
        nk.to_csv(f"{OUT}/paper_repro_nasdaq_epochs.csv", index=False)

    # 逐种子取验证损失最低轮, 再跨种子平均 (论文是 3 次运行取平均)
    picks = []
    for (mkt, sd), g in alld.groupby(["market", "seed"]):
        b = best_epoch(g)
        picks.append(dict(market=mkt, seed=sd, epoch=int(b["epoch"]),
                          test_mse=b["test_mse"], test_IC=b["test_IC"],
                          test_RIC=b["test_RIC"], test_prec10=b["test_prec_10"],
                          test_SR=b["test_sharpe5"],
                          valid_loss=b["valid_loss"], test_loss=b["test_loss"]))
    pk = pd.DataFrame(picks)
    pk.to_csv(f"{OUT}/paper_repro_best.csv", index=False)
    # 兼容旧文件名
    nk = pk[pk.market == "NASDAQ"]
    if len(nk):
        nk.to_csv(f"{OUT}/paper_repro_nasdaq_best.csv", index=False)

    print()
    print("### 官方代码复现: 选模轮次(验证损失最低)的测试表现\n")
    rows = []
    for r in pk.itertuples(index=False):
        rows.append({"市场": r.market, "种子": r.seed, "轮次": r.epoch,
                     "MSE": r.test_mse, "IC": r.test_IC,
                     "RIC(=ICIR)": r.test_RIC, "prec@10": r.test_prec10,
                     "SR": r.test_SR})
    print(_md(pd.DataFrame(rows), "{:.4f}"))
    print()
    print("### 与论文 Table 2 对拍(逐市场)\n")
    for mkt in pk["market"].unique():
        sub = pk[pk.market == mkt]
        cols = ["test_mse", "test_IC", "test_RIC", "test_prec10", "test_SR"]
        avg, std = sub[cols].mean(), sub[cols].std()
        P = PAPER[mkt]
        print(f"\n**{mkt}** (论文: IC {P['IC']}, \"RIC\" {P['RIC']}, "
              f"prec@N {P['prec_10']}, SR {P['sharpe5']})\n")
        cmp_rows = [
            dict(指标="IC", 论文=P["IC"], 复现=avg["test_IC"],
                 差值=avg["test_IC"] - P["IC"], 种子标准差=std["test_IC"]),
            dict(指标="RIC (代码实为 ICIR)", 论文=P["RIC"], 复现=avg["test_RIC"],
                 差值=avg["test_RIC"] - P["RIC"], 种子标准差=std["test_RIC"]),
            dict(指标="prec@10", 论文=P["prec_10"], 复现=avg["test_prec10"],
                 差值=avg["test_prec10"] - P["prec_10"], 种子标准差=std["test_prec10"]),
            dict(指标="SR (top-5 等权)", 论文=P["sharpe5"], 复现=avg["test_SR"],
                 差值=avg["test_SR"] - P["sharpe5"], 种子标准差=std["test_SR"]),
        ]
        print(_md(pd.DataFrame(cmp_rows), "{:.4f}"))
        print()
    print("论文 Table 2 的三个数据集切分见 reports/lit_spec.md；"
          "本项目的复现一律沿用作者 train.py 的 valid_index/test_index")
    print()


def _md(df, floatfmt="{:.4f}"):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


# --------------------------------------------------------------------------
# 补算真正的 Spearman 秩相关 (作者 evaluator 的 RIC 实为 ICIR)
# --------------------------------------------------------------------------
def cmd_rankic(args):
    """用作者的 model.py + 作者的数据 + 官方协议复跑, 落盘测试期预测。

    唯一改动: 训练循环里多存一份预测; 模型与损失函数直接 import 作者的。
    """
    src = os.path.join(VENDOR, "src")
    sys.path.insert(0, src)
    import torch
    from model import StockMixer, get_loss          # noqa: E402  (作者原文件)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    np.random.seed(123456789)
    torch.manual_seed(12345678)

    stock_num, lookback_length, epochs = 1026, 16, args.epochs
    valid_index, test_index, alpha, lr = 756, 1008, 0.1, 1e-3
    dp = os.path.join(VENDOR, "dataset", "NASDAQ")
    eod = pickle.load(open(f"{dp}/eod_data.pkl", "rb"))
    mask = pickle.load(open(f"{dp}/mask_data.pkl", "rb"))
    gt = pickle.load(open(f"{dp}/gt_data.pkl", "rb"))
    price = pickle.load(open(f"{dp}/price_data.pkl", "rb"))
    trade_dates = mask.shape[1]

    model = StockMixer(stocks=stock_num, time_steps=lookback_length, channels=5,
                       market=20, scale=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    offsets = np.arange(0, valid_index, dtype=int)

    def get_batch(offset):
        seq = lookback_length
        mb = np.min(mask[:, offset:offset + seq + 1], axis=1)
        return (eod[:, offset:offset + seq, :],
                np.expand_dims(mb, axis=1),
                np.expand_dims(price[:, offset + seq - 1], axis=1),
                np.expand_dims(gt[:, offset + seq], axis=1))

    def run_range(start, end, keep_pred=False):
        P = np.zeros([stock_num, end - start], dtype=float)
        G = np.zeros([stock_num, end - start], dtype=float)
        M = np.zeros([stock_num, end - start], dtype=float)
        base = start - lookback_length
        with torch.no_grad():
            for off in range(base, end - lookback_length):
                db, mb, pb, gb = map(lambda x: torch.Tensor(x).to(device), get_batch(off))
                pred = model(db)
                _, _, _, rr = get_loss(pred, gb, pb, mb, stock_num, alpha)
                k = off - base
                P[:, k] = rr[:, 0].cpu()
                G[:, k] = gb[:, 0].cpu()
                M[:, k] = mb[:, 0].cpu()
        return P, G, M

    best_val, best_test = np.inf, None
    for ep in range(epochs):
        np.random.shuffle(offsets)
        for j in range(valid_index - lookback_length):
            db, mb, pb, gb = map(lambda x: torch.Tensor(x).to(device),
                                 get_batch(offsets[j]))
            opt.zero_grad()
            pred = model(db)
            loss, _, _, _ = get_loss(pred, gb, pb, mb, stock_num, alpha)
            loss.backward()
            opt.step()
        Pv, Gv, Mv = run_range(valid_index, test_index)
        vloss = float(np.linalg.norm((Pv - Gv) * Mv) ** 2 / np.sum(Mv))
        if vloss < best_val:
            best_val = vloss
            Pt, Gt, Mt = run_range(test_index, trade_dates, keep_pred=True)
            best_test = (ep, Pt, Gt, Mt)
        print(f"  epoch{ep+1:3d} valid_loss={vloss:.3e}  best={best_val:.3e}", flush=True)

    ep, Pt, Gt, Mt = best_test
    print(f"\n最佳轮次(验证损失最低) = epoch {ep+1}")

    # 逐日 Spearman 秩相关
    ric, ic, hit = [], [], []
    for t in range(Pt.shape[1]):
        m = Mt[:, t] > 0.5
        if m.sum() < 20:
            continue
        p, g = Pt[m, t], Gt[m, t]
        if np.std(p) < 1e-12 or np.std(g) < 1e-12:
            continue
        pr = pd.Series(p).rank().to_numpy()
        gr = pd.Series(g).rank().to_numpy()
        ric.append(np.corrcoef(pr, gr)[0, 1])
        ic.append(np.corrcoef(p, g)[0, 1])
        hit.append(1.0 if ric[-1] > 0 else 0.0)
    ric = np.array(ric); ic = np.array(ic)
    res = dict(epoch=ep + 1,
               rankic_spearman=float(ric.mean()),
               rankic_std=float(ric.std(ddof=1)),
               rankicir=float(ric.mean() / ric.std(ddof=1)),
               ic=float(ic.mean()), icir=float(ic.mean() / ic.std(ddof=1)),
               hit_rate=float(np.mean(hit)), n_days=len(ric))
    print(f"  真正的 RankIC (Spearman, 日均) = {res['rankic_spearman']:.4f}")
    print(f"  RankICIR                          = {res['rankicir']:.4f}")
    print(f"  IC (Pearson, 日均)                 = {res['ic']:.4f}")
    print(f"  ICIR (官方 evaluator 的 'RIC')      = {res['icir']:.4f}")
    pd.DataFrame([res]).to_csv(f"{OUT}/paper_repro_nasdaq_rankic.csv", index=False)
    np.savez_compressed(f"{OUT}/paper_repro_nasdaq_testpred.npz",
                        pred=Pt, gt=Gt, mask=Mt)
    print(f"  已写入 {OUT}/paper_repro_nasdaq_rankic.csv 与测试期预测")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["parse", "rankic"])
    ap.add_argument("--logs", nargs="*", default=[],
                    help="tag=path 形式的日志列表")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    if args.cmd == "parse":
        logs = []
        for spec in (args.logs or ["seed0=logs/paper_repro_nasdaq_s0.log"]):
            tag, _, path = spec.partition("=")
            logs.append((tag, path))
        args.logs = logs
        cmd_parse(args)
    else:
        cmd_rankic(args)
