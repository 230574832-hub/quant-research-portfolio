"""Emit the markdown tables used in the report, straight from out/*.csv.

Keeping the tables generated (rather than hand-typed) means the report can be
regenerated after any re-run and the numbers cannot drift from the artefacts.

    ./.venv/bin/python src/report_tables.py            # all tables
    ./.venv/bin/python src/report_tables.py summary    # one section
"""
import sys

import numpy as np
import pandas as pd

OUT = "out"
ORDER = ["ridge", "mlp", "gru", "mgru", "mgru_feat", "hist", "sm", "sm_nomix", "sm_pool"]
NICE = {"ridge": "Ridge", "mlp": "MLP", "gru": "GRU", "mgru": "GRU+市场门控",
        "mgru_feat": "GRU+MASTER 纯特征门控", "hist": "HIST（概念图）", "sm": "**StockMixer（完整）**",
        "sm_nomix": "StockMixer −横截面混合", "sm_pool": "StockMixer −市场池化"}
LABEL = {"y1": "次日收益（t→t+1）", "y5": "5 日收益（t→t+5）"}


def _md(df, floatfmt="{:.4f}"):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def t_summary():
    s = pd.read_csv(f"{OUT}/metrics_summary.csv")
    s = s[s.label == "y1"].set_index("model")
    rows = []
    for m in ORDER:
        if m not in s.index:
            continue
        r = s.loc[m]
        rows.append(dict(
            模型=NICE.get(m, m),
            参数量=PARAMS.get(m, ""),
            IC=r["ic_mean"], ICIR=r["icir"],
            RankIC=r["rankic_mean"], **{"RankIC(种子标准差)": r["rankic_mean_seedstd"]},
            RankICIR=r["rankicir"], t值=r["rankic_t"],
            日胜率=r["hit_rate"], **{"prec@10": r["prec@10"], "prec@30": r["prec@30"]},
            测试天数=int(r["n_days"]),
        ))
    print("### 表 1　主结果（y1 标签，3 种子合并，2016-01 ~ 2026-03）\n")
    print(_md(pd.DataFrame(rows)))
    print()


def t_folds():
    b = pd.read_csv(f"{OUT}/metrics_by_model_fold.csv")
    b = b[b.label == "y1"]
    p = b.groupby(["model", "fold"])["rankic_mean"].mean().unstack()
    p = p.reindex([m for m in ORDER if m in p.index])
    p = p.rename(columns={0: "2016-18", 1: "2018-20", 2: "2020-22", 3: "2022-24", 4: "2024-26"})
    p.index = [NICE.get(m, m) for m in p.index]
    print("### 表 2　逐折测试 RankIC（种子均值）\n")
    print(_md(p.reset_index().rename(columns={"index": "模型"})))
    print()


def t_year():
    y = pd.read_csv(f"{OUT}/metrics_by_year.csv")
    p = y.groupby(["model", "year"])["rankic_mean"].mean().unstack()
    p = p.reindex([m for m in ORDER if m in p.index])
    p.index = [NICE.get(m, m) for m in p.index]
    print("### 表 3　逐年测试 RankIC（折与种子均值）\n")
    print(_md(p.reset_index().rename(columns={"index": "模型"})))
    print()


def t_decomp():
    d = pd.read_csv(f"{OUT}/decomp_overnight_intraday.csv")
    g = d.groupby("model")[["rankic_overnight", "rankic_intraday",
                            "rankic_close2close"]].mean()
    g = g.reindex([m for m in ORDER if m in g.index])
    g.index = [NICE.get(m, m) for m in g.index]
    g.columns = ["隔夜 RankIC", "日内 RankIC", "收盘-收盘 RankIC"]
    print("### 表 4　信号落在哪一段收益上\n")
    print(_md(g.reset_index().rename(columns={"index": "模型"})))
    print()


def t_port():
    p = pd.read_csv(f"{OUT}/portfolio_summary.csv")
    num = [c for c in p.select_dtypes("number").columns]
    for hz, lab in [("c2c", "表 5a　收盘→收盘，日频调仓"),
                    ("c2c5", "表 5b　收盘→收盘，每 5 日调仓（不重叠）")]:
        q = p[p.horizon == hz].groupby("model")[num].mean()
        q = q.reindex([m for m in ORDER if m in q.index])
        if q.empty:
            continue
        out = pd.DataFrame({
            "模型": [NICE.get(m, m) for m in q.index],
            "多空净SR": q["ls_net_sr"].values,
            "随机零假设SR": q["null_ls_net_sr"].values,
            "超额": q["ls_net_sr"].values - q["null_ls_net_sr"].values,
            "多头净SR": q["long_net_sr"].values,
            "随机多头SR": q["null_long_net_sr"].values,
            "多空累计": q["ls_net_cum"].values,
            "多头累计": q["long_net_cum"].values,
        })
        print(f"### {lab}\n")
        print(_md(out, "{:.3f}"))
        print()


def t_lr():
    """Learning-rate fairness check for StockMixer (separate tags)."""
    import glob
    rows = []
    frames = []
    for f in ["out/metrics_by_model_fold.csv", "out/lr_metrics_by_model_fold.csv"]:
        try:
            b = pd.read_csv(f)
        except FileNotFoundError:
            continue
        b = b[b.label == "y1"]
        frames.append(b[["model", "fold", "tag", "rankic_mean"]])
    if not frames:
        return
    b = pd.concat(frames, ignore_index=True)
    setup = [("_s0", "1e-3（默认，论文值）", "sm"),
             ("_lr3e4", "3e-4", "sm"),
             ("_lr3e5", "3e-5", "sm"),
             ("_s0", "1e-3（默认，论文值）", "sm_nomix"),
             ("_lr3e4", "3e-4", "sm_nomix")]
    for tag, lab, m in setup:
        sub = b[(b.model == m) & (b.tag == tag)]
        if sub.empty:
            continue
        # best epoch / final train loss from the epoch history
        ps = sorted(glob.glob(f"{OUT}/hist_{m}{tag}_f0.csv"))
        be, trl = "—", "—"
        if ps:
            h = pd.read_csv(ps[0])
            bb = h.loc[h.val_ic.idxmax()]
            be = int(bb["epoch"])
            trl = f"{h['train_loss'].iloc[-1]:.4f}"
        rows.append(dict(模型=NICE.get(m, m), 学习率=lab, 折数=len(sub),
                         最佳轮=be, 末轮训练损失=trl,
                         测试RankIC均值=round(sub["rankic_mean"].mean(), 4),
                         最差折=round(sub["rankic_mean"].min(), 4),
                         最好折=round(sub["rankic_mean"].max(), 4)))
    if not rows:
        return
    print("### 表 6　StockMixer 学习率敏感性（seed 0，5 折）\n")
    print(_md(pd.DataFrame(rows)))
    print()


def t_overfit():
    """Best epoch and final train loss: the overfitting signature."""
    import glob
    import os
    rows = []
    for m in ["gru", "sm_nomix", "sm"]:
        for fi in [0, 4]:
            ps = sorted(glob.glob(f"{OUT}/hist_{m}_s0_f{fi}.csv"))
            if not ps:
                continue
            h = pd.read_csv(ps[0])
            b = h.loc[h.val_ic.idxmax()]
            rows.append(dict(模型=NICE.get(m, m), 折=fi, 实际轮数=len(h),
                             最佳轮=int(b["epoch"]),
                             最佳验证RankIC=b["val_ic"],
                             末轮训练损失=h["train_loss"].iloc[-1],
                             末轮验证RankIC=h["val_ic"].iloc[-1]))
    if not rows:
        return
    print("### 表 7　训练动态：最佳轮次与末轮训练损失\n")
    print(_md(pd.DataFrame(rows)))
    print()


PARAMS = {"ridge": 129, "mlp": 24_833, "gru": 14_273, "mgru": 23_033,
          "mgru_feat": 14_329, "hist": 80_833, "sm": 21_999, "sm_nomix": 1_479,
          "sm_pool": 3_139}

TABLES = {"summary": t_summary, "folds": t_folds, "year": t_year,
          "decomp": t_decomp, "port": t_port, "lr": t_lr, "overfit": t_overfit}

if __name__ == "__main__":
    keys = sys.argv[1:] or list(TABLES)
    for k in keys:
        if k in TABLES:
            TABLES[k]()
