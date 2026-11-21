"""Figures for the deep cross-sectional benchmark report.

Consumes the CSV products of src/evaluate_wf.py and the per-epoch histories
written by src/train_wf.py.  Output: figs/figN_*.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams

for cand in ("Songti SC", "PingFang SC", "Heiti SC", "Arial Unicode MS",
             "Noto Sans CJK SC", "SimHei"):
    if any(cand == f.name for f in font_manager.fontManager.ttflist):
        rcParams["font.family"] = cand
        print("font:", cand)
        break
rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

OUT, FIGS = "out", "figs"
os.makedirs(FIGS, exist_ok=True)
PAL = ["#1F3864", "#2E75B6", "#8FAADC", "#C00000", "#E69138", "#548235", "#7030A0"]
NICE = {"ridge": "Ridge", "mlp": "MLP", "gru": "GRU", "mgru": "GRU+市场门控",
        "mgru_feat": "GRU+MASTER纯门控", "hist": "HIST(概念图)", "sm": "StockMixer",
        "sm_nomix": "StockMixer-无混合", "sm_pool": "StockMixer-池化"}
ORDER = ["ridge", "mlp", "gru", "mgru", "mgru_feat", "hist", "sm", "sm_nomix", "sm_pool"]
COLOR = {m: PAL[i % len(PAL)] for i, m in enumerate(ORDER)}
FOLDNAME = {0: "2016-18", 1: "2018-20", 2: "2020-22", 3: "2022-24", 4: "2024-26"}

summ = pd.read_csv(f"{OUT}/metrics_summary.csv")
bymf = pd.read_csv(f"{OUT}/metrics_by_model_fold.csv")
byyr = pd.read_csv(f"{OUT}/metrics_by_year.csv")
dec = pd.read_csv(f"{OUT}/decomp_overnight_intraday.csv")
port = pd.read_csv(f"{OUT}/portfolio_daily.csv")
portfolio_summary = pd.read_csv(f"{OUT}/portfolio_summary.csv")

models = [m for m in ORDER if m in set(summ.model)]
TAGS = sorted(port.tag.unique()) if "tag" in port.columns else [""]


# ---------------- fig 1: RankIC by model x fold ---------------------------
def fig1():
    d = bymf[bymf.label == "y1"].groupby(["model", "fold"])["rankic_mean"].mean().unstack()
    d = d.reindex([m for m in models if m in d.index])
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    x = np.arange(len(d.index))
    w = 0.8 / max(d.shape[1], 1)
    for j, fo in enumerate(d.columns):
        vals = d[fo].to_numpy()
        b = ax.bar(x + j * w - 0.4 + w / 2, vals, w * 0.9,
                   label=f"折 {fo} ({FOLDNAME.get(int(fo), fo)})",
                   color=plt.cm.Blues(0.35 + 0.55 * j / max(d.shape[1] - 1, 1)),
                   edgecolor="none")
        for rect, v in zip(b, vals):
            if np.isfinite(v):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + (0.004 if v >= 0 else -0.014),
                        f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE.get(m, m) for m in d.index], fontsize=9)
    ax.set_ylabel("测试集日均 RankIC (Spearman)")
    ax.set_title("图1  各模型 walk-forward 测试 RankIC(按折,3 个种子均值)")
    ax.legend(fontsize=7.5, ncol=5, frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig1_rankic_by_fold.png")
    plt.close(fig)


# ---------------- fig 2: RankIC by year ------------------------------------
def fig2():
    d = byyr.groupby(["model", "year"])["rankic_mean"].mean().unstack(0)
    d = d[[m for m in models if m in d.columns]]
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for m in d.columns:
        ax.plot(d.index, d[m].to_numpy(), marker="o", ms=4, lw=1.6,
                color=COLOR[m], label=NICE.get(m, m))
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xlabel("年份")
    ax.set_ylabel("年化日均 RankIC")
    ax.set_title("图2  各模型测试 RankIC 的逐年稳定性")
    ax.legend(fontsize=8, ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig2_rankic_by_year.png")
    plt.close(fig)


# ---------------- fig 3: overfitting evidence ------------------------------
def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), sharex=True)
    for ax, m in zip(axes, ["gru", "sm_nomix", "sm"]):
        ps = sorted([p for p in os.listdir(OUT)
                     if p.startswith(f"hist_{m}_s0_f0.csv")])
        if not ps:
            continue
        h = pd.read_csv(f"{OUT}/{ps[0]}")
        ax.plot(h["epoch"], h["train_loss"], marker="o", ms=3, lw=1.4,
                color="#C00000", label="训练损失")
        ax2 = ax.twinx()
        ax2.plot(h["epoch"], h["val_ic"], marker="s", ms=3, lw=1.4,
                 color="#1F3864", label="验证 RankIC")
        ax2.grid(False)
        ax.set_title(NICE.get(m, m), fontsize=10)
        ax.set_xlabel("epoch")
        ax2.set_ylabel("验证 RankIC", fontsize=9)
    axes[0].set_ylabel("训练损失")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("图3  折0 训练动态:横截面混合块压低训练损失,但验证 RankIC 提前见顶",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig3_overfit.png")
    plt.close(fig)


# ---------------- fig 4: overnight / intraday decomposition ----------------
def fig4():
    d = dec.groupby("model")[["rankic_overnight", "rankic_intraday",
                              "rankic_close2close"]].mean()
    d = d.reindex([m for m in models if m in d.index])
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    x = np.arange(len(d.index))
    w = 0.26
    for j, (c, lab, col) in enumerate([
            ("rankic_overnight", "隔夜 (次日开盘 ÷ 当日收盘)", "#1F3864"),
            ("rankic_intraday", "日内 (次日收盘 ÷ 次日开盘)", "#E69138"),
            ("rankic_close2close", "收盘-收盘 (合计)", "#548235")]):
        b = ax.bar(x + (j - 1) * w, d[c].to_numpy(), w * 0.9, label=lab,
                   color=col, edgecolor="none")
        for rect, v in zip(b, d[c].to_numpy()):
            if np.isfinite(v):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + (0.004 if v >= 0 else -0.016),
                        f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE.get(m, m) for m in d.index], fontsize=9)
    ax.set_ylabel("RankIC")
    ax.set_title("图4  模型信号落在哪一段收益上(隔夜 vs 日内)")
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig4_overnight_intraday.png")
    plt.close(fig)


# ---------------- fig 5/6/7: executable equity curves ---------------------
def _curve(model, horizon, col):
    """Seed-averaged daily P&L, folds chained in time order."""
    g = port[(port.model == model) & (port.horizon == horizon)]
    if g.empty:
        return None, None
    per = g.groupby(["fold", "date_idx"])[col].mean().reset_index()
    per = per.sort_values(["fold", "date_idx"])
    v = np.cumprod(1.0 + per[col].to_numpy()) - 1.0
    return np.arange(len(v)), v


def fig5():
    specs = [("c2c", "ls_net", "图5  收盘→收盘多空组合累计净收益(日频调仓,与模型目标一致)",
              "fig5_equity_ls_c2c"),
             ("c2c", "long_net", "图6  收盘→收盘多头 Top-50 累计净收益", "fig6_equity_long_c2c"),
             ("c2c5", "ls_net", "图7  收盘→收盘 5 日多空组合累计净收益(每 5 日调仓)",
              "fig7_equity_ls_c2c5")]
    for horizon, col, ttl, tag in specs:
        fig, ax = plt.subplots(figsize=(10.2, 4.9))
        drew = False
        for m in models:
            xs, ys = _curve(m, horizon, col)
            if ys is None:
                continue
            drew = True
            ax.plot(xs, ys, lw=1.5, color=COLOR[m], label=NICE.get(m, m))
        if not drew:
            plt.close(fig)
            continue
        ax.axhline(0, color="#333333", lw=0.9)
        ax.set_xlabel("测试日(5 折顺次拼接)")
        ax.set_ylabel("累计净收益")
        ax.set_title(ttl)
        ax.legend(fontsize=8, ncol=4, frameon=False)
        fig.tight_layout()
        fig.savefig(f"{FIGS}/{tag}.png")
        plt.close(fig)


# ---------------- fig 9: vs the random-score null ---------------------------
def fig9():
    num = [c for c in portfolio_summary.select_dtypes("number").columns]
    for hz, ttl, tag in [
            ("c2c", "图9  收盘→收盘多空组合净 Sharpe:模型 vs 随机打分零假设(日频)",
             "fig9_vs_null_c2c"),
            ("c2c5", "图10  收盘→收盘 5 日多空组合净 Sharpe:模型 vs 随机打分零假设",
             "fig10_vs_null_c2c5")]:
        q = portfolio_summary[portfolio_summary.horizon == hz].groupby("model")[num].mean()
        q = q.reindex([m for m in models if m in q.index])
        if q.empty:
            continue
        nullv = float(q["null_ls_net_sr"].iloc[0])
        fig, ax = plt.subplots(figsize=(9.8, 4.6))
        x = np.arange(len(q.index))
        b = ax.bar(x, q["ls_net_sr"].to_numpy(), 0.5, color=[COLOR[m] for m in q.index],
                   edgecolor="none")
        for rect, v in zip(b, q["ls_net_sr"].to_numpy()):
            if np.isfinite(v):
                ax.text(rect.get_x() + rect.get_width() / 2, v + (0.25 if v >= 0 else -0.45),
                        f"{v:.2f}", ha="center", fontsize=8)
        ax.axhline(nullv, color="#C00000", ls="--", lw=1.4,
                   label=f"随机打分零假设 (同池 50/50 多空): {nullv:.2f}")
        ax.axhline(0, color="#333333", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([NICE.get(m, m) for m in q.index], fontsize=9)
        ax.set_ylabel("多空组合年化净 Sharpe (15bp 单边)")
        ax.set_title(ttl)
        ax.legend(fontsize=8.5, frameon=False)
        fig.tight_layout()
        fig.savefig(f"{FIGS}/{tag}.png")
        plt.close(fig)


# ---------------- fig 8: two ablations -------------------------------------
def fig8():
    d = summ[(summ.label == "y1")].set_index("model")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4))

    def bars(ax, models, title, ylabel):
        models = [m for m in models if m in d.index]
        if not models:
            return
        vals = d.loc[models, "rankic_mean"].to_numpy()
        errs = d.loc[models, "rankic_mean_seedstd"].to_numpy()
        b = ax.bar(range(len(models)), vals, 0.55, yerr=errs, capsize=4,
                   color=[COLOR[m] for m in models], edgecolor="none")
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + (0.006 if v >= 0 else -0.018),
                    f"{v:.4f}", ha="center", fontsize=8)
        ax.axhline(0, color="#333333", lw=0.9)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([NICE.get(m, m) for m in models], fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        lo, hi = vals.min(), vals.max()
        ax.set_ylim(min(0, lo * 1.4), hi * 1.35)

    bars(ax1, ["sm", "sm_pool", "sm_nomix"], "(a) 横截面混合块消融", "测试 RankIC")
    bars(ax2, ["gru", "mgru_feat", "mgru"], "(b) 市场门控消融", "测试 RankIC")
    fig.suptitle("图8  两组消融:横截面混合块是负贡献;纯 MASTER 特征门控无效,增益来自隐藏态门控",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig8_ablation.png")
    plt.close(fig)




# ---------------- fig 11: three ways of adding structure ----------------
def fig11():
    """把'给模型加结构'的三条路放在一起: 跨股票混合 / 概念图 / 市场门控。"""
    d = summ[(summ.label == "y1")].set_index("model")
    groups = [("StockMixer\n完整", "sm", None),
              ("StockMixer\n去横截面混合", "sm_nomix", None),
              ("GRU\n无门控", "gru", None),
              ("GRU\nMASTER特征门控", "mgru_feat", None),
              ("GRU\n隐藏态门控", "mgru", None),
              ("HIST\n概念图", "hist", None)]
    labs = [g[0] for g in groups]
    keys = [g[1] for g in groups]
    vals = [d.loc[k, "rankic_mean"] if k in d.index else np.nan for k in keys]
    errs = [d.loc[k, "rankic_mean_seedstd"] if k in d.index else np.nan for k in keys]
    cols = [COLOR.get(k, "#888888") for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6),
                                   gridspec_kw={"width_ratios": [1.45, 1]})
    b = ax1.bar(range(len(labs)), vals, 0.62, yerr=errs, capsize=4,
                color=cols, edgecolor="none")
    for rect, v in zip(b, vals):
        if np.isfinite(v):
            ax1.text(rect.get_x() + rect.get_width() / 2, v + 0.004,
                     f"{v:.4f}", ha="center", fontsize=8.5)
    ax1.axhline(0, color="#333333", lw=0.9)
    ax1.set_xticks(range(len(labs)))
    ax1.set_xticklabels(labs, fontsize=8)
    ax1.set_ylabel("测试 RankIC(3 种子均值)")
    ax1.set_title("(a) 三个基准 vs 各自的加结构版本", fontsize=10)
    ax1.set_ylim(min(0, np.nanmin(vals) * 1.3), np.nanmax(vals) * 1.3)

    # 增量条: 相对各自基线的增量
    deltas = [("横截面混合块\n(StockMixer)", d.loc["sm", "rankic_mean"] - d.loc["sm_nomix", "rankic_mean"]),
              ("MASTER\n特征门控", d.loc["mgru_feat", "rankic_mean"] - d.loc["gru", "rankic_mean"]),
              ("HIST\n概念图模块", d.loc["hist", "rankic_mean"] - d.loc["gru", "rankic_mean"]),
              ("隐藏态门控\n(本项目)", d.loc["mgru", "rankic_mean"] - d.loc["gru", "rankic_mean"])]
    dv = [x[1] for x in deltas]
    dc = ["#C00000" if v < 0 else ("#8FAADC" if v < 0.02 else "#548235") for v in dv]
    b2 = ax2.bar(range(len(dv)), dv, 0.55, color=dc, edgecolor="none")
    for rect, v in zip(b2, dv):
        ax2.text(rect.get_x() + rect.get_width() / 2, v + (0.002 if v >= 0 else -0.006),
                 f"{v:+.4f}", ha="center", fontsize=8.5)
    ax2.axhline(0, color="#333333", lw=0.9)
    ax2.set_xticks(range(len(dv)))
    ax2.set_xticklabels([x[0] for x in deltas], fontsize=8)
    ax2.set_ylabel("相对基线的 RankIC 增量")
    ax2.set_title("(b) 增量只有隐藏态门控为正", fontsize=10)
    fig.suptitle("图11  给模型加结构的三条路: 只有对隐藏表示做条件化才产生增量",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig11_hist.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig8()
    fig9()
    fig11()
    print("figures written to", FIGS)
