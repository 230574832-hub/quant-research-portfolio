"""Figures for the factor health-check report."""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.ticker import MaxNLocator

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

IC = pd.read_csv("out/ic_by_year.csv", index_col=0)
ICD = pd.read_csv("out/ic_daily.csv", index_col=0, parse_dates=True)
summ = pd.read_csv("out/ic_summary.csv")
dec = pd.read_csv("out/decay_fixed_universe.csv", index_col=0)
curves = pd.read_csv("out/curves_2018.csv", index_col=0, parse_dates=True)
wk = pd.read_csv("out/weekly_longonly.csv", index_col=0)
ml = pd.read_csv("out/ml_walkforward.csv")
mlp = pd.read_csv("out/ml_portfolio.csv", index_col=0)

PAL = ["#1F3864", "#2E75B6", "#8FAADC", "#C00000", "#E69138", "#548235", "#7030A0"]

# ---------------- fig 1: yearly IC heatmap --------------------------------
order = ["IDIO_20", "VOL_20", "MAX_20", "GAPVOL_20", "TURN_20", "VRATIO", "REV_5",
         "REV_1", "SIZE", "MOM_20", "MOM_60", "AMT_20"]
hm = IC[order].T / 100
fig, ax = plt.subplots(figsize=(9.2, 4.4))
vmax = np.nanpercentile(np.abs(hm.values), 95)
im = ax.imshow(hm.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=8)
ax.set_xticks(range(len(hm.columns)))
ax.set_xticklabels(hm.columns, rotation=45, fontsize=8)
ax.set_xlabel("年份", fontsize=9)
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label("RankIC (%)", fontsize=8)
ax.set_title("图1  A股量价因子的年度 RankIC（全市场，2011–2026）", fontsize=10, loc="left")
ax.grid(False)
fig.savefig("figs/fig1_ic_heatmap.png")
plt.close(fig)

# ---------------- fig 2: IC by label ---------------------------------------
sub = summ[summ.version == "raw"].pivot(index="factor", columns="label", values="IC")
sub = sub.loc[[f for f in order if f in sub.index]] * 100
fig, ax = plt.subplots(figsize=(8.6, 4.6))
x = np.arange(len(sub))
w = 0.27
for i, (lab, color, name) in enumerate([("y_all", PAL[0], "次日全收益"),
                                       ("y_on", PAL[3], "隔夜（close→open）"),
                                       ("y_id", PAL[1], "日内（open→close）")]):
    ax.bar(x + (i - 1) * w, sub[lab].values, w, label=name, color=color,
           edgecolor="white", linewidth=0.4)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(sub.index, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("平均 RankIC (%)", fontsize=9)
ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
ax.set_title("图2  同一因子对三种收益标签的预测力对比（数值越高越强）", fontsize=10, loc="left")
fig.savefig("figs/fig2_label_ic.png")
plt.close(fig)

# ---------------- fig 3: decay, all vs big cap -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
sel = ["REV_5", "VOL_20", "IDIO_20", "MOM_20"]
for ax, tag, title in [(axes[0], "all", "全市场"), (axes[1], "big", "前1000大市值")]:
    for i, f in enumerate(sel):
        col = f"{f}|{tag}"
        if col in dec.columns:
            ax.plot(dec.index, dec[col].values / 100, marker="o", ms=3,
                    color=PAL[i], label=f)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{title}", fontsize=9)
    ax.set_xlabel("年份", fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
axes[0].set_ylabel("年度 RankIC (%)", fontsize=9)
axes[0].legend(fontsize=7.5, frameon=False)
fig.suptitle("图3  因子有效性的时间趋势：只有 5 日反转在衰减", fontsize=10, x=0.02, ha="left")
fig.savefig("figs/fig3_decay.png")
plt.close(fig)

# ---------------- fig 4: cumulative curves ---------------------------------
fig, ax = plt.subplots(figsize=(9.0, 4.2))
mapping = [("HGBR", "梯度提升树", 2.0, "-"), ("SIZE", "小市值因子", 1.6, "--"),
           ("PRICE", "低价因子", 1.3, "-."), ("IDIO_20", "低特质波动因子", 1.2, ":"),
           ("EW_MARKET", "等权基准", 1.2, "-")]
for i, (name, lab, lw, ls) in enumerate(mapping):
    if name in curves.columns:
        ser = curves[name].dropna()
        ax.plot(ser.index, ser.values, lw=lw, ls=ls, label=lab, color=PAL[i])
ax.legend(fontsize=8, frameon=False)
ax.set_ylabel("累计净值（周频，扣 15bp 单边）", fontsize=9)
ax.set_title("图4  T+1 可执行的周频多头组合（2018–2026，Top-50 等权）", fontsize=10, loc="left")
fig.savefig("figs/fig4_curves.png")
plt.close(fig)

# ---------------- fig 5: ML vs single factor IC by test year ----------------
fig, ax = plt.subplots(figsize=(8.4, 3.6))
yrs = ml.test_year.values
for i, (k, lab) in enumerate([("IC_hgb", "梯度提升树"), ("IC_ridge", "Ridge"),
                              ("IC_IDIO_20", "低特质波动因子"),
                              ("IC_VOLCOMP", "波动率因子等权组合")]):
    ax.plot(yrs, ml[k].values * 100, marker="o", ms=4, lw=1.6,
            color=PAL[i], label=lab)
ax.set_xticks(yrs)
ax.set_ylabel("OOS RankIC (%)", fontsize=9)
ax.legend(fontsize=8, frameon=False, ncol=2)
ax.set_title("图5  滚动样本外 RankIC：机器学习 vs 单因子（预测未来一周收益）",
             fontsize=10, loc="left")
fig.savefig("figs/fig5_ml.png")
plt.close(fig)
print("figures done")
