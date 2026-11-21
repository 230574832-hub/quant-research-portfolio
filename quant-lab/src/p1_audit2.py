"""P1 第二轮审计 (v2: 统一 canonical 网格 + 统一基准).

三个版本对照 (同一因子、同一 canonical 期间网格、同一成本、同一基准):
  V0 = 旧版: 价格阈值涨跌停近似 + 逐笔独立期收益(无持仓延续)
  V1 = 阈值近似 + 真实持仓状态机
  V2 = 真实涨跌停价格 + 真实持仓状态机     <- 主口径

关键口径修复:
  * canonical 网格只由股票池决定(mask), 与组合/因子无关 -> 三版本期间完全可比
  * 基准只在 canonical 网格上算一次 -> 三版本共用, 消除口径漂移
  * 基准口径 = 同窗口等权全池 open-to-open

附加检验:
  A. 稳健性窗口: 全样本 / 剔除2015 / 2017年起 / 2020年起
  B. 因子相关性: Pearson + Spearman(原始值), 含 NW 稳健 t 值
  C. 显著性: Newey-West + 区块自助抽样(block bootstrap)
  D. 归因回归: 超额SP ~ 超额SIZE + 超额PRICE (NW 稳健标准误)
"""
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from p1_engine import Executor, cs_rank, evaluate  # noqa: E402
from p1_portfolio import PortfolioExecutor  # noqa: E402

warnings.filterwarnings("ignore")
t0 = time.time()

CORE = ["SIZE", "PRICE", "TURN_20", "IDIO_20"]
COMBOS = {
    "SIZE": ["SIZE"],
    "PRICE": ["PRICE"],
    "SIZE_PRICE": ["SIZE", "PRICE"],
    "SIZE_PRICE_TURN": ["SIZE", "PRICE", "TURN_20"],
    "SIZE_TURN": ["SIZE", "TURN_20"],
}
HOLD, K, COST = 10, 50, 15.0
PPY = 52.0 * 5 / HOLD

print("loading ...", flush=True)
D = pickle.load(open("data/factors.pkl", "rb"))
P = pickle.load(open("data/panel_wide.pkl", "rb"))
LP = pickle.load(open("data/limit_panel.pkl", "rb"))
dates, codes = D["dates"], D["codes"]
mask = np.asarray(D["mask"])
F = {k: D["factors"][k].values for k in CORE}
del D
print("loaded  %.1fs" % (time.time() - t0), flush=True)

ranks = {f: cs_rank(F[f], mask) for f in CORE}
SCORE = {n: np.nanmean([ranks[f] for f in fs], axis=0).astype(np.float32)
         for n, fs in COMBOS.items()}

# ---------------------------------------------------------------- 统计工具
def nw_mean_t(x, lags=13):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return np.nan, np.nan
    u = x - x.mean()
    s0 = float((u ** 2).sum()) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        s0 += 2.0 * w * float((u[l:] * u[:-l]).sum()) / n
    se = np.sqrt(max(s0, 0.0) / n)
    return (float(x.mean() / se) if se > 0 else np.nan), float(se)


def block_bootstrap_ci(x, block=13, B=5000, seed=11, alpha=0.05):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * block:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]) % n
        means[b] = x[idx.ravel()[:n]].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float((means <= 0).mean())


def nw_ols(y, X, lags=13):
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    if n < k + 10:
        return np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan)
    A = np.linalg.pinv(X.T @ X)
    beta = A @ (X.T @ y)
    e = y - X @ beta
    u = X * e[:, None]
    M = (u.T @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        M += w * (u[l:].T @ u[:-l]) / n
    # Var = (1/n) S^-1 M S^-1,  S = X'X/n, S^-1 = n*A  ->  V = n * A M A
    V = n * (A @ M @ A)
    se = np.sqrt(np.maximum(np.diag(V), 0))
    t = np.where(se > 0, beta / np.where(se > 0, se, 1), np.nan)
    return beta, se, t


# ---------------------------------------------------------------- 执行器
ex0 = Executor(P, mask, dates, cost_bps=COST, max_delay=5, k=K)
ex1 = PortfolioExecutor(P, mask, dates, cost_bps=COST, use_real_limits=False)
ex2 = PortfolioExecutor(P, mask, dates, cost_bps=COST, limit_panel=LP, use_real_limits=True)
grid = ex2.grid(HOLD)

# canonical 网格: 只由股票池决定, 与因子无关
canon = [g for g in grid if mask[g].sum() >= 100]
PAIRS = [(canon[a], canon[a + 1]) for a in range(len(canon) - 1)]
print("\ncanonical 网格: %d 个调仓点 -> %d 个期间 (股票池门槛仅由 mask 决定)" % (
    len(canon), len(PAIRS)), flush=True)

# 统一基准: 只算一次
ao = ex2.ao
BENCH = []
for (i0, i1) in PAIRS:
    with np.errstate(all="ignore"):
        br = ao[i1 + 1] / ao[i0 + 1] - 1.0
    bm = br[mask[i0] & np.isfinite(br)]
    BENCH.append(bm.mean() if len(bm) else np.nan)
BENCH = np.array(BENCH)
eqb = np.cumprod(1 + BENCH)
print("统一基准: 期数 %d  几何CAGR %.4f  最大回撤 %.4f" % (
    len(BENCH), eqb[-1] ** (1 / (len(BENCH) / PPY)) - 1,
    float((eqb / np.maximum.accumulate(eqb) - 1).min())))
# 期间长度分布
LENS = pd.Series([canon[b + 1] - canon[b] for b in range(len(canon) - 1)])
print("期间跨度(交易日): 中位 %.0f  众数 %.0f  最大 %d" % (
    LENS.median(), LENS.mode().iloc[0], LENS.max()))

print("\n=== 涨跌停口径差异诊断 ===", flush=True)
sel = mask & np.isfinite(ex2.open_raw) & (ex2.open_raw > 0)
print("可交易样本 %d" % sel.sum())
print("阈值近似 开盘涨停 %d (%.3f%%)  跌停 %d (%.3f%%)" % (
    (ex0.up_open & sel).sum(), 100 * (ex0.up_open & sel).sum() / sel.sum(),
    (ex0.dn_open & sel).sum(), 100 * (ex0.dn_open & sel).sum() / sel.sum()))
print("真实涨跌停 开盘涨停 %d (%.3f%%)  跌停 %d (%.3f%%)" % (
    (ex2.up_open & sel).sum(), 100 * (ex2.up_open & sel).sum() / sel.sum(),
    (ex2.dn_open & sel).sum(), 100 * (ex2.dn_open & sel).sum() / sel.sum()))
print("阈值漏判真实涨停 %d ; 阈值误判涨停 %d" % (
    (ex2.up_open & ~ex0.up_open & sel).sum(), (ex0.up_open & ~ex2.up_open & sel).sum()))
print("阈值漏判真实跌停 %d ; 阈值误判跌停 %d" % (
    (ex2.dn_open & ~ex0.dn_open & sel).sum(), (ex0.dn_open & ~ex2.dn_open & sel).sum()))
print("无成交(停牌)日 %d (%.2f%%)" % (
    (ex2.no_trade & sel).sum(), 100 * (ex2.no_trade & sel).sum() / sel.sum()))

# ---------------------------------------------------------------- 三版本回测
print("\n=== 三版本回测 (hold=%d, K=%d, 单边成本 %.0fbp, 统一网格/基准) ===" % (HOLD, K, COST),
      flush=True)
results, curves = [], {}
DIAG = {}
for name in COMBOS:
    sc = SCORE[name]
    d0 = ex0.simulate(sc, grid, mode="C", periods=PAIRS)
    d1 = ex1.simulate(sc, grid, periods=PAIRS)
    d2 = ex2.simulate(sc, grid, periods=PAIRS)
    for d in (d0, d1, d2):
        d["bench"] = BENCH[:len(d)]
    DIAG[name] = d2.attrs["diag"]
    for d, ver, desc in [(d0, "V0_旧版", "阈值涨跌停+逐笔独立期, 无持仓延续"),
                         (d1, "V1_阈值+状态机", "阈值涨跌停+持仓状态机"),
                         (d2, "V2_真实涨跌停+状态机", "真实涨跌停+持仓状态机")]:
        m = evaluate(d, label=name)
        m.update({"combo": name, "version": ver, "factors": "+".join(COMBOS[name]),
                  "hold": HOLD, "exec": desc, "periods": len(d)})
        results.append(m)
        curves[f"{name}|{ver[0]}"] = (1 + d["net"]).cumprod()
    print("  %-16s V0 CAGR=%6.2f%% exc=%6.2f%% IR=%.2f | V1 CAGR=%6.2f%% exc=%6.2f%% IR=%.2f "
          "| V2 CAGR=%6.2f%% exc=%6.2f%% IR=%.2f" % (
              name, results[-3]["cagr_net"] * 100, results[-3]["excess_geo"] * 100,
              results[-3]["ir"], results[-2]["cagr_net"] * 100,
              results[-2]["excess_geo"] * 100, results[-2]["ir"],
              results[-1]["cagr_net"] * 100, results[-1]["excess_geo"] * 100,
              results[-1]["ir"]), flush=True)

base = pd.DataFrame(results)
base.to_csv("out/p2_versions.csv", index=False)
pd.DataFrame(curves).to_csv("out/p2_curves.csv")
cols = ["combo", "version", "periods", "years", "ppy_actual", "cagr_net", "cagr_bench",
        "excess_geo", "ir", "sharpe", "maxDD", "calmar", "turnover_annual", "cost_drag",
        "hit_rate", "avg_hold_n"]
print("\n=== 三版本完整指标 ===")
print(base[[c for c in cols if c in base.columns]].round(4).to_string(index=False))

print("\n=== 成交诊断 (V2) ===")
for name in COMBOS:
    dg = DIAG[name]
    print("  %-16s 买入受阻 %5d  卖出受阻 %5d  持仓延续 %5d  未成交目标 %5d  "
          "停牌命中 %5d" % (name, dg["buy_blocked"], dg["sell_blocked"], dg["carried"],
                            dg["unfilled_target"], dg["suspended_hit"]))

print("\n=== 长期持仓检查 (SIZE_PRICE, V2) ===")
d_sp = ex2.simulate(SCORE["SIZE_PRICE"], grid, periods=PAIRS)
hd = d_sp.attrs["held_detail"]
print("  期末仍持有 %d 只; 持有最久的 8 只:" % len(hd))
for h in hd[:8]:
    print("    %s  首次买入 %s  持有 %d 个交易日  期末权重 %.4f" % (
        codes[h["code"]], dates[h["first_buy"]].date(), h["age_days"], h["weight"]))
age = np.array([h["age_days"] for h in hd]) if hd else np.array([0])
print("  持有账龄: 中位 %.0f  90分位 %.0f  最大 %.0f 个交易日" % (
    np.median(age), np.percentile(age, 90), age.max()))

# ---------------------------------------------------------------- A. 稳健性窗口
print("\n=== A. 稳健性窗口 (V2 主口径, 统一基准) ===", flush=True)
WINDOWS = {"全样本": None, "剔除2015": "ex2015",
           "2017年起": ("2017-01-01", "2026-12-31"), "2020年起": ("2020-01-01", "2026-12-31")}
win_rows = []
for name in COMBOS:
    d2 = ex2.simulate(SCORE[name], grid, periods=PAIRS)
    d2["bench"] = BENCH[:len(d2)]
    for wname, rng in WINDOWS.items():
        if rng == "ex2015":
            m = ~((d2.index >= "2015-01-01") & (d2.index <= "2015-12-31"))
        elif rng is None:
            m = np.ones(len(d2), dtype=bool)
        else:
            m = (d2.index >= rng[0]) & (d2.index <= rng[1])
        dd = d2[m]
        if len(dd) < 20:
            continue
        met = evaluate(dd, label=name)
        met.update({"combo": name, "window": wname, "hold": HOLD})
        win_rows.append(met)
        print("  %-16s %-9s 期数=%3d CAGR=%7.2f%% 超额=%7.2f%% IR=%.2f DD=%7.2f%% "
              "基准CAGR=%6.2f%% 命中率=%.1f%%" % (
                  name, wname, met["periods"], met["cagr_net"] * 100,
                  met["excess_geo"] * 100, met["ir"], met["maxDD"] * 100,
                  met["cagr_bench"] * 100, met["hit_rate"] * 100), flush=True)
wdf = pd.DataFrame(win_rows)
wdf.to_csv("out/p2_windows.csv", index=False)

# ---------------------------------------------------------------- B. 相关性
print("\n=== B. 因子相关性 (同一股票池, 同一日期截面, 逐日) ===", flush=True)
N = mask.shape[0]
pairs_corr = [("SIZE", "PRICE"), ("SIZE", "TURN_20"), ("PRICE", "TURN_20"),
              ("SIZE", "IDIO_20"), ("PRICE", "IDIO_20"), ("TURN_20", "IDIO_20")]
corr_rows = []
for a, b in pairs_corr:
    pe, sp = [], []
    for i in range(N):
        m = mask[i] & np.isfinite(F[a][i]) & np.isfinite(F[b][i])
        if m.sum() < 100:
            continue
        x, y = F[a][i][m], F[b][i][m]
        pe.append(np.corrcoef(x, y)[0, 1])
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        sp.append(np.corrcoef(rx, ry)[0, 1])
    pe, sp = np.array(pe), np.array(sp)
    t_pe, _ = nw_mean_t(pe, 13)
    t_sp, _ = nw_mean_t(sp, 13)
    corr_rows.append({"pair": f"{a}~{b}", "n_days": len(pe),
                      "pearson_mean": pe.mean(), "pearson_median": np.median(pe),
                      "pearson_std": pe.std(), "pearson_nw_t": t_pe,
                      "spearman_mean": sp.mean(), "spearman_median": np.median(sp),
                      "spearman_std": sp.std(), "spearman_nw_t": t_sp,
                      "abs_pearson_mean": np.abs(pe).mean(),
                      "frac_days_abs_gt_0.3": float((np.abs(pe) > 0.3).mean())})
cdf = pd.DataFrame(corr_rows)
cdf.to_csv("out/p2_correlations.csv", index=False)
print(cdf.round(4).to_string(index=False))

# ---------------------------------------------------------------- C. 显著性
print("\n=== C. 显著性检验 (期间超额序列) ===", flush=True)
sig_rows = []
for name in COMBOS:
    d2 = ex2.simulate(SCORE[name], grid, periods=PAIRS)
    d2["bench"] = BENCH[:len(d2)]
    ex = (d2["net"] - d2["bench"]).values
    met_ppy = float(len(ex) / max((d2["end_date"].iloc[-1] - d2["start_date"].iloc[0]).days / 365.25, 1e-6))
    t_nw, se = nw_mean_t(ex, 13)
    lo, hi, p = block_bootstrap_ci(ex, block=13, B=5000)
    t_ols, _ = nw_mean_t(ex, 1)
    sig_rows.append({"combo": name, "n_periods": len(ex),
                     "mean_excess_per_period": ex.mean(), "t_iid": t_ols, "t_nw13": t_nw,
                     "se_nw": se, "boot_ci_lo": lo, "boot_ci_hi": hi, "boot_p_le0": p,
                     "ann_excess_arith": (1 + ex.mean()) ** met_ppy - 1})
sdf = pd.DataFrame(sig_rows)
sdf.to_csv("out/p2_significance.csv", index=False)
print(sdf.round(5).to_string(index=False))

# ---------------------------------------------------------------- D. 归因回归
print("\n=== D. 超额归因回归 (NW 稳健标准误) ===", flush=True)
ser = {}
for name in ("SIZE_PRICE", "SIZE", "PRICE"):
    d2 = ex2.simulate(SCORE[name], grid, periods=PAIRS)
    d2["bench"] = BENCH[:len(d2)]
    ser[name] = (d2["net"] - d2["bench"]).values
n = min(len(v) for v in ser.values())
X = np.column_stack([np.ones(n), ser["SIZE"][:n], ser["PRICE"][:n]])
beta, se, t = nw_ols(ser["SIZE_PRICE"][:n], X, lags=13)
r2 = 1 - np.sum((ser["SIZE_PRICE"][:n] - X @ beta) ** 2) / np.sum(
    (ser["SIZE_PRICE"][:n] - ser["SIZE_PRICE"][:n].mean()) ** 2)
print("  超额SP = %.5f (%+.2f) + %.4f*超额SIZE (%+.1f) + %.4f*超额PRICE (%+.1f)   R2=%.4f" % (
    beta[0], t[0], beta[1], t[1], beta[2], t[2], r2))
X1 = np.column_stack([np.ones(n), ser["SIZE"][:n]])
b1, s1, t1 = nw_ols(ser["SIZE_PRICE"][:n], X1, 13)
r2_1 = 1 - np.sum((ser["SIZE_PRICE"][:n] - X1 @ b1) ** 2) / np.sum(
    (ser["SIZE_PRICE"][:n] - ser["SIZE_PRICE"][:n].mean()) ** 2)
print("  仅 SIZE 解释: R2=%.4f  NW t(SIZE)=%+.1f" % (r2_1, t1[1]))
X2 = np.column_stack([np.ones(n), ser["PRICE"][:n]])
b2, s2, t2 = nw_ols(ser["SIZE_PRICE"][:n], X2, 13)
r2_2 = 1 - np.sum((ser["SIZE_PRICE"][:n] - X2 @ b2) ** 2) / np.sum(
    (ser["SIZE_PRICE"][:n] - ser["SIZE_PRICE"][:n].mean()) ** 2)
print("  仅 PRICE 解释: R2=%.4f  NW t(PRICE)=%+.1f" % (r2_2, t2[1]))
pd.DataFrame({"coef": beta, "se": se, "t_nw": t,
              "model": ["const", "excess_SIZE", "excess_PRICE"]}).to_csv(
    "out/p2_attribution.csv", index=False)

print("\ndone  %.1fs" % (time.time() - t0))
