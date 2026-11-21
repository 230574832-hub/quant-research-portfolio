"""P1 引擎: 统一收益评价体系 + 可交易执行模拟器 + 因子中性化/回归工具.

验收标准对应:
  真实性 -> execute_at_next_open (信号 d 收盘生成, d+1 开盘成交) + 涨跌停/停牌处理 + 显式成本
  唯一口径 -> 所有收益指标只从 evaluate() 产生; 几何CAGR为唯一年化定义, 算术年化仅作对照列
  唯一基准 -> 同窗口、同持有期的等权全池(不交易, 无成本)
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 基础工具


def cs_rank(a, m):
    """截面 percentile rank (0-1). a:(N,M) float, m:(N,M) bool -> (N,M) float32"""
    big = np.where(m, a, -np.inf)
    order = np.argsort(big, axis=1, kind="stable")
    ranks = np.empty_like(order)
    idx = np.arange(a.shape[1], dtype=order.dtype)[None, :]
    np.put_along_axis(ranks, order, np.broadcast_to(idx, order.shape), axis=1)
    cnt = m.sum(1, keepdims=True)
    out = np.where(m, (ranks + 0.5) / np.maximum(cnt, 1), np.nan)
    return out.astype(np.float32)


def row_corr(X, Y, m):
    """逐日截面 Spearman 相关(输入须为 rank). -> (N,) float64"""
    ok = m & np.isfinite(X) & np.isfinite(Y)
    n = ok.sum(1)
    Xc = np.where(ok, X, 0.0).astype(np.float64)
    Yc = np.where(ok, Y, 0.0).astype(np.float64)
    Xc -= (Xc.sum(1, keepdims=True) / np.maximum(n, 1)[:, None]) * ok
    Yc -= (Yc.sum(1, keepdims=True) / np.maximum(n, 1)[:, None]) * ok
    Xc *= ok
    Yc *= ok
    cov = (Xc * Yc).sum(1)
    ic = cov / np.maximum(np.sqrt((Xc ** 2).sum(1) * (Yc ** 2).sum(1)), 1e-12)
    ic[n < 100] = np.nan
    return ic


def cs_zscore(a, m, winsor=0.01):
    """截面标准化(可选缩尾). a:(N,M) -> (N,M)"""
    x = np.where(m, a, np.nan).astype(np.float64)
    lo = np.nanquantile(x, winsor, axis=1, keepdims=True)
    hi = np.nanquantile(x, 1 - winsor, axis=1, keepdims=True)
    x = np.clip(x, lo, hi)
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    return ((x - mu) / np.maximum(sd, 1e-12)).astype(np.float32)


# ------------------------------------------------------------ 中性化工具


def industry_matrix(codes, industry):
    """静态行业 one-hot I:(M,K) float32, 以及每只股票的行业索引"""
    ind = pd.Series(np.asarray(industry), index=codes)
    uniq = sorted(ind.unique())
    K = len(uniq)
    I = np.zeros((len(codes), K), dtype=np.float32)
    kidx = ind.map({u: k for k, u in enumerate(uniq)}).values
    I[np.arange(len(codes)), kidx] = 1.0
    return I, kidx, uniq


def neutralize(a, m, I=None, ctrl=None):
    """截面中性化残差: 先行业内去均值(可选), 再对 ctrl(如 ln 市值)做截面回归取残差.
    a:(N,M), m:(N,M), I:(M,K), ctrl:(N,M) -> (N,M)"""
    y = np.where(m, a, np.nan).astype(np.float64)
    ok = m.copy()
    if I is not None:
        W = ok.astype(np.float64)                  # (N,M)
        V = np.where(ok, y, 0.0)
        S = I.T @ V.T                             # (K,N)
        C = I.T @ W.T                             # (K,N)
        with np.errstate(all="ignore"):
            ind_mean = (S / np.maximum(C, 1)).T   # (N,K)
        y = y - ind_mean[:, kidx_of(I)]
        ok = ok & np.isfinite(y)
    if ctrl is not None:
        x = np.where(ok, ctrl, np.nan).astype(np.float64)
        xm = np.nanmean(x, axis=1, keepdims=True)
        ym = np.nanmean(y, axis=1, keepdims=True)
        xc = np.where(ok, x - xm, 0.0)
        yc = np.where(ok, y - ym, 0.0)
        beta = (xc * yc).sum(1) / np.maximum((xc ** 2).sum(1), 1e-12)
        y = y - beta[:, None] * (x - xm)
    out = np.where(ok, y, np.nan).astype(np.float32)
    return out


def kidx_of(I):
    return I.argmax(1)


def cross_sectional_regression(Y, Xs, m):
    """逐日截面 OLS: Y ~ 1 + X1..Xk. Xs 已标准化. 返回 coef/t/p/R2 的 (N,k)/(N,) 数组."""
    N, M = Y.shape
    K = len(Xs)
    ok = m & np.isfinite(Y)
    for X in Xs:
        ok &= np.isfinite(X)
    n = ok.sum(1)
    yv = np.where(ok, Y, 0.0).astype(np.float64)
    yv -= (yv.sum(1, keepdims=True) / np.maximum(n, 1)[:, None]) * ok
    Xv = []
    for X in Xs:
        x = np.where(ok, X, 0.0).astype(np.float64)
        x -= (x.sum(1, keepdims=True) / np.maximum(n, 1)[:, None]) * ok
        x *= ok
        Xv.append(x)
    # 正规方程 (K+1)x(K+1)
    A = np.zeros((N, K, K))
    b = np.zeros((N, K))
    for a in range(K):
        b[:, a] = (Xv[a] * yv).sum(1)
        for c in range(K):
            A[:, a, c] = (Xv[a] * Xv[c]).sum(1)
    A[:, np.arange(K), np.arange(K)] += 1e-9
    try:
        beta = np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        beta = np.zeros((N, K))
    resid = yv.copy()
    for a in range(K):
        resid -= beta[:, a][:, None] * Xv[a]
    dof = np.maximum(n - K - 1, 1)
    sigma2 = (resid ** 2).sum(1) / dof
    try:
        inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        inv = np.zeros((N, K, K))
    se = np.sqrt(np.maximum(sigma2[:, None] * np.stack([inv[:, a, a] for a in range(K)], 1), 1e-24))
    t = beta / se
    ss_tot = (yv ** 2).sum(1)
    ss_res = (resid ** 2).sum(1)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)
    bad = n < (K + 2)
    beta[bad] = np.nan
    t[bad] = np.nan
    r2[bad] = np.nan
    return beta, t, r2, n


# ------------------------------------------------------- 交易执行模拟器


class Executor:
    """次日开盘成交 + 涨跌停/停牌处理 + 显式成本."""

    def __init__(self, panel, mask, dates, cost_bps=15.0, max_delay=5, k=50):
        self.dates = dates
        self.mask = mask
        self.N, self.M = mask.shape
        self.K = k
        self.cost = cost_bps / 1e4
        self.max_delay = max_delay
        adjc = panel["adjclose"].values.astype(np.float64)
        close = panel["close"].values.astype(np.float64)
        openp = panel["open"].values.astype(np.float64)
        precl = panel["preclose"].values.astype(np.float64)
        board = panel["board"]
        with np.errstate(all="ignore"):
            self.ao = openp * (adjc / close)
        lim = np.where(np.asarray(board) == "main", 0.098, 0.198)[None, :]
        with np.errstate(all="ignore"):
            oret = openp / precl - 1.0
        self.up_open = oret >= lim
        self.dn_open = oret <= -lim
        self.ok_open = np.isfinite(self.ao) & (openp > 0)
        self.adjc = adjc

    def grid(self, hold_days=5):
        """按 ISO 周对齐的调仓网格: hold 5/10/20 交易日 -> 每 1/2/4 周调仓一次."""
        iso = self.dates.isocalendar()
        wk = self.dates.to_series().groupby([iso.year, iso.week]).max()
        wdates = pd.DatetimeIndex(sorted(set(wk.values)))
        step = {5: 1, 10: 2, 20: 4}[hold_days]
        return np.array([self.dates.get_loc(d) for d in wdates[::step]])

    def simulate(self, score, rebal, mode="C", exclude=None, periods=None):
        """score:(N,M) 信号(越大越买); rebal: 调仓日索引数组.
        mode A=收盘成交(报告口径) B=次日开盘 C=次日开盘+涨跌停/停牌.
        exclude: (N,M) bool, True 表示该股当期不可持有(用于极端子集检验).
        periods: 可选, 显式给定 [(i0,i1), ...] 期间对, 用于跨版本统一口径.
        返回 df[date, gross, net, turn, bench, n_hold]"""
        prev, rows = set(), []
        diag = {"buy_blocked": 0, "sell_delayed": 0, "unresolved": 0, "trades": 0,
                "excluded": 0}
        pairs = periods if periods is not None else [
            (rebal[k], rebal[k + 1]) for k in range(len(rebal) - 1)]
        for k in range(len(pairs)):
            i, i2 = pairs[k]
            if i + 1 >= self.N or i2 + 1 >= self.N:
                continue
            d = self.dates[i]
            s = score[i]
            ok = self.mask[i] & np.isfinite(s)
            if ok.sum() < 100:
                continue
            idx = np.where(ok)[0]
            pick = idx[np.argsort(-s[idx])[:self.K]]
            ps = set(pick.tolist())
            turn = 1 - len(ps & prev) / len(ps) if prev else 1.0
            prev = ps
            if mode == "A":
                with np.errstate(all="ignore"):
                    r = self.adjc[i2, pick] / self.adjc[i, pick] - 1
                    br = self.adjc[i2] / self.adjc[i] - 1
            elif mode == "B":
                with np.errstate(all="ignore"):
                    r = self.ao[i2 + 1, pick] / self.ao[i + 1, pick] - 1
                    br = self.ao[i2 + 1] / self.ao[i + 1] - 1
            else:
                rr = []
                with np.errstate(all="ignore"):
                    br = self.ao[i2 + 1] / self.ao[i + 1] - 1
                for c in pick:
                    if exclude is not None and exclude[i, c]:
                        diag["excluded"] += 1
                        continue
                    if self.up_open[i + 1, c] or not self.ok_open[i + 1, c]:
                        diag["buy_blocked"] += 1
                        continue
                    bp = self.ao[i + 1, c]
                    j = i2 + 1
                    delay = 0
                    while (self.dn_open[j, c] or not self.ok_open[j, c]) and delay < self.max_delay and j + 1 < self.N:
                        j += 1
                        delay += 1
                    sp = self.ao[j, c] if np.isfinite(self.ao[j, c]) else self.adjc[j, c]
                    if np.isfinite(sp) and np.isfinite(bp) and bp > 0:
                        rr.append(sp / bp - 1)
                        diag["trades"] += 1
                        if delay:
                            diag["sell_delayed"] += 1
                    else:
                        diag["unresolved"] += 1
                r = np.array(rr) if rr else np.array([np.nan])
            r = np.asarray(r, dtype=np.float64)
            r = r[np.isfinite(r)]
            gross = r.mean() if len(r) else np.nan
            bm = br[self.mask[i] & np.isfinite(br)]
            bench = bm.mean() if len(bm) else np.nan
            if not np.isfinite(gross):
                continue
            rows.append({"date": d, "gross": gross, "net": gross - 2 * self.cost * turn,
                         "turn": turn, "bench": bench, "n_hold": len(r),
                         "start_date": self.dates[i + 1], "end_date": self.dates[i2 + 1]})
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["date", "gross", "net", "turn", "bench", "n_hold"])
            df = df.set_index("date")
        else:
            df = df.set_index("date")
        df.attrs["diag"] = diag
        return df


# -------------------------------------------------------- 统一评价指标


def evaluate(df, label="", periods_per_year=None):
    """唯一收益评价口径. 输入 Executor.simulate 的输出.

    periods_per_year=None -> 按真实日历时间年化(推荐):
        span = (最后一期结束日 - 第一期开始日) / 365.25
        ppy  = 期数 / span  (实际每年期数)
    这样当调仓点因股票池不足而被合并、期间长度不等时, 年化仍然正确.
    """
    if len(df) == 0:
        return {"label": label, "periods": 0, "years": 0.0, "cagr_net": np.nan,
                "cagr_bench": np.nan, "cagr_gross": np.nan, "excess_geo": np.nan,
                "arith_ann": np.nan, "vol_ann": np.nan, "sharpe": np.nan, "ir": np.nan,
                "maxDD": np.nan, "calmar": np.nan, "turnover_annual": np.nan,
                "cost_drag": np.nan, "hit_rate": np.nan, "avg_hold_n": 0.0}
    x = df["net"].values.astype(np.float64)
    g = df["gross"].values.astype(np.float64)
    b = df["bench"].values.astype(np.float64)
    n = len(x)
    if periods_per_year is None:
        if "start_date" in df.columns and "end_date" in df.columns:
            span = (df["end_date"].iloc[-1] - df["start_date"].iloc[0]).days / 365.25
        else:
            span = n / 52.0
        span = max(span, 1e-6)
        ppy = n / span
        yrs = span
    else:
        ppy = float(periods_per_year)
        yrs = n / ppy
    eq = np.cumprod(1 + x)
    eqb = np.cumprod(1 + b)
    eqg = np.cumprod(1 + g)
    cagr = eq[-1] ** (1 / yrs) - 1
    cagr_b = eqb[-1] ** (1 / yrs) - 1
    cagr_g = eqg[-1] ** (1 / yrs) - 1
    ex = x - b
    sd = x.std(ddof=1)
    sd_ex = ex.std(ddof=1)
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return {
        "label": label,
        "periods": n,
        "years": round(yrs, 2),
        "ppy_actual": round(ppy, 3),
        "cagr_net": cagr,
        "cagr_bench": cagr_b,
        "cagr_gross": cagr_g,
        "excess_geo": (eq[-1] / eqb[-1]) ** (1 / yrs) - 1,
        "arith_ann": (1 + x.mean()) ** ppy - 1,
        "vol_ann": sd * np.sqrt(ppy),
        "sharpe": x.mean() / sd * np.sqrt(ppy) if sd > 0 else np.nan,
        "ir": ex.mean() / sd_ex * np.sqrt(ppy) if sd_ex > 0 else np.nan,
        "maxDD": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "turnover_annual": df["turn"].mean() * ppy,
        "cost_drag": cagr_g - cagr,
        "hit_rate": float((ex > 0).mean()),
        "avg_hold_n": float(df["n_hold"].mean()),
    }
