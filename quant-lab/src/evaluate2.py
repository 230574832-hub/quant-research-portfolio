"""Factor evaluation v2: rank IC under three label versions + neutralisation + horizon.

Label versions
  raw : next-day total / overnight / intraday return
  neu : residual after daily cross-sectional regression on beta60 and log size
  dm  : deviation from each stock's trailing 250d mean of the same label
        (kills persistent stock-level "level effects")
"""
import pickle
import numpy as np
import pandas as pd

D = pickle.load(open("data/factors.pkl", "rb"))
dates, codes = D["dates"], D["codes"]
mask, labels, F = D["mask"], D["labels"], D["factors"]
N, M = mask.shape
COST_ONEWAY = 0.0015

# ------------------------------------------------------------------ helpers
def row_rank(a):
    return pd.DataFrame(a).where(mask).rank(axis=1, method="average").values.astype(np.float64)


def xs_resid(y, X):
    """Per-day cross-sectional OLS residual of y on X (both date x code)."""
    ok = mask & ~np.isnan(y)
    for x in X:
        ok &= ~np.isnan(x)
    yy = np.where(ok, y, 0.0)
    n = ok.sum(1)
    cols = [np.where(ok, x, 0.0) for x in X]
    # build normal equations per row
    k = len(cols) + 1
    A = np.zeros((N, k, k))
    b = np.zeros((N, k))
    A[:, 0, 0] = n
    for i, ci in enumerate(cols):
        A[:, 0, i + 1] = A[:, i + 1, 0] = ci.sum(1)
        b[:, i + 1] = (ci * yy).sum(1)
        for j, cj in enumerate(cols):
            A[:, i + 1, j + 1] = (ci * cj).sum(1)
    b[:, 0] = yy.sum(1)
    # ridge-stabilised solve
    A = A + np.eye(k)[None] * 1e-8
    coef = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    fit = coef[:, 0][:, None] * np.ones_like(yy)
    for i in range(k - 1):
        fit += coef[:, i + 1][:, None] * cols[i]
    res = np.where(ok, y - fit, np.nan)
    return res


print("preparing label versions ...")
beta = -F["BETA_60"]                      # recover raw beta
logsize = np.log(np.exp(-F["SIZE"]))
lab = {}
for k, y in labels.items():
    lab[(k, "raw")] = y
for k, y in labels.items():
    lab[(k, "neu")] = xs_resid(y.astype(np.float64), [beta, logsize]).astype(np.float32)
for k, y in labels.items():
    dm = y.astype(np.float64) - pd.DataFrame(y).rolling(250, min_periods=120).mean().shift(1).values
    lab[(k, "dm")] = dm.astype(np.float32)
print("label set:", list(lab))

YR = {}
for key, y in lab.items():
    YR[key] = row_rank(y)
del lab

print("computing IC ...")
rows, ic_series = [], {}
for fname, fmat in F.items():
    fr = row_rank(fmat)
    for (lname, ver), yr in YR.items():
        ok = mask & ~np.isnan(fr) & ~np.isnan(yr)
        n = ok.sum(1)
        have = n > 50
        mx = np.where(ok, fr, 0.0).sum(1) / np.maximum(n, 1)
        my = np.where(ok, yr, 0.0).sum(1) / np.maximum(n, 1)
        xc = np.where(ok, fr - mx[:, None], 0.0)
        yc = np.where(ok, yr - my[:, None], 0.0)
        cov = (xc * yc).sum(1) / np.maximum(n, 1)
        vx = (xc ** 2).sum(1) / np.maximum(n, 1)
        vy = (yc ** 2).sum(1) / np.maximum(n, 1)
        ic = cov / np.sqrt(vx * vy + 1e-18)
        ic[~have] = np.nan
        s = pd.Series(ic, index=dates).dropna()
        ic_series[(fname, lname, ver)] = s
        if len(s) > 100:
            rows.append({"factor": fname, "label": lname, "version": ver,
                         "n_days": len(s), "IC": s.mean(), "IC_std": s.std(),
                         "ICIR": s.mean() / s.std(),
                         "t": s.mean() / s.std() * np.sqrt(len(s)),
                         "pos_rate": (s > 0).mean(),
                         "ICIR_ann": s.mean() / s.std() * np.sqrt(252)})

ic_df = pd.DataFrame(rows)
ic_df.to_csv("out/ic_summary.csv", index=False)
pd.DataFrame({f"{f}|{l}|{v}": s for (f, l, v), s in ic_series.items()}).to_csv("out/ic_daily.csv")

print("\n=== IC (raw) by label ===")
p = ic_df[ic_df.version == "raw"].pivot(index="factor", columns="label", values="IC")
print((p * 100).round(3).sort_values("y_all", ascending=False).to_string())

print("\n=== overnight IC: raw vs neutralised vs stock-demeaned ===")
q = ic_df[ic_df.label == "y_on"].pivot(index="factor", columns="version", values="IC")
q["collapse_neu"] = 1 - q["neu"] / q["raw"]
q["collapse_dm"] = 1 - q["dm"] / q["raw"]
print((q[["raw", "neu", "dm", "collapse_neu", "collapse_dm"]] * [100, 100, 100, 1, 1]).round(3)
      .sort_values("raw", ascending=False).to_string())

print("\n=== yearly IC (y_all, raw) bps ===")
ye = {}
for (f, l, v), s in ic_series.items():
    if v == "raw" and l == "y_all":
        ye[f] = s.groupby(s.index.year).mean() * 1e4
year_df = pd.DataFrame(ye)
year_df.index.name = "year"
year_df.to_csv("out/ic_by_year.csv")
print(year_df.round(1).to_string())

# ------------------------------------------------------- horizon decay (y_all)
print("\ncomputing horizon IC (y_all, forward 1/5/20/60d) ...")
adjc = pickle.load(open("data/panel_wide.pkl", "rb"))["adjclose"]
hz_rows = []
for h in (1, 5, 20, 60):
    yh = adjc.shift(-h) / adjc - 1
    yr = row_rank(yh.astype(np.float32))
    for fname, fmat in F.items():
        fr = row_rank(fmat)
        ok = mask & ~np.isnan(fr) & ~np.isnan(yr)
        n = ok.sum(1)
        mx = np.where(ok, fr, 0.0).sum(1) / np.maximum(n, 1)
        my = np.where(ok, yr, 0.0).sum(1) / np.maximum(n, 1)
        xc = np.where(ok, fr - mx[:, None], 0.0)
        yc = np.where(ok, yr - my[:, None], 0.0)
        ic = (xc * yc).sum(1) / np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1) + 1e-18)
        ic[n <= 50] = np.nan
        hz_rows.append({"factor": fname, "horizon": h,
                        "IC": np.nanmean(ic),
                        "t": np.nanmean(ic) / np.nanstd(ic) * np.sqrt(np.isfinite(ic).sum())})
hz = pd.DataFrame(hz_rows)
hz.to_csv("out/ic_horizon.csv", index=False)
print(hz.pivot(index="factor", columns="horizon", values="IC").mul(100).round(2).to_string())

# ------------------------------------------------------------- decile books
print("\nbuilding decile books ...")
books, curves = [], {}
for fname, fmat in F.items():
    pct = row_rank(fmat)
    pct = (pct - 1) / np.maximum(mask.sum(1)[:, None] - 1, 1)
    dec = np.floor(np.where(mask, pct, np.nan) * 10).clip(0, 9)
    for lname, ymat in labels.items():
        y = np.where(mask, ymat, np.nan)
        top = np.where(dec >= 9, y, np.nan)
        bot = np.where(dec <= 0, y, np.nan)
        long = np.nansum(top, 1) / np.maximum((~np.isnan(top)).sum(1), 1)
        short = np.nansum(bot, 1) / np.maximum((~np.isnan(bot)).sum(1), 1)
        ls = long - short
        ind = ~np.isnan(top)
        overlap = (ind[1:] & ind[:-1]).sum(1)
        size = np.maximum(ind[1:].sum(1), 1)
        turn = np.r_[np.nan, 1 - overlap / size]
        s = pd.Series(ls, index=dates)
        net = s - COST_ONEWAY * 2 * s.index.to_series().map(
            pd.Series(turn, index=dates)).fillna(0) * 2
        curves[(fname, lname)] = s
        ann = lambda x: (1 + x.mean()) ** 252 - 1
        eq = (1 + net).cumprod()
        books.append({"factor": fname, "label": lname,
                      "LS_gross_ann": ann(s), "LS_net_ann": ann(net),
                      "Sharpe_gross": s.mean() / s.std() * np.sqrt(252),
                      "Sharpe_net": net.mean() / net.std() * np.sqrt(252),
                      "maxDD_net": (eq / eq.cummax() - 1).min(),
                      "turnover_daily": np.nanmean(turn),
                      "long_ann": ann(pd.Series(long, index=dates))})

bk = pd.DataFrame(books)
bk.to_csv("out/portfolio_summary.csv", index=False)
print(bk.sort_values("Sharpe_net", ascending=False).head(20).to_string(index=False))
pd.DataFrame({f"{f}|{l}": v for (f, l), v in curves.items()}).to_csv("out/ls_daily.csv")
print("\ndone")
