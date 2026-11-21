"""Realistic (T+1-feasible) weekly portfolios + fixed-universe decay diagnostics."""
import pickle
import numpy as np
import pandas as pd

D = pickle.load(open("data/factors.pkl", "rb"))
P = pickle.load(open("data/panel_wide.pkl", "rb"))
dates, codes = D["dates"], D["codes"]
mask, F, labels = D["mask"], D["factors"], D["labels"]
adjc, mktcap, board = P["adjclose"], P["mktcap"], P["board"]

N, M = mask.shape
np.random.seed(7)

# ------------------------------------------------------------------ weekly grid
dow = pd.Series(dates.dayofweek, index=dates)
# last trading day of each ISO week
wk = dates.to_series().groupby([dates.isocalendar().year, dates.isocalendar().week]).max()
wdates = pd.DatetimeIndex(sorted(set(wk.values)))
widx = [dates.get_loc(d) for d in wdates]
print("weekly dates:", len(wdates), wdates[0].date(), "->", wdates[-1].date())

# 5-day forward return measured on adjusted close (close T -> close T+5)
fwd5 = adjc.shift(-5) / adjc - 1
ew_ret = fwd5.where(mask).mean(axis=1)           # equal-weight universe benchmark
bench = ew_ret.loc[wdates]

COST_ONEWAY = 0.0015
K = 50


def weekly_book(score, cost=COST_ONEWAY, topk=K, long_only=True):
    """score: date x code matrix; hold top-k equal weight for 5 days."""
    prev = set()
    rows = []
    ind_all = {}
    for d in wdates:
        i = dates.get_loc(d)
        if i + 5 >= N:
            continue
        s = score[i]
        ok = mask[i] & ~np.isnan(s)
        if ok.sum() < 100:
            continue
        idx = np.where(ok)[0]
        order = idx[np.argsort(-s[idx])]
        pick = order[:topk]
        r = fwd5.iloc[i].values[pick]
        r = r[~np.isnan(r)]
        gross = r.mean() if len(r) else np.nan
        pick_set = set(pick)
        turn = 1 - len(pick_set & prev) / len(pick_set) if prev else 1.0
        net = gross - cost * 2 * turn
        prev = pick_set
        rows.append({"date": d, "gross": gross, "net": net, "turnover": turn,
                     "bench": bench.loc[d] if d in bench.index else np.nan})
    df = pd.DataFrame(rows).set_index("date")
    return df


def stats(df, ann=52):
    x = df["net"].dropna()
    excess = (df["net"] - df["bench"]).dropna()
    ann_ret = (1 + x.mean()) ** ann - 1
    eq = (1 + x).cumprod()
    return {
        "net_ann": ann_ret,
        "excess_ann": (1 + excess.mean()) ** ann - 1,
        "sharpe_net": x.mean() / x.std() * np.sqrt(ann),
        "IR_excess": excess.mean() / excess.std() * np.sqrt(ann),
        "maxDD": (eq / eq.cummax() - 1).min(),
        "turnover_wk": df["turnover"].mean(),
        "hit": (excess > 0).mean(),
    }


print("\n=== weekly long-only top-%d, net of 15bp one-way ===" % K)
res = {}
for fname, fmat in F.items():
    score = fmat.values
    df = weekly_book(score)
    if len(df) < 200:
        continue
    res[fname] = stats(df)
    res[fname]["factor"] = fname

# composite: equal-weight z-score of the 8 best (by y_all IC) as a quick benchmark
best = ["IDIO_20", "MAX_20", "GAPVOL_20", "VOL_20", "REV_5", "TURN_20", "VRATIO", "VOL_60"]
comp = np.nanmean([pd.DataFrame(F[f]).where(mask).rank(axis=1).values for f in best], axis=0)
res["COMPOSITE"] = stats(weekly_book(comp))
res["COMPOSITE"]["factor"] = "COMPOSITE"

# benchmark itself
bench_stats = {"factor": "EW_MARKET", "net_ann": (1 + bench.mean()) ** 52 - 1,
               "excess_ann": 0.0, "sharpe_net": bench.mean() / bench.std() * np.sqrt(52),
               "IR_excess": np.nan, "maxDD": np.nan, "turnover_wk": 0.0, "hit": np.nan}

out = pd.DataFrame(list(res.values()) + [bench_stats]).set_index("factor")
out["net_ann"] = out["net_ann"].astype(float)
out.sort_values("IR_excess", ascending=False).to_csv("out/weekly_longonly.csv")
print(out.sort_values("IR_excess", ascending=False).round(3).to_string())

# ---------------------------------------------------------------- decay test
print("\n=== yearly RankIC on a fixed large-cap universe (top 1000 by mktcap) ===")
lg = np.log(mktcap)
rk = lg.where(mask).rank(axis=1, ascending=False)
big = mask & (rk <= 1000)


def yearly_ic(fmat, ymat, m, min_n=100):
    fr = pd.DataFrame(fmat).where(m).rank(axis=1, method="average").values.astype(np.float64)
    yr = pd.DataFrame(ymat).where(m).rank(axis=1, method="average").values.astype(np.float64)
    ok = m & ~np.isnan(fr) & ~np.isnan(yr)
    n = ok.sum(1)
    mx = np.asarray(np.where(ok, fr, 0).sum(1) / np.maximum(n, 1))
    my = np.asarray(np.where(ok, yr, 0).sum(1) / np.maximum(n, 1))
    xc = np.where(ok, fr - mx[:, None], 0)
    yc = np.where(ok, yr - my[:, None], 0)
    ic = (xc * yc).sum(1) / np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1) + 1e-18)
    ic[n < min_n] = np.nan
    return pd.Series(ic, index=dates)


y_all = labels["y_all"]
rows = {}
for f in best + ["MOM_20", "REV_1", "SIZE", "ID_20"]:
    s_all = yearly_ic(F[f], y_all, mask)
    s_big = yearly_ic(F[f], y_all, big)
    rows[f] = {"all_years": s_all.groupby(s_all.index.year).mean() * 1e4,
               "bigcap_years": s_big.groupby(s_big.index.year).mean() * 1e4}
dec = pd.DataFrame({f"{f}|all": v["all_years"] for f, v in rows.items()})
dec2 = pd.DataFrame({f"{f}|big": v["bigcap_years"] for f, v in rows.items()})
dec.join(dec2).to_csv("out/decay_fixed_universe.csv")

print("\n(mean RankIC bps by year, y_all)  全部股票池 vs 前1000大市值")
show = pd.DataFrame({f: rows[f]["all_years"].round(1) for f in ["IDIO_20", "VOL_20", "TURN_20", "REV_5", "MOM_20", "SIZE"]})
show2 = pd.DataFrame({f: rows[f]["bigcap_years"].round(1) for f in ["IDIO_20", "VOL_20", "TURN_20", "REV_5", "MOM_20", "SIZE"]})
print("ALL:\n", show.to_string())
print("BIG:\n", show2.to_string())

# trend test: slope of yearly IC on time, both universes
print("\n=== linear trend of yearly RankIC (bps/yr) ===")
tr = []
for f, v in rows.items():
    for uni in ("all_years", "bigcap_years"):
        s = v[uni].dropna()
        if len(s) > 8:
            x = np.arange(len(s))
            b = np.polyfit(x, s.values, 1)[0]
            tr.append({"factor": f, "universe": uni, "slope_bps_per_yr": b,
                       "mean_bps": s.mean(), "first5": s.iloc[:5].mean(), "last5": s.iloc[-5:].mean()})
print(pd.DataFrame(tr).round(2).to_string(index=False))

# ---------------------------------------------- T+1 feasibility decomposition
print("\n=== label decomposition by decile (best factors) ===")
rows = []
for f in ["MOM_20", "ID_20", "IDIO_20", "TURN_20"]:
    pct = pd.DataFrame(F[f]).where(mask).rank(axis=1, method="average").values
    pct = (pct - 1) / np.maximum(mask.sum(1)[:, None] - 1, 1)
    dec = np.floor(np.where(mask, pct, np.nan) * 10).clip(0, 9)
    for lab in ("y_on", "y_id", "y_all"):
        y = np.where(mask, labels[lab], np.nan)
        top = np.where(dec >= 9, y, np.nan)
        rows.append({"factor": f, "label": lab,
                     "top_decile_bps": np.nanmean(top) * 1e4,
                     "LS_bps": (np.nanmean(top) - np.nanmean(np.where(dec <= 0, y, np.nan))) * 1e4})
dec_df = pd.DataFrame(rows).pivot(index="factor", columns="label", values=["top_decile_bps", "LS_bps"])
print(dec_df.round(2).to_string())
dec_df.to_csv("out/label_decomposition.csv")
print("\ndone")
