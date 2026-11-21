"""Compare ML prediction portfolios with single factors on the same window (2018+)."""
import pickle
import numpy as np
import pandas as pd

D = pickle.load(open("data/factors.pkl", "rb"))
P = pickle.load(open("data/panel_wide.pkl", "rb"))
dates, codes = D["dates"], D["codes"]
mask, F = D["mask"], D["factors"]
adjc = P["adjclose"]
N, M = mask.shape
pred = pd.read_parquet("data/ml_predictions.parquet")

s = pd.Series(dates, index=dates)
wdates = pd.DatetimeIndex(s.groupby(pd.Grouper(freq="W-FRI")).max().dropna().values)
fwd5 = adjc.shift(-5) / adjc - 1
ew = fwd5.where(mask).mean(axis=1)
COST = 0.0015
K = 50


def book(score_mat, start=None):
    prev, rows = set(), []
    for d in wdates:
        if start is not None and d < start:
            continue
        i = dates.get_loc(d)
        if i + 5 >= N:
            continue
        sc = score_mat[i]
        ok = mask[i] & np.isfinite(sc)
        if ok.sum() < 100:
            continue
        idx = np.where(ok)[0]
        pick = idx[np.argsort(-sc[idx])][:K]
        r = fwd5.values[i][pick]
        r = r[np.isfinite(r)]
        gross = r.mean()
        ps = set(pick)
        turn = 1 - len(ps & prev) / len(ps) if prev else 1.0
        prev = ps
        rows.append({"date": d, "gross": gross, "net": gross - COST * 2 * turn,
                     "turn": turn, "bench": ew.loc[d]})
    df = pd.DataFrame(rows).set_index("date")
    x = df.net
    ex = df.net - df.bench
    eq = (1 + x).cumprod()
    return {"ann": (1 + x.mean()) ** 52 - 1, "excess": (1 + ex.mean()) ** 52 - 1,
            "sharpe": x.mean() / x.std() * np.sqrt(52),
            "IR": ex.mean() / ex.std() * np.sqrt(52),
            "maxDD": (eq / eq.cummax() - 1).min(),
            "turn": df.turn.mean(), "curve": eq}


score = np.full((N, M), np.nan)
for d, g in pred.groupby("date"):
    i = dates.get_loc(d)
    ci = codes.get_indexer(g.code)
    score[i, ci] = g.pred_hgb.values

start = pd.Timestamp("2018-01-01")
res, curves = {}, {}
for name, mat in [("HGBR", score), ("RIDGE", None), ("IDIO_20", F["IDIO_20"].values),
                  ("SIZE", F["SIZE"].values), ("PRICE", F["PRICE"].values),
                  ("MOM_20", F["MOM_20"].values)]:
    if name == "RIDGE":
        sc = np.full((N, M), np.nan)
        for d, g in pred.groupby("date"):
            i = dates.get_loc(d)
            ci = codes.get_indexer(g.code)
            sc[i, ci] = g.pred_ridge.values
        mat = sc
    st = book(mat, start)
    curves[name] = st.pop("curve")
    res[name] = st

out = pd.DataFrame(res).T
out.to_csv("out/ml_portfolio.csv")
print(out.drop(columns=[]).round(3).to_string())
ew_wk = ew.loc[[d for d in wdates if d >= start]]
curves["EW_MARKET"] = (1 + ew_wk).cumprod()
pd.DataFrame(curves).to_csv("out/curves_2018.csv")
print("\nwindow:", start.date(), "->", ew.loc[ew.index >= start].index[-1].date())
print("EW market ann: %.3f" % ((1 + ew.loc[ew.index >= start].mean()) ** 52 - 1))
