"""Walk-forward ML baseline on the factor library (weekly, purged by design).

Compare: Ridge vs HistGradientBoosting vs best single factor.
Targets: next-week return (tradable) and next-day return (for the horizon question).
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

D = pickle.load(open("data/factors.pkl", "rb"))
P = pickle.load(open("data/panel_wide.pkl", "rb"))
dates, codes = D["dates"], D["codes"]
mask, F, labels = D["mask"], D["factors"], D["labels"]
adjc = P["adjclose"]

N, M = mask.shape
fnames = [k for k in F if k not in ("GAPMOM",)]   # duplicate of ON_20
print("features:", len(fnames))

# weekly grid
s = pd.Series(dates, index=dates)
wdates = pd.DatetimeIndex(s.groupby(pd.Grouper(freq="W-FRI")).max().dropna().values)
widx = np.array([dates.get_loc(d) for d in wdates])
fwd5 = (adjc.shift(-5) / adjc - 1)

# stack weekly cross-sections
rows = []
for d in wdates:
    i = dates.get_loc(d)
    if i + 5 >= N:
        continue
    m = mask[i]
    if m.sum() < 200:
        continue
    X = np.column_stack([F[f].values[i] for f in fnames])
    y5 = fwd5.values[i]
    y1 = labels["y_all"].values[i]
    ok = m & np.isfinite(y5) & np.isfinite(y1) & np.isfinite(X).all(1)
    if ok.sum() < 200:
        continue
    df = pd.DataFrame(X[ok], columns=fnames)
    df["date"] = d
    df["code"] = codes[ok]
    df["y5"] = y5[ok]
    df["y1"] = y1[ok]
    df["y5rank"] = pd.Series(y5[ok]).rank(pct=True).values - 0.5
    rows.append(df)

Df = pd.concat(rows, ignore_index=True)
print("weekly panel:", Df.shape, "dates:", Df.date.nunique())
Df.to_parquet("data/weekly_ml.parquet", index=False)

# rank-normalise features cross-sectionally each week (robust to outliers)
def rank_norm(df):
    X = df[fnames].values
    r = pd.DataFrame(X).rank(axis=0, pct=True).values - 0.5
    return r


X = rank_norm(Df)
Df[fnames] = X

years = sorted(Df.date.dt.year.unique())
print("years:", years)

records, preds = [], []
for ty in [y for y in years if y >= 2018]:
    tr = Df.date.dt.year < ty - 1          # 1-year gap = purge
    va = Df.date.dt.year == ty - 1
    te = Df.date.dt.year == ty
    if tr.sum() < 50000 or te.sum() < 5000:
        continue
    Xtr, ytr = Df.loc[tr, fnames].values, Df.loc[tr, "y5rank"].values
    Xva, yva = Df.loc[va, fnames].values, Df.loc[va, "y5rank"].values
    Xte = Df.loc[te, fnames].values

    # --- nonlinear
    hgb = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05,
                                        max_leaf_nodes=31, min_samples_leaf=200,
                                        l2_regularization=1.0, early_stopping=True,
                                        validation_fraction=0.15, random_state=0)
    hgb.fit(Xtr, ytr)
    ph = hgb.predict(Xte)
    # --- linear
    rid = Ridge(alpha=100.0)
    rid.fit(Xtr, ytr)
    pr = rid.predict(Xte)

    tmp = Df.loc[te, ["date", "code", "y5", "y1", fnames[0]]].copy()
    tmp["pred_hgb"] = ph
    tmp["pred_ridge"] = pr
    # single-factor benchmarks
    for f in ("SIZE", "IDIO_20", "TURN_20", "PRICE", "VOL_20", "REV_5"):
        tmp[f"pred_{f}"] = Df.loc[te, f].values
    volset = ["IDIO_20", "VOL_20", "VOL_60", "TURN_20", "GAPVOL_20", "MAX_20"]
    tmp["pred_VOLCOMP"] = Df.loc[te, volset].mean(axis=1).values
    preds.append(tmp)

    def ic_by_week(score):
        s = []
        for d, g in tmp.assign(score=score).groupby("date"):
            if len(g) > 100:
                s.append(pd.Series(g.score).corr(pd.Series(g.y5), method="spearman"))
        s = pd.Series(s)
        return s.mean(), s.mean() / s.std() * np.sqrt(len(s))

    row = {"test_year": ty, "n": len(tmp)}
    for k in ["pred_hgb", "pred_ridge", "pred_SIZE", "pred_IDIO_20",
              "pred_TURN_20", "pred_PRICE", "pred_VOL_20", "pred_REV_5",
              "pred_VOLCOMP"]:
        ic, icir = ic_by_week(tmp[k])
        row[f"IC_{k[5:]}"] = ic
        row[f"ICIR_{k[5:]}"] = icir
    records.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

ml = pd.DataFrame(records)
ml.to_csv("out/ml_walkforward.csv", index=False)
print("\nmean across test years:")
print(ml.drop(columns=["test_year"]).mean().round(4).to_string())

allpred = pd.concat(preds, ignore_index=True)
allpred.to_parquet("data/ml_predictions.parquet", index=False)

# aggregate IC time series for the report
agg = []
for k in ["pred_hgb", "pred_ridge", "pred_SIZE", "pred_IDIO_20", "pred_TURN_20",
          "pred_PRICE", "pred_VOL_20", "pred_REV_5", "pred_VOLCOMP"]:
    ics = []
    for d, g in allpred.groupby("date"):
        if len(g) > 100:
            ics.append((d, pd.Series(g[k]).corr(pd.Series(g.y5), method="spearman")))
    s = pd.DataFrame(ics, columns=["date", "ic"]).set_index("date")
    agg.append(s.rename(columns={"ic": k}))
agg = pd.concat(agg, axis=1)
agg.to_csv("out/ml_ic_daily.csv")
print("\npooled OOS RankIC:")
print(agg.mean().round(4).to_string())

# feature importance via permutation on the last fit
imp = pd.Series(hgb.feature_importances_ if hasattr(hgb, "feature_importances_") else np.zeros(len(fnames)),
                index=fnames)
print("\nlast-year HGBR feature importance (gain-free; permutation below):")
from sklearn.inspection import permutation_importance
idx = np.random.choice(len(Xte), size=min(20000, len(Xte)), replace=False)
pi = permutation_importance(hgb, Xte[idx], tmp["y5"].values[idx] if "y5" in tmp else Df.loc[te, "y5"].values[idx],
                            n_repeats=3, random_state=0, scoring="neg_mean_squared_error")
pim = pd.Series(pi.importances_mean, index=fnames).sort_values(ascending=False)
pim.to_csv("out/ml_importance.csv")
print(pim.head(12).round(6).to_string())
