"""Build point-in-time sequence dataset for the StockMixer / GRU reproduction.

Inputs (read-only):
  ../quant-lab/data/panel_wide.pkl          wide daily panel 2011-03-28 .. 2026-03-25
  ../quant-lab/data/drop/ts/ref_namechange.parquet   TuShare PIT name history (ST fix)

Outputs:
  data/dataset_top800.npz   compact arrays, one row per (date, stock) sample
  data/dataset_meta.pkl     metadata: dates, codes, universe definition, feature names,
                            market-state matrix, fold plan, build manifest

Design notes (leakage discipline):
  * every feature at day t uses prices/volumes with index <= t only;
  * cross-sectional z-scores are computed within date (no future information);
  * the universe is selected from data <= t (ST flag as of t, listing age, 20d ADTV, top-N mktcap);
    it does NOT condition on next-day tradability (that would be a look-ahead);
  * labels are forward returns and are stored separately; they never enter X;
  * ST flag = CSMAR st_history (as already in panel) UNION TuShare namechange, fixing the
    known 2025 ST data gap documented in quant-lab reports.
"""
import json
import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd

QL = "../quant-lab/data"
OUT_DS = "data/dataset_top800.npz"
OUT_META = "data/dataset_meta.pkl"
T = 16            # input window (trading days)
H = 1             # primary label horizon (next-day close-to-close)
TOPN = 800        # universe: top-N by market cap each day
MIN_LISTED = 120
MIN_PRICE = 2.0
MIN_ADTV = 2e7
EMBARGO = 5       # trading days gap between train and test
VAL_FRAC = 0.15

print("loading panel ...")
P = pickle.load(open(f"{QL}/panel_wide.pkl", "rb"))
dates = pd.DatetimeIndex(P["dates"])
codes = np.asarray(P["codes"])
A = P["adjclose"].values.astype(np.float64)
O = P["open"].values.astype(np.float64)
Hh = P["high"].values.astype(np.float64)
Ll = P["low"].values.astype(np.float64)
C = P["close"].values.astype(np.float64)
V = P["volume"].values.astype(np.float64)
M = P["amount"].values.astype(np.float64)
MC = P["mktcap"].values.astype(np.float64)
board = np.asarray(P["board"])
is_st = P["is_st"].astype(bool).copy()
nD, nC = A.shape
print("panel", A.shape, dates[0].date(), dates[-1].date())

# ---------------------------------------------------------------- trading-day axis
# The source daily panel carries one row per (date, code) on CALENDAR days:
# on Chinese market holidays the prices are carried forward from the previous
# session while volume/amount are 0.  Those are NOT trading days.  Leaving them
# on the axis breaks the dataset in three ways:
#   (a) `traded` is false market-wide, so every 16-day lookback that straddles
#       a holiday becomes unusable for the ENTIRE cross-section (~43% of days);
#   (b) y1[t] = adjclose[t+1]/adjclose[t]-1 is 0 for the last session before a
#       holiday (prices are carried, not traded), so the real multi-day return
#       lands one day late;
#   (c) the market-state features (5d/20d return, 20d vol) absorb those fake
#       zero returns.
# Fix: restrict the date axis to days on which the market actually traded.
mkt_turnover = np.nansum(np.where(np.isfinite(V), V, 0.0), axis=1)
trade_day = mkt_turnover > 0
n_all = len(dates)
if not trade_day.all():
    keep = np.nonzero(trade_day)[0]
    print(f"dropping {n_all - len(keep)} non-trading days (market-wide zero volume)")
    dates = dates[keep]
    A, O, Hh, Ll = A[keep], O[keep], Hh[keep], Ll[keep]
    C, V, M, MC = C[keep], V[keep], M[keep], MC[keep]
    is_st = is_st[keep]
preclose = P["preclose"].values.astype(np.float64)[
    np.nonzero(trade_day)[0]] if not trade_day.all() else P["preclose"].values.astype(np.float64)
cir_shr = P["cir_shr"].values.astype(np.float64)[
    np.nonzero(trade_day)[0]] if not trade_day.all() else P["cir_shr"].values.astype(np.float64)
nD, nC = A.shape
print(f"panel after trading-day filter: {A.shape} {dates[0].date()} .. {dates[-1].date()}"
      f"  ({nD} trading days)")
assert np.nansum(np.where(np.isfinite(V), V, 0.0), axis=1).min() > 0, \
    "still contains a market-wide zero-volume day"

# ---------------------------------------------------------------- ST union fix
print("building ST union (CSMAR + TuShare namechange) ...")
rn = pd.read_parquet(f"{QL}/drop/ts/ref_namechange.parquet")
rn = rn[rn.ts_code.str.endswith((".SH", ".SZ", ".BJ"), na=False)]
rn["code"] = rn.ts_code.str.slice(0, 6)
rn["start"] = pd.to_datetime(rn["start_date"], format="%Y%m%d", errors="coerce")
rn["end"] = pd.to_datetime(rn["end_date"], format="%Y%m%d", errors="coerce")
st_rows = rn[rn["name"].str.contains("ST", na=False) & rn["start"].notna()]
didx = {d: i for i, d in enumerate(dates)}
cidx = {c: i for i, c in enumerate(codes)}
n_rn = n_rn2 = 0
for code, g in st_rows.groupby("code"):
    if code not in cidx:
        continue
    c = cidx[code]
    for s, e in zip(g["start"], g["end"]):
        i0 = int(np.searchsorted(dates.values, np.datetime64(s), side="left"))
        i1 = nD if pd.isna(e) else int(np.searchsorted(dates.values, np.datetime64(e), side="right"))
        if i1 > i0:
            is_st[i0:i1, c] = True
            n_rn += i1 - i0
print(f"TuShare ST union added {n_rn} stock-days; total ST stock-days {int(is_st.sum())}")
# coverage diagnostic for the known 2025 gap
for yr in (2024, 2025, 2026):
    m = (dates.year == yr)
    print(f"  ST stock-days {yr}: {int(is_st[m].sum())}")

# ---------------------------------------------------------------- raw features
print("computing features ...")
f_adj = A / np.where(C > 0, C, np.nan)
aO, aH, aL = O * f_adj, Hh * f_adj, Ll * f_adj
Aprev = np.vstack([np.full((1, nC), np.nan), A[:-1]])

def roll_mean(X, n):
    df = pd.DataFrame(X)
    return df.rolling(n, min_periods=max(5, n // 2)).mean().values

Vm20 = roll_mean(V, 20)
Mm20 = roll_mean(M, 20)
MCm20 = roll_mean(MC, 20)
r1 = A / Aprev - 1.0

eps = 1e-12
F = np.empty((nD, nC, 8), dtype=np.float32)
with np.errstate(invalid="ignore", divide="ignore"):
    F[..., 0] = np.log(np.clip(aO / Aprev, 1e-4, 1e4))
    F[..., 1] = np.log(np.clip(aH / Aprev, 1e-4, 1e4))
    F[..., 2] = np.log(np.clip(aL / Aprev, 1e-4, 1e4))
    F[..., 3] = np.log(np.clip(A / Aprev, 1e-4, 1e4))
    F[..., 4] = np.log(np.clip(V / (Vm20 + eps), 1e-3, 1e3))
    F[..., 5] = np.log(np.clip(M / (Mm20 + eps), 1e-3, 1e3))
    F[..., 6] = np.log(np.clip(A / np.where(aO > 0, aO, np.nan), 1e-4, 1e4))
    F[..., 7] = np.log(np.clip(MC / (MCm20 + eps), 1e-2, 1e2))
F[~np.isfinite(F)] = np.nan
FEAT = ["log_gap_open", "log_high", "log_low", "log_ret", "log_vratio",
        "log_amtratio", "log_intraday", "log_capratio"]

# ---------------------------------------------------------------- PIT universe
print("building PIT universe ...")
traded = (V > 0) & (M > 0) & np.isfinite(A) & np.isfinite(C)
listed = pd.DataFrame(np.isfinite(A)).cumsum().values >= MIN_LISTED
ok_board = np.isin(board, ["main", "chinext", "star"])[None, :]
base = traded & listed & (~is_st) & ok_board & (C >= MIN_PRICE) & (Vm20 > 0) & (MC > 0)
base &= pd.DataFrame(M).rolling(20, min_periods=10).mean().values >= MIN_ADTV
# top-N by market cap within date
MCr = np.where(base, MC, np.nan)
rank = pd.DataFrame(MCr).rank(axis=1, ascending=False, method="first").values
univ = base & (rank <= TOPN)
print("universe names/day: mean %.0f  min %d  max %d" % (
    np.nansum(univ, axis=1).mean(), np.nansum(univ, 1).min(), np.nansum(univ, 1).max()))

# cross-sectional z-score over the full PIT base pool (same day only), clip +-3.
# z-scoring over the whole investable base (not the top-N subset) keeps features
# usable for any fixed fold universe; it still uses same-day info only.
U = base.copy()
Fm = np.where(U[..., None], F, np.nan)
mu = np.nanmean(Fm, axis=1, keepdims=True)
sd = np.nanstd(Fm, axis=1, keepdims=True)
mu = np.nan_to_num(mu, nan=0.0)
sd = np.where(~np.isfinite(sd) | (sd < 1e-8), 1.0, sd)
Fz = np.clip((F - mu) / sd, -3, 3).astype(np.float32)
Fz[~U] = np.nan
del F, Fm, mu, sd


print("z-scored features; nan fraction %.4f" % np.isnan(Fz).mean())

# ---------------------------------------------------------------- labels
Anext = np.vstack([A[1:], np.full((1, nC), np.nan)])
y1 = Anext / A - 1.0
A5 = np.vstack([A[5:], np.full((5, nC), np.nan)])
y5 = A5 / A - 1.0
# executable: buy next open, sell open 5 trading days later
aOnext = np.vstack([aO[1:], np.full((1, nC), np.nan)])
aO6 = np.vstack([aO[6:], np.full((6, nC), np.nan)])
ytrade5 = aO6 / aOnext - 1.0
# next-open limit-up proxy: blocked_buy[t] == True means the stock cannot be
# bought at the open of t+1 (it gaps limit-up there).  The array is indexed by
# the SIGNAL date, so a sample on date t must read blocked_buy[t] -- reading
# blocked_buy[t+1] would test the open of t+2 (off-by-one).
preclose_next = np.vstack([preclose[1:], np.full((1, nC), np.nan)])
lim = np.where(board == "main", 0.098, 0.198)[None, :]
Onext_raw = np.vstack([O[1:], np.full((1, nC), np.nan)])
gap_next = Onext_raw / preclose_next - 1.0
blocked_buy = gap_next >= lim
blocked_buy[~np.isfinite(blocked_buy)] = False

# ---------------------------------------------------------------- market state
mkt_r = np.nanmean(np.where(base, r1, np.nan), axis=1)
cs_disp = np.nanstd(np.where(base, r1, np.nan), axis=1)
mkt_turn = np.nanmean(np.where(base, V / np.maximum(cir_shr, 1), np.nan), axis=1)
MS = np.column_stack([
    mkt_r,
    pd.Series(mkt_r).rolling(5).mean().values,
    pd.Series(mkt_r).rolling(20).mean().values,
    pd.Series(mkt_r).rolling(20).std().values,
    cs_disp,
    np.log1p(np.nan_to_num(pd.Series(mkt_turn).rolling(20).mean().values)),
])
MS = np.nan_to_num(MS, nan=0.0).astype(np.float32)
MS = (MS - MS.mean(0)) / (MS.std(0) + 1e-8)
MS = np.clip(MS, -3, 3).astype(np.float32)
MSNAMES = ["mkt_ret_1", "mkt_ret_5", "mkt_ret_20", "mkt_vol_20", "cs_disp", "log_turn_20"]

# ---- save full-panel features for fixed-universe fold training -------------
print("saving features.npz ...")
np.savez_compressed(
    "data/features.npz",
    Fz=Fz.astype(np.float16),
    y1=y1.astype(np.float32), y5=y5.astype(np.float32),
    ytrade5=ytrade5.astype(np.float32),
    base=base, blocked_buy=blocked_buy,
    MC=MC.astype(np.float32), MS=MS,
    dates=np.array([d.strftime("%Y-%m-%d") for d in dates]),
    codes=codes.astype(str),
)
print("features.npz saved")

# ---------------------------------------------------------------- compact rows
print("building compact sample rows ...")
valid_all = np.isfinite(Fz).all(axis=2)
win_ok = (pd.DataFrame(valid_all).astype(np.float32)
          .rolling(T, min_periods=T).sum().values == T)
win_ok[: T - 1] = False
# base sample: universe at t, full finite window, label present, entry price present
samp = univ & win_ok & np.isfinite(y1) & np.isfinite(aOnext) & (aOnext > 0)
row_date, row_code = np.nonzero(samp)
n = len(row_date)
print("samples:", n)
# X[t] = window ending at t  => gather Fz[t-T+1 .. t]
print("gathering X ...")
X = np.empty((n, T, 8), dtype=np.float16)
for k in range(T):
    # for row i at date d: window day d-T+1+k
    X[:, k, :] = Fz[row_date - (T - 1) + k, row_code, :]
y1v = y1[row_date, row_code].astype(np.float32)
y5v = y5[row_date, row_code].astype(np.float32)
yt5 = ytrade5[row_date, row_code].astype(np.float32)
bb = blocked_buy[row_date, row_code]
mkt_row = MS[row_date]
np.savez_compressed(
    OUT_DS,
    X=X, y1=y1v, y5=y5v, ytrade5=yt5, blocked_buy=bb.astype(np.int8),
    date_idx=row_date.astype(np.int32), code_idx=row_code.astype(np.int32),
    mkt=mkt_row,
)
print("X float16 bytes: %.0f MB" % (X.nbytes / 1e6))

# ---------------------------------------------------------------- fold plan
print("building fold plan ...")
blocks = [("2016-01-01", "2018-01-01"), ("2018-01-01", "2020-01-01"),
          ("2020-01-01", "2022-01-01"), ("2022-01-01", "2024-01-01"),
          ("2024-01-01", "2026-03-26")]
fold_meta = []
for ts, te in blocks:
    i_te0 = int(np.searchsorted(dates.values, np.datetime64(ts), "left"))
    i_te1 = int(np.searchsorted(dates.values, np.datetime64(te), "left"))
    i_tr_end = i_te0 - EMBARGO - 1          # last train day strictly before embargo window
    tr_dates = np.arange(0, i_tr_end)
    # validation: last VAL_FRAC of train
    nv = int(len(tr_dates) * VAL_FRAC)
    va_dates = tr_dates[-nv:]
    tr_dates = tr_dates[:-nv]
    # purge validation tail too
    va_dates = va_dates[va_dates < i_tr_end - H - EMBARGO]
    fold_meta.append(dict(
        name=f"{ts[:4]}-{te[:4]}", test_start=ts, test_end=te,
        train_range=[str(dates[0].date()), str(dates[int(tr_dates.max())].date())],
        val_range=[str(dates[int(va_dates.min())].date()), str(dates[int(va_dates.max())].date())],
        test_range=[str(dates[i_te0].date()), str(dates[i_te1 - 1].date())],
        test_date_lo=i_te0, test_date_hi=i_te1,
        train_date_hi=int(tr_dates.max()), val_date_lo=int(va_dates.min()),
        val_date_hi=int(va_dates.max()),
        n_train=int(np.isin(row_date, tr_dates).sum()),
        n_val=int(np.isin(row_date, va_dates).sum()),
        n_test=int(((row_date >= i_te0) & (row_date < i_te1)).sum()),
    ))
for f in fold_meta:
    print("  fold", f["name"], "train", f["train_range"], "val", f["val_range"],
          "test", f["test_range"], f["n_train"], f["n_val"], f["n_test"])

meta = dict(dates=dates, codes=codes, feat_names=FEAT, T=T, H=H, TOPN=TOPN,
            MIN_LISTED=MIN_LISTED, MIN_PRICE=MIN_PRICE, MIN_ADTV=MIN_ADTV,
            EMBARGO=EMBARGO, folds=fold_meta, mkt_state_names=MSNAMES,
            n_samples=int(n), build_time=datetime.now().isoformat(timespec="seconds"),
            st_union_extra_stock_days=int(n_rn))
pickle.dump(meta, open(OUT_META, "wb"))
print("saved", OUT_DS, OUT_META)
