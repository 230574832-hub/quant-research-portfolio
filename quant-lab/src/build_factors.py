"""Construct the factor library and forward-looking labels on the wide panel."""
import pickle
import numpy as np
import pandas as pd

P = pickle.load(open("data/panel_wide.pkl", "rb"))
close, open_, high, low = P["close"], P["open"], P["high"], P["low"]
adjc, volume, amount = P["adjclose"], P["volume"], P["amount"]
mktcap, cir, is_st = P["mktcap"], P["cir_shr"], P["is_st"]
preclose, ret = P["preclose"], P["ret"]
dates, codes = P["dates"], P["codes"]
board = P["board"]

# adjusted OHLC (dividend/split consistent)
f = (adjc / close).replace([np.inf, -np.inf], np.nan)
adj_open, adj_high, adj_low = open_ * f, high * f, low * f
r = adjc.pct_change(fill_method=None).astype(np.float32)
traded = (volume > 0) & (amount > 0)
r = r.where(traded)                      # suspended day -> no information
mkt = r.mean(axis=1)                                  # equal-weight market return

EPS = 1e-12
turnover = (volume / cir).replace([np.inf, -np.inf], np.nan)  # shares / float shares
turnover = turnover.where(traded)


def cumret(n):
    return adjc.shift(1) / adjc.shift(1 + n) - 1


def rmax(mat, n):
    return mat.rolling(n, min_periods=max(5, n // 2)).max()


def rmin(mat, n):
    return mat.rolling(n, min_periods=max(5, n // 2)).min()


def rmean(mat, n):
    return mat.rolling(n, min_periods=max(5, n // 2)).mean()


def rstd(mat, n):
    return mat.rolling(n, min_periods=max(5, n // 2)).std()


F = {}
print("building factors ...")
# --- momentum / reversal ---------------------------------------------------
F["REV_1"] = -r
F["REV_5"] = -cumret(5)
F["MOM_20"] = cumret(20)
F["MOM_60"] = cumret(60)
F["MOM_120"] = cumret(120)
F["MOM_250"] = cumret(250)

# --- risk ------------------------------------------------------------------
F["VOL_20"] = -rstd(r, 20)
F["VOL_60"] = -rstd(r, 60)
down = r.where(r < 0)
F["DSIDE_20"] = -rstd(down, 20)
F["MAX_20"] = -rmax(r, 20)
F["SKEW_20"] = -r.rolling(20, min_periods=10).skew()

# --- liquidity / size ------------------------------------------------------
F["SIZE"] = -np.log(mktcap)
F["AMT_20"] = np.log(rmean(amount, 20).clip(lower=1))
F["TURN_20"] = -rmean(turnover, 20)
F["VRATIO"] = -(rmean(volume, 5) / rmean(volume, 20))
F["ILIiq_20"] = rmean((r.abs() / (amount + 1)), 20) * 1e8

# --- position / price ------------------------------------------------------
F["PRICE"] = -np.log(close)
F["DISTHIGH"] = adjc / rmax(adj_high, 250) - 1
F["GAPVOL_20"] = -rmean((open_ / preclose - 1).abs(), 20)

# --- overnight / intraday (factor side) ------------------------------------
on = (adj_open / adjc.shift(1) - 1)
idr = (adjc / adj_open - 1)
F["ON_20"] = rmean(on, 20)
F["ID_20"] = rmean(idr, 20)

# --- gap momentum ----------------------------------------------------------
F["GAPMOM"] = rmean((open_ - preclose) / preclose, 20)

# --- beta / idio -----------------------------------------------------------
cov = rmean(r.mul(mkt, axis=0), 60) - rmean(r, 60).mul(mkt.rolling(60, min_periods=30).mean(), axis=0)
var = (mkt ** 2).rolling(60, min_periods=30).mean() - mkt.rolling(60, min_periods=30).mean() ** 2
beta = (cov.div(var + EPS, axis=0)).clip(-3, 3)
F["BETA_60"] = -beta
F["IDIO_20"] = -rstd(r.sub(beta.mul(mkt, axis=0)), 20)

# --- labels: next trading day ---------------------------------------------
y_all = adjc.shift(-1) / adjc - 1
y_on = adj_open.shift(-1) / adjc - 1
y_id = adjc.shift(-1) / adj_open.shift(-1) - 1

# --- universe mask ---------------------------------------------------------
listed = P["close"].notna().cumsum()
liq = rmean(amount, 20) >= 2e7
traded_t1 = (volume.shift(-1) > 0) & (amount.shift(-1) > 0)
mask = (listed >= 120) & (~is_st) & board.isin(["main", "chinext", "star"]).values[None, :] \
    & (close >= 2) & (volume > 0) & (amount > 0) & liq \
    & y_all.notna() & preclose.notna() & traded_t1 \
    & (open_.shift(-1) > 0) & (adj_open.shift(-1) > 0)

print("building limit-status matrices ...")
# limit flags: can't buy at T+1 open if the next open is limit-up (approx by board)
lim_pct = pd.Series(np.where(board.values == "main", 0.098,
                    np.where(board.values == "chinext", 0.198, 0.198)), index=codes)
up_open = open_.shift(-1) / preclose.shift(-1) - 1 >= lim_pct
dn_open = open_.shift(-1) / preclose.shift(-1) - 1 <= -lim_pct
blocked_buy = up_open.fillna(False)
blocked_sell = dn_open.fillna(False)

out = {
    "factors": {k: v.astype(np.float32) for k, v in F.items()},
    "labels": {"y_all": y_all.astype(np.float32),
               "y_on": y_on.astype(np.float32),
               "y_id": y_id.astype(np.float32)},
    "mask": mask.values if hasattr(mask, "values") else mask,
    "blocked_buy": blocked_buy.values,
    "blocked_sell": blocked_sell.values,
    "dates": dates, "codes": codes,
}
pickle.dump(out, open("data/factors.pkl", "wb"), protocol=4)
print("factors:", len(F), "| mask coverage %.3f" % out["mask"].mean())
print("mask rows per year:")
m = pd.DataFrame(out["mask"], index=dates)
print(m.sum(axis=1).groupby(dates.year).mean().round(0).to_dict())
print("label means (bps):", {k: round(float(v.stack().mean()) * 1e4, 2) for k, v in out["labels"].items()})
print("blocked buy share: %.4f  blocked sell: %.4f" % (out["blocked_buy"].mean(), out["blocked_sell"].mean()))
