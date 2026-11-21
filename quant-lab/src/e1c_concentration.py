"""P1.5-E1-c: concentration audit of the SIZE+PRICE joint selection.

Uses the exact-mirror engine (dCAGR 0.04pct vs the frozen engine) with per-code,
per-period PnL accounting, plus the frozen A engine for the ex-ante exclusion run.
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from p15d_core import Env, CapacityExecutor
from p1_engine import cs_rank, evaluate
import e1_engine_exact as E

env = Env(verbose=False)
D = pickle.load(open("data/factors_v5.pkl", "rb"))
SIZE = np.asarray(D["factors"]["SIZE"].values, np.float64)
PRICE = np.asarray(D["factors"]["PRICE"].values, np.float64)
score = env.score
pairs = env.PAIRS
bench = env.BENCH
P = env.ao

# ---------- 1. per-period, per-code PnL from the mirror ----------------------
B = E.run(env, score, pairs, track_pnl=True, return_pos=True)
df = B["df"].copy(); df["bench"] = bench[:len(df)]
m = evaluate(df)
print(f"mirror: CAGR {m['cagr_net']*100:.4f} excess {m['excess_geo']*100:.4f} IR {m['ir']:.4f}")
pos_daily = B["pos_daily"]
rows = []
for k, (i0, i1) in enumerate(pairs):
    e, s = i0 + 1, i1 + 1
    if s >= env.N or s - 1 not in [env.dates.get_loc(d) for d in []]:
        pass
    pos = pos_daily.get(env.dates[e])
    if pos is None or s >= env.N:
        continue
    r = P[s] / P[e] - 1.0
    good = np.isfinite(r)
    pnl = np.where(good, pos * r, 0.0)
    idx = np.where(np.abs(pnl) > 1e-9)[0]
    for c in idx:
        rows.append({"period": k, "end_date": env.dates[s], "code": env.codes[c],
                     "pnl": float(pnl[c]), "pos": float(pos[c])})
pnl = pd.DataFrame(rows)
pnl.to_csv("out/e1c_pnl_panel.csv.gz", index=False, compression="gzip")
tot = pnl.pnl.sum()
print(f"period-PnL rows {len(pnl):,} | total {tot:,.0f} | final equity {B['equity'][max(B['equity'])]:,.0f}")

# ---------- 2. concentration -------------------------------------------------
by_code = pnl.groupby("code").pnl.sum().sort_values(ascending=False)
by_code.to_csv("out/e1c_pnl_by_code.csv")
gross_pos = by_code[by_code > 0].sum()
conc = {
    "n_codes_traded": int(len(by_code)),
    "n_codes_positive": int((by_code > 0).sum()),
    "total_pnl": float(tot),
    "top1_share_of_total": float(by_code.iloc[0] / tot),
    "top5_share_of_total": float(by_code.iloc[:5].sum() / tot),
    "top10_share_of_total": float(by_code.iloc[:10].sum() / tot),
    "top20_share_of_total": float(by_code.iloc[:20].sum() / tot),
    "top5_share_of_positive": float(by_code.iloc[:5].sum() / gross_pos),
    "top10_share_of_positive": float(by_code.iloc[:10].sum() / gross_pos),
    "top5_codes": list(by_code.index[:5]),
}
print("\nconcentration:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in conc.items()}, ensure_ascii=False))

# ---------- 3. yearly contribution ------------------------------------------
pnl["year"] = pnl.end_date.dt.year
yr = pnl.groupby("year").pnl.sum()
yr_share = pnl.groupby(["year", "code"]).pnl.sum().reset_index()
def _yr(d):
    t = d["pnl"].sum()
    return pd.Series({"total": t,
                      "top5_share": d.nlargest(5, "pnl")["pnl"].sum() / t if t else np.nan,
                      "top1_share": d.nlargest(1, "pnl")["pnl"].sum() / t if t else np.nan})
yrc = yr_share.groupby("year").apply(_yr)
yrc["total"] = yr
yrc.to_csv("out/e1c_yearly_concentration.csv")
print("\nyearly PnL and concentration:")
print(yrc.round(3).to_string())

# ---------- 4. remove best 5 (ex-post profit deduction) ---------------------
eq = pd.Series(B["equity"]).sort_index()
e0, e1 = eq.index[0], eq.index[-1]
yrs = (env.dates[e1] - env.dates[e0]).days / 365.25
eq_end = eq.iloc[-1]
top5 = by_code.index[:5]
pnl_top5 = by_code.iloc[:5].sum()
eq_adj = eq_end - pnl_top5
cagr_adj = (eq_adj / eq.iloc[0]) ** (1 / yrs) - 1
eqb = np.cumprod(1 + bench[:len(df)])
bench_cagr = eqb[-1] ** (1 / yrs) - 1
excess_adj = ((eq_adj / eq.iloc[0]) / eqb[-1]) ** (1 / yrs) - 1
print(f"\nremove best-5 (ex-post deduction): PnL removed {pnl_top5:,.0f} "
      f"({pnl_top5/tot:.1%} of total)")
print(f"  full:    CAGR {m['cagr_net']*100:.2f}%  excess {m['excess_geo']*100:.2f}%")
print(f"  adjusted: CAGR {cagr_adj*100:.2f}%  excess {excess_adj*100:.2f}%")

# ---------- 5. ex-ante exclusion run (fixed list from 2011-2018 only) -------
pre = pnl[pnl.year <= 2018].groupby("code").pnl.sum().sort_values(ascending=False)
pre5 = list(pre.index[:5])
print("\ntop-5 winners selected on 2011-2018 only:", pre5)
codes_idx = [env.codes.get_loc(c) for c in pre5]
mask2 = env.mask.copy()
mask2[:, codes_idx] = False
score2 = np.nanmean([cs_rank(SIZE, mask2), cs_rank(PRICE, mask2)], axis=0).astype(np.float32)
bench2 = E.bench_for(env, pairs, "open")
env2 = env
env2.mask = mask2
ex = CapacityExecutor(env2, capital=1e5, Y=0.0, unlimited=True, save_orders=False)
d2 = ex.simulate(score=score2, pairs=pairs, bench=bench2)
m2 = evaluate(d2)
print(f"ex-ante exclusion run: CAGR {m2['cagr_net']*100:.2f}%  excess {m2['excess_geo']*100:.2f}%  "
      f"IR {m2['ir']:.3f}")

# ---------- 6. SIZE x PRICE 5x5 grid ----------------------------------------
print("\nSIZE x PRICE 5x5 (equal-weight forward period return, %)")
s_r = pd.DataFrame(SIZE).where(env.mask).rank(axis=1, pct=True).values
p_r = pd.DataFrame(PRICE).where(env.mask).rank(axis=1, pct=True).values
s_b = np.clip(np.floor(s_r * 5), 0, 4)
p_b = np.clip(np.floor(p_r * 5), 0, 4)
# bucket 0 = smallest SIZE rank (most negative -ln(mktcap) => smallest cap), 4 = largest
grid_sum = np.zeros((5, 5)); grid_n = np.zeros((5, 5)); grid_amt = np.zeros((5, 5))
for i0, i1 in pairs:
    r = P[i1 + 1] / P[i0 + 1] - 1.0
    ok = env.mask[i0] & np.isfinite(r)
    for a in range(5):
        for b in range(5):
            sel = ok & (s_b[i0] == a) & (p_b[i0] == b)
            n = sel.sum()
            if n > 0:
                grid_sum[a, b] += np.nansum(r[sel]); grid_n[a, b] += n
                grid_amt[a, b] += np.nansum(env.adtv[i0][sel])
grid = pd.DataFrame((grid_sum / np.maximum(grid_n, 1) * 100),
                    index=[f"SIZE Q{i+1}" for i in range(5)],
                    columns=[f"PRICE Q{j+1}" for j in range(5)])
grid_n_df = pd.DataFrame(grid_n / len(pairs), index=grid.index, columns=grid.columns)
grid.to_csv("out/e1c_grid_5x5.csv")
grid_n_df.to_csv("out/e1c_grid_5x5_count.csv")
print(grid.round(2).to_string())
print("\n(B 引擎组合的选股: SIZE Q1/Q2 且 PRICE Q1/Q2 的联合角落)")

# ---------- 7. exposure concentration ---------------------------------------
ci = pd.read_csv("data/support/company_info.csv", dtype=str)
ci = ci[ci.Scode.str.match(r"^\d{6}$", na=False)].set_index("Scode")
ind_map = ci["IndusB"].fillna("未知").to_dict()
ind = np.array([ind_map.get(c, "未知") for c in env.codes])

def picks_at(i0):
    idx = np.where(env.mask[i0] & np.isfinite(score[i0]))[0]
    return idx[np.argsort(-score[i0][idx])[:50]]

ind_w, ind_u = {}, {}
adtv_r = pd.DataFrame(env.adtv).where(env.mask).rank(axis=1, pct=True).values
price_r = pd.DataFrame(PRICE).where(env.mask).rank(axis=1, pct=True).values
mcap_r = pd.DataFrame(SIZE).where(env.mask).rank(axis=1, pct=True).values
buckets = {"ADTV": (adtv_r, {}), "PRICE": (price_r, {}), "SIZE": (mcap_r, {})}
for i0, _ in pairs[::3]:
    pk = picks_at(i0)
    uni = np.where(env.mask[i0])[0]
    for k in np.unique(ind[pk]):
        ind_w[k] = ind_w.get(k, 0) + (ind[pk] == k).sum()
    for k in np.unique(ind[uni]):
        ind_u[k] = ind_u.get(k, 0) + (ind[uni] == k).sum()
    for name, (rr, store) in buckets.items():
        q = np.clip(np.floor(rr[i0] * 5), 0, 4)
        for b in range(5):
            store[b] = store.get(b, 0) + (q[pk] == b).sum()
inds = pd.DataFrame({"组合权重": pd.Series(ind_w), "股票池权重": pd.Series(ind_u)}).fillna(0)
inds = inds.div(inds.sum(0), axis=1).sort_values("组合权重", ascending=False)
inds.to_csv("out/e1c_industry_exposure.csv")
print("\nindustry exposure (top 8):")
print(inds.head(8).round(3).to_string())
buck = pd.DataFrame({name: pd.Series(store) / sum(store.values())
                     for name, (_, store) in buckets.items()})
buck.index.name = "quintile(1=low)"
buck.to_csv("out/e1c_bucket_exposure.csv")
print("\nfactor-bucket exposure (share of holdings):")
print(buck.round(3).to_string())

# ---------- 8. 2022 decomposition ------------------------------------------
y22 = pnl[pnl.year == 2022]
by22 = y22.groupby("code").pnl.sum().sort_values(ascending=False)
print("\n2022: total PnL %.0f | top5 share %.1f%% | n codes %d" %
      (by22.sum(), 100 * by22.iloc[:5].sum() / by22.sum(), len(by22)))
top22 = by22.head(5)
print(top22.round(0).to_string())
# industry of 2022 winners
ind22 = pd.DataFrame({"code": top22.index, "pnl": top22.values})
ind22["industry"] = [ind_map.get(c, "未知") for c in ind22.code]
print(ind22.to_string(index=False))
monthly = pnl[pnl.year == 2022].groupby(pnl.end_date.dt.month).pnl.sum()
monthly.to_csv("out/e1c_2022_monthly.csv")
print("\n2022 monthly PnL:")
print(monthly.round(0).to_string())

summary = {
    "mirror_metrics": {"cagr": m["cagr_net"], "excess": m["excess_geo"], "ir": m["ir"]},
    "concentration": conc,
    "remove_best5": {"removed_pnl": float(pnl_top5), "share_of_total": float(pnl_top5 / tot),
                     "cagr_adj": float(cagr_adj), "excess_adj": float(excess_adj)},
    "exante_exclusion": {"codes": pre5, "cagr": float(m2["cagr_net"]),
                         "excess": float(m2["excess_geo"]), "ir": float(m2["ir"])},
    "grid_5x5": grid.round(3).to_dict(),
    "industry_top": inds.head(5).round(3).to_dict(),
}
json.dump(summary, open("out/e1c_summary.json", "w"), ensure_ascii=False, indent=1)
print("\nsaved out/e1c_*.csv/json")
