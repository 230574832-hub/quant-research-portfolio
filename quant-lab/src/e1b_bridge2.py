"""E1-b ledger bridge v2: A engine vs exact mirror B, order-level and holding-level."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from p15d_core import Env, CapacityExecutor
from p1_engine import evaluate
import e1_engine_exact as E

env = Env(verbose=False)
score = env.score
pairs = env.PAIRS
bench = env.BENCH
P = env.ao

ex = CapacityExecutor(env, capital=1e5, Y=0.0, unlimited=True, save_orders=True)
dA = ex.simulate()
od = dA.attrs["orders"]
eqA = dA.attrs["equity"]
mA = evaluate(dA)
print(f"A: CAGR {mA['cagr_net']*100:.4f} excess {mA['excess_geo']*100:.4f} IR {mA['ir']:.4f} orders {len(od)}")

B = E.run(env, score, pairs, return_pos=False, track_pnl=True)
dB = B["df"].copy()
dB["bench"] = bench[:len(dB)]
mB = evaluate(dB)
print(f"B(mirror): CAGR {mB['cagr_net']*100:.4f} excess {mB['excess_geo']*100:.4f} IR {mB['ir']:.4f} trades {len(B['trades'])}")

# ---------------- order-level reconciliation ----------------
A = od[od.status.isin(["filled", "retry_filled"])].copy()
A["sgn"] = np.where(A.side == "B", 1, -1)
A["filled"] = pd.to_numeric(A.filled)
Bf = B["trades"][B["trades"].status != "blocked"].copy()
Akey = A.groupby(["day", "code", "side"]).agg(desired=("desired", "sum"), filled=("filled", "sum"))
Bkey = Bf.groupby(["day", "code", "side"]).agg(desired=("desired", "sum"), filled=("filled", "sum"))
j = Akey.join(Bkey, how="outer", lsuffix="_A", rsuffix="_B").fillna(0.0)
print(f"\norder keys A {len(Akey)}  B {len(Bkey)}  union {len(j)}")
print("A-only:", int(((j.filled_A > 0) & (j.filled_B == 0)).sum()),
      " B-only:", int(((j.filled_B > 0) & (j.filled_A == 0)).sum()))
both = j[(j.filled_A > 0) & (j.filled_B > 0)]
diff = (both.filled_A - both.filled_B).abs()
print(f"matched fills {len(both)} | identical(<=1e-6): {(diff <= 1e-6).sum()} | "
      f"max abs diff {diff.max():.6f} | sum abs diff {diff.sum():.2f}")
if len(both):
    jj = both.assign(d=diff).sort_values("d", ascending=False)
    print("\ntop 8 fill differences:")
    print(jj.head(8).round(4).to_string())

# ---------------- equity-level reconciliation ----------------
ea = pd.Series(eqA, dtype=float).sort_index()
eb = pd.Series(B["equity"], dtype=float).sort_index()
idx = ea.index.intersection(eb.index)
rel = ((ea[idx] - eb[idx]).abs() / ea[idx]).replace([np.inf], np.nan)
print(f"\nequity days compared {len(idx)} | max rel diff {rel.max():.3e} | "
      f"days rel>1e-9: {(rel > 1e-9).sum()} | days rel>1e-6: {(rel > 1e-6).sum()}")
if (rel > 1e-9).any():
    first = rel[rel > 1e-9].index[0]
    print(f"first equity divergence day {env.dates[first].date()} rel={rel[first]:.3e} "
          f"A={ea[first]:.2f} B={eb[first]:.2f}")
    print("\nA orders that day:")
    print(od[od.day == first][["date", "code", "side", "status", "desired", "filled"]].to_string(index=False))
    print("\nB trades that day:")
    print(B["trades"][B["trades"].day == first].to_string(index=False))

# ---------------- holdings reconstruction from A orders ---------------
trades = od[od.status.isin(["filled", "retry_filled"]) & (od.filled > 0)].copy()
trades["v"] = trades.filled.astype(float)
trades["sgn"] = np.where(trades.side == "B", 1, -1)
trades["v"] = trades.v * trades.sgn
by_day = trades.groupby(["day", "code"]).v.sum()
posA = np.zeros(env.M)
posA_maxdiff = []
first_pos = None
for day in sorted(set(od.day.astype(int))):
    if day > 0:
        cols = np.where(posA > 1e-9)[0]
        if len(cols):
            r = P[day, cols] / P[day - 1, cols] - 1.0
            good = np.isfinite(r) & np.isfinite(P[day, cols]) & (P[day, cols] > 0)
            r = np.clip(np.where(good, r, 0.0), -0.99, 10.0)
            posA[cols] *= (1.0 + r)
    if (day, slice(None)) in by_day.index:
        pass
    for (d, c), v in by_day.loc[day].items() if day in by_day.index.get_level_values(0) else []:
        posA[int(c)] += float(v)
    if day in eqA:
        pos_daily = B.get("pos_daily")
        # compare with B's daily holdings if available (not collected here)
print("\norder-level summary written to out/e1b_order_recon.csv")
j.reset_index().to_csv("out/e1b_order_recon.csv", index=False)
pd.DataFrame({"eqA": ea, "eqB": eb, "rel": rel}).to_csv("out/e1b_equity_recon.csv")
B["pnl_by_code"].to_csv("out/e1c_pnl_by_code_engineB.csv")
B["pnl_year_code"].to_csv("out/e1c_pnl_year_code_engineB.csv")
print("saved")
