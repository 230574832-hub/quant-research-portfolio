"""E1-d1: tick-level reconciliation at capital = 10,000,000 yuan."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from p15d_core import Env, CapacityExecutor
from p1_engine import evaluate
import e1_engine_cap as C

CAP = 1e7
env = Env(verbose=False)
ex = CapacityExecutor(env, capital=CAP, p_max=0.05, open_share=0.10, Y=1.0,
                      st_mult=0.5, delist_mult=0.25, funding="compound",
                      min_comm_yuan=0.0, save_orders=True)
dA = ex.simulate()
od = dA.attrs["orders"]
mA = evaluate(dA)
print(f"A : CAGR {mA['cagr_net']*100:.4f} excess {mA['excess_geo']*100:.4f} IR {mA['ir']:.4f} "
      f"orders {len(od)} | final cash {dA.attrs['final_cash']:.2f} pos {dA.attrs['final_equity']-dA.attrs['final_cash']:.2f}")

B = C.run(env, env.score, env.PAIRS, env.BENCH, capital=CAP, track_pnl=True)
mB = evaluate(B["df"])
print(f"B : CAGR {mB['cagr_net']*100:.4f} excess {mB['excess_geo']*100:.4f} IR {mB['ir']:.4f} "
      f"trades {len(B['trades'])} | final cash {B['cash_end']:.2f} pos {B['pos_end']:.2f}")

# ---------- order aggregation: explain 23,354 records vs keys ----------
st = od.status.value_counts().to_dict()
keys = od[od.status.isin(["filled", "retry_filled", "partial"])].groupby(["day", "code", "side"]).size()
print("\nA order records by status:", st)
print("A unique (day,code,side) keys with any fill:", len(keys),
      "| keys with >1 record:", int((keys > 1).sum()), "| max records per key:", int(keys.max()))
print("A blocked records:", st.get("blocked", 0), "| retry_filled records:", st.get("retry_filled", 0))
# aggregate to keys
A = od[od.status.isin(["filled", "retry_filled", "partial"])].copy()
Akey = A.groupby(["day", "code", "side"]).agg(desired=("desired", "sum"), filled=("filled", "sum"),
                                              fee=("fee_total", "sum"))
Bf = B["trades"][B["trades"].status != "blocked"].copy()
Bkey = Bf.groupby(["day", "code", "side"]).agg(desired=("desired", "sum"), filled=("filled", "sum"),
                                               fee=("fee", "sum"))
j = Akey.join(Bkey, how="outer", suffixes=("_A", "_B")).fillna(0.0)
print("\nkeys A/B:", len(Akey), len(Bkey), "| union:", len(j),
      "| A-only:", int(((j.filled_A > 0) & (j.filled_B == 0)).sum()),
      "| B-only:", int(((j.filled_B > 0) & (j.filled_A == 0)).sum()))
both = j[(j.filled_A > 0) & (j.filled_B > 0)]
d = (both.filled_A - both.filled_B).abs()
print(f"matched {len(both)} | identical(<=1e-6): {(d <= 1e-6).sum()} | max |d| {d.max():.6f} | sum |d| {d.sum():.2f}")
print(f"fee: sum A {both.fee_A.sum():.2f} vs B {both.fee_B.sum():.2f} | max |d| {(both.fee_A-both.fee_B).abs().max():.6f}")

# ---------- equity / cash / backlog ----------
ea = pd.Series(dA.attrs["equity"], dtype=float).sort_index()
eb = pd.Series(B["equity"], dtype=float).sort_index()
idx = ea.index.intersection(eb.index)
rel = ((ea[idx] - eb[idx]).abs() / ea[idx])
print(f"\nequity days {len(idx)} | max rel {rel.max():.3e} | days>1e-9 {(rel>1e-9).sum()} | days>1e-6 {(rel>1e-6).sum()}")

# ---------- save ----------
j.reset_index().to_csv("out/e1d1_order_keys.csv", index=False)
pd.DataFrame({"eqA": ea, "eqB": eb, "rel": rel}).to_csv("out/e1d1_equity.csv")
B["trades"].to_csv("out/e1d1_trades_B.csv", index=False)
od.to_csv("out/e1d1_orders_A.csv", index=False)
import json
json.dump({"A": {"cagr": mA["cagr_net"], "excess": mA["excess_geo"], "ir": mA["ir"],
                 "orders": len(od), "final_cash": dA.attrs["final_cash"],
                 "final_equity": dA.attrs["final_equity"]},
           "B": {"cagr": mB["cagr_net"], "excess": mB["excess_geo"], "ir": mB["ir"],
                 "trades": len(B["trades"]), "final_cash": B["cash_end"], "pos_end": B["pos_end"]},
           "order_records_by_status": st, "unique_keys": len(Akey),
           "keys_multi_record": int((keys > 1).sum()),
           "matched": len(both), "identical": int((d <= 1e-6).sum()),
           "max_abs_diff": float(d.max()), "sum_abs_diff": float(d.sum()),
           "equity_max_rel": float(rel.max())},
          open("out/e1d1_summary.json", "w"), ensure_ascii=False, indent=1, default=float)
print("\nsaved out/e1d1_summary.json")
