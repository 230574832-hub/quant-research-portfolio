"""P1.5-D+ (1): 委托/成交/费用会计关系的独立复算与单位勘误.

不重跑策略、不改任何参数. 只做三件事:
  A. 对 all_runs.csv 的 96 次仿真逐行检查会计恒等式;
  B. 从 8 份逐笔订单日志独立聚合, 与 metrics 汇总对账 (完全独立于引擎 diag);
  C. 复算备忘录 4.4 节 "复投/固定本金委托总额" 的单位, 生成勘误表.
"""
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

A = pd.read_csv("out/p15d_all_runs.csv")
CAPS = [1e5, 5e5, 1e6, 3e6, 1e7, 3e7, 1e8]
LAB = {1e5: "10万元", 5e5: "50万元", 1e6: "100万元", 3e6: "300万元",
       1e7: "1,000万元", 3e7: "3,000万元", 1e8: "1亿元"}
FEE_COLS = ["费用_佣金元", "费用_过户费元", "费用_印花税元",
            "费用_价差及其他元", "费用_市场冲击元"]
BASE_BPS = 15.0
FEE_LOG = ["fee_佣金", "fee_过户费", "fee_印花税",
           "fee_价差及其他", "fee_市场冲击"]

# ------------------------------------------------------------------ A. 恒等式
rows = []
for _, r in A.iterrows():
    des, trd, unf = r["委托总额元"], r["成交总额元"], r["未成交总额元"]
    fee = float(r[FEE_COLS].sum())
    base = fee - r["费用_市场冲击元"]
    mincomm = float(r["最低佣金元"])
    e = {}
    e["d_委托减成交减未成交_元"] = des - trd - unf
    e["d_成交率_pp"] = 100 * trd / des - r["金额成交率%"] if des > 0 else np.nan
    e["d_费用合计_元"] = fee - r["费用_合计元"]
    # 最低佣金档有按笔固定加成, 基础费率恒等式不成立, 单独标记
    e["基础费用隐含bp"] = 1e4 * base / trd if trd > 0 else np.nan
    e["d_基础费率_bp"] = (e["基础费用隐含bp"] - BASE_BPS) if mincomm == 0 else np.nan
    e["冲击隐含bp"] = 1e4 * r["费用_市场冲击元"] / trd if trd > 0 else np.nan
    e["总费用隐含bp"] = 1e4 * r["费用_合计元"] / trd if trd > 0 else np.nan
    e["单位换算_委托亿元"] = des / 1e8
    rows.append({"标签": r["标签"], "资金规模": r["资金规模"],
                 "资金规则": r["资金规则"], "最低佣金元": mincomm,
                 "委托总额元": des, "成交总额元": trd, "未成交总额元": unf,
                 "费用_合计元": r["费用_合计元"], "费用_市场冲击元": r["费用_市场冲击元"],
                 **e})
CHK = pd.DataFrame(rows)
CHK.to_csv("out/p15dplus_accounting_check.csv", index=False, encoding="utf-8-sig")

tol = CHK[CHK["最低佣金元"] == 0]
print("=== A. 会计恒等式 (96 次仿真) ===", flush=True)
print("  委托=成交+未成交 最大绝对误差 %.6f 元 (n=%d)" %
      (CHK["d_委托减成交减未成交_元"].abs().max(), len(CHK)), flush=True)
print("  金额成交率 最大绝对误差 %.8f pp" % CHK["d_成交率_pp"].abs().max(), flush=True)
print("  费用合计=五项之和 最大绝对误差 %.6f 元" % CHK["d_费用合计_元"].abs().max(), flush=True)
print("  基础费率(不含最低佣金档) 隐含bp 范围 %.6f ~ %.6f, 偏离15bp最大 %.3e" %
      (tol["基础费用隐含bp"].min(), tol["基础费用隐含bp"].max(),
       tol["d_基础费率_bp"].abs().max()), flush=True)
print("  最低佣金档 n=%d: 基础费率 %.3f ~ %.3f bp (按笔固定费, 属预期)" %
      (len(CHK[CHK["最低佣金元"] > 0]),
       CHK[CHK["最低佣金元"] > 0]["基础费用隐含bp"].min(),
       CHK[CHK["最低佣金元"] > 0]["基础费用隐含bp"].max()), flush=True)

# 用户反推的算术复现 (10万元复投主口径)
r10 = A[(A["资金规模元"] == 1e5) & (A["资金规则"] == "compound")
        & (A["参与率上限%"] == 5) & (A["冲击系数Y"] == 1.0)
        & (A["开盘份额"] == 0.10) & (A["最低佣金元"] == 0)].iloc[0]
base_fee = r10["费用_合计元"] - r10["费用_市场冲击元"]
print("\n=== A2. 用户反推验算 (10万元复投) ===", flush=True)
print("  费用合计 %.2f 元 - 冲击 %.2f 元 = 基础 %.2f 元" %
      (r10["费用_合计元"], r10["费用_市场冲击元"], base_fee), flush=True)
print("  基础/15bp => 成交金额 %.2f 元 = %.4f 万元" %
      (base_fee / 0.0015, base_fee / 0.0015 / 1e4), flush=True)
print("  再除以成交率 %.4f%% => 委托金额 %.2f 元 = %.4f 亿元" %
      (r10["金额成交率%"], base_fee / 0.0015 / (r10["金额成交率%"] / 100),
       base_fee / 0.0015 / (r10["金额成交率%"] / 100) / 1e8), flush=True)
print("  all_runs 实际: 委托 %.2f 元 = %.4f 亿元; 成交 %.2f 元" %
      (r10["委托总额元"], r10["委托总额元"] / 1e8, r10["成交总额元"]), flush=True)

# ------------------------------------------------------------------ B. 逐笔订单独立对账
rec = []
for cap in CAPS:
    f = "out/p15d_orders_%d.csv.gz" % int(cap)
    o = pd.read_csv(f)
    m = A[(A["资金规模元"] == cap) & (A["资金规则"] == "compound")
          & (A["参与率上限%"] == 5) & (A["冲击系数Y"] == 1.0)
          & (A["开盘份额"] == 0.10) & (A["最低佣金元"] == 0)].iloc[0]
    rec.append({
        "资金规模": LAB[cap], "订单行数": len(o),
        "日志委托合计元": o["desired"].sum(), "metrics委托元": m["委托总额元"],
        "d_委托元": o["desired"].sum() - m["委托总额元"],
        "日志成交合计元": o["filled"].sum(), "metrics成交元": m["成交总额元"],
        "d_成交元": o["filled"].sum() - m["成交总额元"],
        "日志费用合计元": o["fee_total"].sum(), "metrics费用元": m["费用_合计元"],
        "d_费用元": o["fee_total"].sum() - m["费用_合计元"],
        "日志冲击合计元": o["fee_市场冲击"].sum(),
        "metrics冲击元": m["费用_市场冲击元"],
        "d_冲击元": o["fee_市场冲击"].sum() - m["费用_市场冲击元"],
        "日志费用五项合计元": o[FEE_LOG].sum().sum(),
        "会计恒等_fee_total减五项": (o["fee_total"]
                                 - o[FEE_LOG].sum(axis=1)).abs().max(),
    })
o = pd.read_csv("out/p15d_orders_V5_unlimited.csv.gz")
rec.append({"资金规模": "V5无约束校准", "订单行数": len(o),
            "日志委托合计元": o["desired"].sum(), "metrics委托元": np.nan,
            "d_委托元": np.nan, "日志成交合计元": o["filled"].sum(),
            "metrics成交元": np.nan, "d_成交元": np.nan,
            "日志费用合计元": o["fee_total"].sum(), "metrics费用元": np.nan,
            "d_费用元": np.nan, "日志冲击合计元": o["fee_市场冲击"].sum(),
            "metrics冲击元": np.nan, "d_冲击元": np.nan,
            "日志费用五项合计元": o[FEE_LOG].sum().sum(),
            "会计恒等_fee_total减五项": (o["fee_total"]
                                     - o[FEE_LOG].sum(axis=1)).abs().max()})
REC = pd.DataFrame(rec)
REC.to_csv("out/p15dplus_order_reconcile.csv", index=False, encoding="utf-8-sig")
print("\n=== B. 逐笔订单日志独立对账 ===", flush=True)
print(REC[["资金规模", "订单行数", "d_委托元", "d_成交元", "d_费用元", "d_冲击元",
           "会计恒等_fee_total减五项"]].round(6).to_string(index=False), flush=True)

# ------------------------------------------------------------------ C. 4.4 节单位勘误
reported = {
    "10万元": ("9.72亿元", "2.26亿元"),
    "1,000万元": ("71.54亿元", "22.40亿元"),
    "1亿元": ("790.18亿元", "248.77亿元"),
}
err = []
for cap, lab in [(1e5, "10万元"), (1e7, "1,000万元"), (1e8, "1亿元")]:
    c = A[(A["资金规模元"] == cap) & (A["资金规则"] == "compound")
          & (A["参与率上限%"] == 5) & (A["冲击系数Y"] == 1.0)
          & (A["开盘份额"] == 0.10) & (A["最低佣金元"] == 0)].iloc[0]["委托总额元"]
    f = A[(A["资金规模元"] == cap) & (A["资金规则"] == "fixed")].iloc[0]["委托总额元"]
    for kind, val, rep in [("复投委托总额", c, reported[lab][0]),
                           ("固定本金委托总额", f, reported[lab][1])]:
        num = float(rep.replace("亿元", ""))
        ratio = (num * 1e8) / val
        err.append({"资金规模": lab, "项目": kind,
                    "备忘录原值": rep, "原值换算元": num * 1e8,
                    "CSV实际值元": round(val, 2),
                    "正确亿元": round(val / 1e8, 4),
                    "原值/实际倍数": round(ratio, 4),
                    "判定": "错，放大10倍" if abs(ratio - 10) < 0.15 else
                            ("正确" if abs(ratio - 1) < 0.02 else "需复核")})
ERR = pd.DataFrame(err)
ERR.to_csv("out/p15dplus_errata.csv", index=False, encoding="utf-8-sig")
print("\n=== C. 备忘录 4.4 节单位勘误 ===", flush=True)
print(ERR.to_string(index=False), flush=True)

# 汇总
summary = pd.DataFrame([
    {"项目": "恒等式: 委托=成交+未成交 最大误差元",
     "数值": CHK["d_委托减成交减未成交_元"].abs().max()},
    {"项目": "恒等式: 金额成交率 最大误差pp", "数值": CHK["d_成交率_pp"].abs().max()},
    {"项目": "恒等式: 费用合计=五项之和 最大误差元",
     "数值": CHK["d_费用合计_元"].abs().max()},
    {"项目": "基础费率隐含bp(不含最低佣金档)最大偏离",
     "数值": tol["d_基础费率_bp"].abs().max()},
    {"项目": "订单日志与metrics委托最大误差元", "数值": REC["d_委托元"].abs().max()},
    {"项目": "订单日志与metrics成交最大误差元", "数值": REC["d_成交元"].abs().max()},
    {"项目": "订单日志与metrics费用最大误差元", "数值": REC["d_费用元"].abs().max()},
    {"项目": "10万元复投正确委托总额亿元", "数值": r10["委托总额元"] / 1e8},
    {"项目": "10万元固定本金正确委托总额亿元",
     "数值": A[(A["资金规模元"] == 1e5) & (A["资金规则"] == "fixed")].iloc[0]["委托总额元"] / 1e8},
    {"项目": "报告10倍单位错误格数", "数值": int((ERR["判定"] == "错，放大10倍").sum())},
])
summary.to_csv("out/p15dplus_accounting_summary.csv", index=False, encoding="utf-8-sig")
print("\n=== 汇总 ===", flush=True)
print(summary.to_string(index=False), flush=True)
print("\ndone", flush=True)
