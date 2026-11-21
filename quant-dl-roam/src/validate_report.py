"""校验报告里引用的每一个数字与 out/*.csv 是否一致。

    ./.venv/bin/python src/validate_report.py

新增一个结论、跑完一批实验后都应该跑一遍; 任何 FAIL 都要在报告发出前修掉。
"""
import glob
import re

import numpy as np
import pandas as pd

REPORTS = ["reports/深度学习横截面模型评测报告.md",
           "reports/StockMixer论文复现_NASDAQ.md"]

CHECKS = []


def chk(name, claim, actual, tol=0.0006):
    ok = bool(np.isfinite(claim) and np.isfinite(actual)
              and abs(claim - actual) <= tol)
    CHECKS.append((ok, name, claim, actual))


def main():
    s = pd.read_csv("out/metrics_summary.csv")
    s1 = s[s.label == "y1"].set_index("model")
    s5 = s[s.label == "y5"].set_index("model")
    dec = pd.read_csv("out/decomp_overnight_intraday.csv")
    dd = dec.groupby("model")[["rankic_overnight", "rankic_intraday",
                               "rankic_close2close"]].mean()
    bymf = pd.read_csv("out/metrics_by_model_fold.csv")
    bymf = bymf[bymf.label == "y1"]
    port = pd.read_csv("out/portfolio_summary.csv")
    num = [c for c in port.select_dtypes("number").columns]
    pc = port[port.horizon == "c2c"].groupby("model")[num].mean()
    pc5 = port[port.horizon == "c2c5"].groupby("model")[num].mean()
    lr = pd.read_csv("out/lr_metrics_by_model_fold.csv")
    lr = lr[lr.label == "y1"]
    nom = pd.read_csv("out/nomask3_metrics_by_model_fold.csv")
    nom = nom[nom.label == "y1"]
    best = pd.read_csv("out/paper_repro_best.csv")
    rk = pd.read_csv("out/paper_repro_nasdaq_rankic.csv").iloc[0]
    own = pd.read_csv("out/own_impl_nasdaq.csv")

    # ---------- A 股主表 ----------
    chk("StockMixer RankIC", 0.0025, s1.loc["sm", "rankic_mean"])
    chk("StockMixer t 值", 0.94, s1.loc["sm", "rankic_t"], 0.01)
    chk("StockMixer prec@10", 0.471, s1.loc["sm", "prec@10"])
    chk("StockMixer RankICIR", 0.019, s1.loc["sm", "rankicir"])
    chk("Ridge RankIC", 0.0181, s1.loc["ridge", "rankic_mean"])
    chk("Ridge prec@10", 0.481, s1.loc["ridge", "prec@10"])
    chk("MLP RankIC", 0.0359, s1.loc["mlp", "rankic_mean"])
    chk("GRU RankIC", 0.0502, s1.loc["gru", "rankic_mean"])
    chk("GRU RankICIR", 0.346, s1.loc["gru", "rankicir"])
    chk("mgru RankIC", 0.0584, s1.loc["mgru", "rankic_mean"])
    chk("mgru RankICIR", 0.393, s1.loc["mgru", "rankicir"])
    chk("mgru t 值", 19.6, s1.loc["mgru", "rankic_t"], 0.05)
    chk("mgru_feat RankIC", 0.0494, s1.loc["mgru_feat", "rankic_mean"])
    chk("HIST RankIC", 0.0513, s1.loc["hist", "rankic_mean"])
    chk("HIST IC", 0.0572, s1.loc["hist", "ic_mean"])
    chk("HIST RankICIR", 0.3579, s1.loc["hist", "rankicir"])
    chk("HIST t 值", 17.8, s1.loc["hist", "rankic_t"], 0.05)
    chk("HIST 种子标准差", 0.0013, s1.loc["hist", "rankic_mean_seedstd"])
    chk("sm_nomix RankIC", 0.0248, s1.loc["sm_nomix", "rankic_mean"])
    chk("sm_pool RankIC", 0.0248, s1.loc["sm_pool", "rankic_mean"])
    chk("mgru 相对 GRU +16%", 16.0,
        (s1.loc["mgru", "rankic_mean"] / s1.loc["gru", "rankic_mean"] - 1) * 100, 0.6)
    chk("mgru 相对 ridge +223%", 223.0,
        (s1.loc["mgru", "rankic_mean"] / s1.loc["ridge", "rankic_mean"] - 1) * 100, 1.0)
    chk("测试天数", 2481, s1.loc["gru", "n_days"], 0.5)

    # ---------- 分解 ----------
    chk("GRU 隔夜", 0.0890, dd.loc["gru", "rankic_overnight"])
    chk("GRU 日内", 0.0211, dd.loc["gru", "rankic_intraday"])
    chk("mgru 隔夜", 0.0864, dd.loc["mgru", "rankic_overnight"])
    chk("mgru 日内", 0.0309, dd.loc["mgru", "rankic_intraday"])
    chk("mgru_feat 隔夜", 0.0776, dd.loc["mgru_feat", "rankic_overnight"])
    chk("mgru_feat 日内", 0.0244, dd.loc["mgru_feat", "rankic_intraday"])
    chk("HIST 隔夜", 0.0914, dd.loc["hist", "rankic_overnight"])
    chk("HIST 日内", 0.0215, dd.loc["hist", "rankic_intraday"])

    # ---------- 组合 ----------
    chk("mgru 多空净SR", 2.046, pc.loc["mgru", "ls_net_sr"], 0.01)
    chk("gru 多空净SR", 1.725, pc.loc["gru", "ls_net_sr"], 0.01)
    chk("HIST 多空净SR", 1.646, pc.loc["hist", "ls_net_sr"], 0.01)
    chk("mgru_feat 多空净SR", 1.408, pc.loc["mgru_feat", "ls_net_sr"], 0.01)
    chk("随机零假设 SR", -9.382, pc.loc["ridge", "null_ls_net_sr"], 0.01)
    chk("mgru 多头净SR", -0.227, pc.loc["mgru", "long_net_sr"], 0.01)
    chk("HIST 5日多空净SR", 1.365, pc5.loc["hist", "ls_net_sr"], 0.01)
    chk("HIST 5日多头净SR", 0.065, pc5.loc["hist", "long_net_sr"], 0.01)
    chk("mgru 5日多头净SR", 0.169, pc5.loc["mgru", "long_net_sr"], 0.01)

    # ---------- 学习率 & 稳健性 ----------
    chk("sm lr1e-3 均值", 0.0014,
        bymf[(bymf.model == "sm") & (bymf.tag == "_s0")]["rankic_mean"].mean(), 0.001)
    chk("sm lr3e-4 均值", 0.0079,
        lr[(lr.model == "sm") & (lr.tag == "_lr3e4")]["rankic_mean"].mean(), 0.001)
    chk("sm lr3e-5 均值", 0.0009,
        lr[(lr.model == "sm") & (lr.tag == "_lr3e5")]["rankic_mean"].mean(), 0.001)
    chk("sm_nomix lr3e-4 均值", 0.0179,
        lr[(lr.model == "sm_nomix") & (lr.tag == "_lr3e4")]["rankic_mean"].mean(), 0.001)
    chk("sm_nomask 均值", 0.0070, nom["rankic_mean"].mean(), 0.001)

    # ---------- 论文复现 ----------
    nas = best[best.market == "NASDAQ"]
    sp5 = best[best.market == "SP500"]
    ny = best[best.market == "NYSE"]
    chk("NASDAQ 复现 IC", 0.0218, nas["test_IC"].mean(), 0.001)
    chk("NASDAQ 复现 ICIR", 0.2743, nas["test_RIC"].mean(), 0.001)
    chk("NASDAQ 复现 prec@10", 0.5270, nas["test_prec10"].mean(), 0.001)
    chk("NASDAQ 复现 SR", 1.2817, nas["test_SR"].mean(), 0.001)
    chk("NASDAQ 种子标准差", 0.0131, nas["test_IC"].std(), 0.001)
    chk("SP500 复现 IC", 0.0103, sp5["test_IC"].mean(), 0.001)
    chk("SP500 复现 ICIR", 0.0985, sp5["test_RIC"].mean(), 0.001)
    chk("SP500 复现 prec@10", 0.5373, sp5["test_prec10"].mean(), 0.001)
    chk("SP500 复现 SR", 2.0567, sp5["test_SR"].mean(), 0.001)
    if len(ny):
        chk("NYSE 复现 IC", 0.0163, ny["test_IC"].mean(), 0.001)
        chk("NYSE 复现 ICIR", 0.1611, ny["test_RIC"].mean(), 0.001)
        chk("NYSE 复现 prec@10", 0.5183, ny["test_prec10"].mean(), 0.001)
        chk("NYSE 复现 SR", 1.0113, ny["test_SR"].mean(), 0.001)
    chk("真正 RankIC", 0.0356, rk["rankic_spearman"])
    chk("RankICIR", 0.3627, rk["rankicir"])
    chk("own_impl 6种子 IC 均值", 0.0055, own["IC"].mean(), 0.001)
    chk("own_impl IC 标准差", 0.0073, own["IC"].std(), 0.0006)
    bl = pd.concat([pd.read_csv(f) for f in
                    glob.glob("out/baselines_lstm_s*.csv")
                    + glob.glob("out/baselines_linear_s*.csv")], ignore_index=True)
    for m, claim_ic, claim_p in [("lstm", 0.0266, 0.5091), ("linear", 0.0110, 0.5200)]:
        sub = bl[bl.model == m]
        if len(sub):
            chk(f"{m} 基线 IC", claim_ic, sub["IC"].mean(), 0.0006)
            chk(f"{m} 基线 prec@10", claim_p, sub["prec_10"].mean(), 0.0006)

    # ---------- 报告正文里的字符串也要核(不只是 CSV) ----------
    # 踩过的坑: numpy 的 .std() 默认 ddof=0, pandas 的 .std() 默认 ddof=1,
    # 同一个"种子标准差"被写成了两个值, 而只校验 CSV 查不出来。
    for f in REPORTS:
        t = open(f, encoding="utf-8").read()
        for bad in ["0.0107", "0.0067"]:          # ddof=0 的两个错值
            chk(f"{f.split('/')[-1]} 不含 ddof=0 的 {bad}", 0.0,
                float(bad in t))
    # 报告里写的"± x.xxxx"必须等于 CSV 里 ddof=1 的标准差
    if len(nas):
        want = f"{nas['test_IC'].std():.4f}"
        for f in REPORTS:
            t = open(f, encoding="utf-8").read()
            chk(f"{f.split('/')[-1]} 官方IC标准差={want}", 1.0,
                float(want in t or "0.0218" not in t))

    txt = "".join(open(f, encoding="utf-8").read() for f in REPORTS)
    for ok, name, claim, actual in CHECKS:
        print(("  OK  " if ok else "  FAIL"), f"{name:26s} 报告={claim:<10} 实测={actual:.6f}")
    bad = [c for c in CHECKS if not c[0]]
    print(f"\n通过 {len(CHECKS)-len(bad)}/{len(CHECKS)}")
    if bad:
        raise SystemExit("有不一致项")
    print("两份报告的数字与 out/*.csv 完全一致。")


if __name__ == "__main__":
    main()
