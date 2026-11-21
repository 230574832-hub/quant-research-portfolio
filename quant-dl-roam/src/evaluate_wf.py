"""Evaluate the walk-forward predictions of the deep cross-sectional benchmark.

Three layers, all computed from out/preds_{model}[_tag]_f{fold}.parquet:

  A. Signal quality   : daily IC (Pearson), RankIC (Spearman), ICIR, RankICIR,
                         prec@k, hit rate, t-stats; per model x fold, pooled, by year.
  B. Decomposition    : RankIC of the prediction against the overnight leg
                         (open[t+1]/close[t]-1), the intraday leg
                         (close[t+1]/open[t+1]-1) and the close-to-close return.
                         Answers "what exactly is the model predicting?".
  C. Tradable portfolio: buys at the OPEN of t+1, sells at the close of t+1
                         (1-day) or at the open of t+6 (5-day, non-overlapping
                         rebalance on a fixed 5-day grid), skipping names that are
                         limit-up / suspended at entry, net of ONE_WAY_COST per side.

Cost convention follows quant-lab: 15bp one-way, no market impact.
Long-short books are dollar-neutral (200% gross): portfolio P&L = long leg + short leg.

Several seeds can be evaluated together (--tags "'',_s1,_s2"): every metric is
computed per seed and then averaged, with the across-seed dispersion reported.

Outputs (all under out/):
  metrics_by_model_fold.csv   one row per model x tag x fold x label
  metrics_summary.csv         one row per model x label (seed-averaged)
  metrics_by_year.csv
  decomp_overnight_intraday.csv
  portfolio_daily.csv         per model x tag x fold x date
  portfolio_summary.csv
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

OUT = "out"
QL = "../quant-lab/data"
ONE_WAY_COST = 0.0015      # 15bp one-way
TOPN = 50                  # portfolio breadth
ANN = 252                  # trading days per year
MIN_NAMES = 20             # minimum names in a day for the metrics to count


# ------------------------------------------------------------------ helpers
def _rank(a):
    return pd.Series(a).rank(method="average").to_numpy()


def _corr(a, b):
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12 or len(a) < 3:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def per_day_metrics(g, label_col="y1", ks=(10, 30, 50)):
    """Per-date IC / RankIC / prec@k for one (model, fold) prediction block."""
    recs = []
    for d, gg in g.groupby("date_idx", sort=True):
        p = gg["pred"].to_numpy(dtype=np.float64)
        y = gg[label_col].to_numpy(dtype=np.float64)
        if len(p) < MIN_NAMES:
            continue
        rec = dict(date_idx=int(d), ic=_corr(p, y), rankic=_corr(_rank(p), _rank(y)))
        order = np.argsort(-p)
        for k in ks:
            if len(p) >= k:
                rec[f"prec@{k}"] = float((y[order[:k]] > 0).mean())
        recs.append(rec)
    return pd.DataFrame(recs)


def _agg(daily):
    """Aggregate a per-day frame into the headline numbers."""
    if daily.empty:
        return dict(n_days=0)
    ic = daily["ic"].to_numpy()
    ric = daily["rankic"].to_numpy()
    out = dict(
        n_days=int(len(daily)),
        ic_mean=float(np.nanmean(ic)),
        ic_std=float(np.nanstd(ic, ddof=1)) if len(ic) > 1 else np.nan,
        rankic_mean=float(np.nanmean(ric)),
        rankic_std=float(np.nanstd(ric, ddof=1)) if len(ric) > 1 else np.nan,
    )
    out["icir"] = out["ic_mean"] / out["ic_std"] if out["ic_std"] > 0 else np.nan
    out["rankicir"] = (out["rankic_mean"] / out["rankic_std"]
                        if out["rankic_std"] > 0 else np.nan)
    for k in (10, 30, 50):
        c = f"prec@{k}"
        if c in daily:
            out[c] = float(daily[c].mean())
    out["rankic_t"] = (out["rankic_mean"] / (out["rankic_std"] / np.sqrt(len(ic)))
                        if out["rankic_std"] > 0 else np.nan)
    out["hit_rate"] = float(np.nanmean(ric > 0))
    return out


def load_preds(models, folds, tag):
    blocks = []
    for m in models:
        for fi in folds:
            p = f"{OUT}/preds_{m}{tag}_f{fi}.parquet"
            if not os.path.exists(p):
                print(f"  [warn] missing {p}")
                continue
            df = pd.read_parquet(p)
            df["model"] = m
            df["fold"] = fi
            df["tag"] = tag
            blocks.append(df)
    return blocks


# ------------------------------------------------------- panel-side quantities
def load_panel():
    """Raw open/close/preclose, tradability flags and the tradable legs.

    The date axis is filtered to real trading days with exactly the same rule
    as build_dataset.py (market-wide zero volume = holiday), so that the
    date_idx values stored in the prediction files line up 1:1.
    """
    print("loading panel_wide.pkl ...")
    P = pickle.load(open(f"{QL}/panel_wide.pkl", "rb"))
    O = P["open"].values.astype(np.float64)
    C = P["close"].values.astype(np.float64)
    pre = P["preclose"].values.astype(np.float64)
    board = np.asarray(P["board"])
    V = P["volume"].values.astype(np.float64)
    A = P["adjclose"].values.astype(np.float64)
    del P                       # release the rest of the panel (high/low/amount/...)

    mkt_turnover = np.nansum(np.where(np.isfinite(V), V, 0.0), axis=1)
    keep = np.nonzero(mkt_turnover > 0)[0]
    print(f"  dropping {len(mkt_turnover) - len(keep)} non-trading days")
    O, C, pre, V, A = O[keep], C[keep], pre[keep], V[keep], A[keep]
    del mkt_turnover

    nD, nC = O.shape
    lim = np.where(board == "main", 0.098, 0.198)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = O / pre - 1.0
    lim_up = np.greater_equal(gap, lim)
    lim_up[~np.isfinite(gap)] = False
    traded = V > 0

    # legs indexed by the SIGNAL day t
    overnight = np.full((nD, nC), np.nan)
    intraday = np.full((nD, nC), np.nan)
    c2c = np.full((nD, nC), np.nan)          # buy close(t) -> sell close(t+1)
    c2c5 = np.full((nD, nC), np.nan)         # buy close(t) -> sell close(t+5)
    overnight[:-1] = O[1:] / C[:-1] - 1.0
    intraday[:-1] = C[1:] / O[1:] - 1.0
    c2c[:-1] = C[1:] / C[:-1] - 1.0
    c2c5[:-5] = C[5:] / C[:-5] - 1.0
    for arr in (overnight, intraday, c2c, c2c5):
        arr[~np.isfinite(arr)] = np.nan

    # 5-day open->open: buy at open of t+1, sell at open of t+6
    f_adj = A / np.where(C > 0, C, np.nan)
    aO = O * f_adj
    ret5 = np.full((nD, nC), np.nan)
    ret5[:-6] = aO[6:] / aO[1:-5] - 1.0
    ret5[~np.isfinite(ret5)] = np.nan
    del f_adj, aO

    # tradable at the open of t+1 (what a signal on day t can actually buy)
    entry_ok = np.zeros((nD, nC), dtype=bool)
    entry_ok[:-1] = (~lim_up[1:]) & traded[1:]

    # close-of-day tradability: limit-up at the close -> cannot buy;
    # limit-down at the close -> cannot sell.
    with np.errstate(invalid="ignore", divide="ignore"):
        cgap = C / pre - 1.0
    close_up = np.greater_equal(cgap, lim)
    close_dn = np.less_equal(cgap, -lim)
    close_up[~np.isfinite(cgap)] = False
    close_dn[~np.isfinite(cgap)] = False
    n_close_up = int(close_up.sum())
    bu_ok = (~close_up) & traded          # can buy at the close of t
    se_ok = (~close_dn) & traded          # can sell at the close of t
    # a long entered at the close of t and exited at the close of t+k
    entry_long = np.zeros((nD, nC), dtype=bool)
    entry_short = np.zeros((nD, nC), dtype=bool)
    entry_long5 = np.zeros((nD, nC), dtype=bool)
    entry_short5 = np.zeros((nD, nC), dtype=bool)
    entry_long[:-1] = bu_ok[:-1] & se_ok[1:]
    entry_short[:-1] = se_ok[:-1] & bu_ok[1:]
    entry_long5[:-5] = bu_ok[:-5] & se_ok[5:]
    entry_short5[:-5] = se_ok[:-5] & bu_ok[5:]
    del close_up, close_dn, cgap, bu_ok, se_ok, O, C, pre, V, A, gap

    print(f"  panel {nD} x {nC} trading days; limit-up-at-open stock-days "
          f"{int(lim_up.sum())}; limit-up-at-close stock-days {n_close_up}")
    return dict(overnight=overnight, intraday=intraday, ret5=ret5,
                c2c=c2c, c2c5=c2c5,
                entry_ok=entry_ok, entry_long=entry_long, entry_short=entry_short,
                entry_long5=entry_long5, entry_short5=entry_short5,
                lim_up=lim_up, traded=traded)


# ------------------------------------------------------- layer C: portfolios
def portfolio(df, panel, col_of, n=TOPN, horizon="1d", rebal=5):
    """Portfolio legs for one (model, fold) block.

    horizon 'oc'   : buy open(t+1) -> sell close(t+1)   (intraday leg only)
    horizon 'c2c'  : buy close(t) -> sell close(t+1)    (matches the model target)
    horizon 'c2c5' : buy close(t) -> sell close(t+5)    (non-overlapping, 5-day grid)
    horizon 'oo5'  : buy open(t+1) -> sell open(t+6)    (5-day open->open)
    """
    key = {"oc": "intraday", "c2c": "c2c", "c2c5": "c2c5", "oo5": "ret5"}[horizon]
    recs = []
    cols = col_of.reindex(df["code"].to_numpy()).to_numpy()
    df = df.assign(_col=cols)
    for d, g in df.groupby("date_idx", sort=True):
        d = int(d)
        if horizon in ("c2c5", "oo5") and d % rebal != 0:
            continue
        if len(g) < MIN_NAMES:
            continue
        col = g["_col"].to_numpy()
        p = g["pred"].to_numpy(dtype=np.float64)
        ret = panel[key][d, col]
        fin = np.isfinite(ret)
        if horizon == "oc":
            ok_l = ok_s = fin & panel["entry_ok"][d, col]
        elif horizon == "c2c":
            ok_l = fin & panel["entry_long"][d, col]
            ok_s = fin & panel["entry_short"][d, col]
        elif horizon == "c2c5":
            ok_l = fin & panel["entry_long5"][d, col]
            ok_s = fin & panel["entry_short5"][d, col]
        else:                                     # oo5
            ok_l = ok_s = fin & panel["entry_ok"][d, col]
        if ok_l.sum() < max(n, MIN_NAMES) or ok_s.sum() < max(n, MIN_NAMES):
            continue
        # long leg: top-n among long-eligible names
        idx_l = np.nonzero(ok_l)[0]
        L = float(ret[idx_l[np.argsort(-p[idx_l])[:n]]].mean())
        # short leg: bottom-n among short-eligible names
        idx_s = np.nonzero(ok_s)[0]
        S = float(ret[idx_s[np.argsort(p[idx_s])[:n]]].mean())
        # NULL: identical procedure with random scores -- isolates how much of
        # the long-short spread is just "being short the A-share market"
        # (the universe's own long-short drift) rather than model alpha.
        rng = np.random.default_rng(9000 + d)
        rp = rng.random(len(p))
        nl = float(ret[idx_l[np.argsort(-rp[idx_l])[:n]]].mean())
        ns = float(ret[idx_s[np.argsort(rp[idx_s])[:n]]].mean())
        # equal-weight universe return on the same eligible names
        ew = float(ret[ok_l].mean())
        recs.append(dict(
            date_idx=d,
            long_gross=L, short_gross=-S,
            long_net=L - ONE_WAY_COST, short_net=-S - ONE_WAY_COST,
            ls_net=(L - ONE_WAY_COST) + (-S - ONE_WAY_COST),
            ls_gross=L - S,
            null_ls_net=(nl - ONE_WAY_COST) + (-ns - ONE_WAY_COST),
            null_long_net=nl - ONE_WAY_COST,
            ew_ret=ew,
            n_names=int(ok_l.sum()),
        ))
    return pd.DataFrame(recs)


def sharpe(x, ann=ANN):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.std(x) < 1e-12:
        return np.nan
    return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(ann))


def cumcurve(x):
    return np.cumprod(1.0 + np.asarray(x, dtype=np.float64)) - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ridge,mlp,gru,mgru,sm,sm_nomix,sm_pool")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--tags", default="",
                    help="comma separated output-name suffixes, one per seed")
    ap.add_argument("--out-prefix", default="",
                    help="prefix for the output CSV names (e.g. 'lr_' for ablations)")
    args = ap.parse_args()
    PFX = args.out_prefix
    models = args.models.split(",")
    folds = [int(x) for x in args.folds.split(",")]
    tags = [t.strip() for t in args.tags.split(",")]

    meta = pickle.load(open("data/dataset_meta.pkl", "rb"))
    dates = meta["dates"]
    codes_all = np.array([str(c) for c in meta["codes"]])
    col_of = pd.Series(np.arange(len(codes_all)), index=codes_all)

    panel = load_panel()

    mf_rows, seed_rows, yr_rows, dec_rows, port_all = [], [], [], [], []
    for tag in tags:
        blocks = load_preds(models, folds, tag)
        print(f"tag '{tag}': loaded {len(blocks)} prediction blocks")
        if not blocks:
            continue

        # ---------------------------------------------------- layer A
        for df in blocks:
            for lab in ("y1", "y5"):
                a = _agg(per_day_metrics(df, lab))
                a.update(model=df["model"].iloc[0], fold=int(df["fold"].iloc[0]),
                         tag=tag, label=lab)
                mf_rows.append(a)
        for m in models:
            for lab in ("y1", "y5"):
                daily_all = [per_day_metrics(df, lab) for df in blocks
                             if df["model"].iloc[0] == m]
                if not daily_all:
                    continue
                d = pd.concat(daily_all, ignore_index=True)
                a = _agg(d)
                a.update(model=m, label=lab, tag=tag)
                seed_rows.append(a)

        d = None
        for df in blocks:
            dd = per_day_metrics(df, "y1")
            dd["year"] = [dates[i].year for i in dd["date_idx"]]
            for y, g in dd.groupby("year"):
                a = _agg(g)
                a.update(model=df["model"].iloc[0], fold=int(df["fold"].iloc[0]),
                         tag=tag, year=int(y))
                yr_rows.append(a)

        # ---------------------------------------------------- layer B
        for df in blocks:
            col = col_of.reindex(df["code"].to_numpy()).to_numpy()
            di = df["date_idx"].to_numpy()
            dd = df.assign(overnight=panel["overnight"][di, col],
                           intraday=panel["intraday"][di, col])
            dd = dd[np.isfinite(dd["overnight"]) & np.isfinite(dd["intraday"])]
            o = _agg(per_day_metrics(dd.assign(y1=dd["overnight"]), "y1"))
            i = _agg(per_day_metrics(dd.assign(y1=dd["intraday"]), "y1"))
            c = _agg(per_day_metrics(dd, "y1"))
            dec_rows.append(dict(model=df["model"].iloc[0], fold=int(df["fold"].iloc[0]),
                                 tag=tag, n_days=o["n_days"],
                                 rankic_overnight=o["rankic_mean"],
                                 rankic_intraday=i["rankic_mean"],
                                 rankic_close2close=c["rankic_mean"],
                                 ic_overnight=o["ic_mean"], ic_intraday=i["ic_mean"],
                                 ic_close2close=c["ic_mean"]))

        # ---------------------------------------------------- layer C
        for df in blocks:
            for hz in ("oc", "c2c", "c2c5", "oo5"):
                p = portfolio(df, panel, col_of, TOPN, hz)
                if p.empty:
                    continue
                p["model"] = df["model"].iloc[0]
                p["fold"] = int(df["fold"].iloc[0])
                p["tag"] = tag
                p["horizon"] = hz
                port_all.append(p)

    bymf = pd.DataFrame(mf_rows)
    bymf.to_csv(f"{OUT}/{PFX}metrics_by_model_fold.csv", index=False)
    print(f"wrote metrics_by_model_fold.csv ({len(bymf)} rows)")

    # seed-averaged summary
    sdf = pd.DataFrame(seed_rows)
    keep = ["n_days", "ic_mean", "ic_std", "icir", "rankic_mean", "rankic_std",
            "rankicir", "rankic_t", "hit_rate", "prec@10", "prec@30", "prec@50"]
    keep = [c for c in keep if c in sdf.columns]
    g = sdf.groupby(["model", "label"])[keep]
    mean = g.mean()
    std = g.std(ddof=1).add_suffix("_seedstd")
    summ = mean.join(std).reset_index()
    summ["n_seeds"] = sdf.groupby(["model", "label"]).size().to_numpy()
    # fold dispersion (on the per-fold table)
    fb = bymf.groupby(["model", "label", "tag"])["rankic_mean"].mean().reset_index()
    summ = summ.merge(
        fb.groupby(["model", "label"])["rankic_mean"].agg(["mean", "std"])
        .reset_index().rename(columns={"mean": "fold_rankic_mean",
                                       "std": "fold_rankic_std"}),
        on=["model", "label"], how="left")
    summ.to_csv(f"{OUT}/{PFX}metrics_summary.csv", index=False)
    print("wrote metrics_summary.csv")

    pd.DataFrame(yr_rows).to_csv(f"{OUT}/{PFX}metrics_by_year.csv", index=False)
    print("wrote metrics_by_year.csv")
    pd.DataFrame(dec_rows).to_csv(f"{OUT}/{PFX}decomp_overnight_intraday.csv", index=False)
    print("wrote decomp_overnight_intraday.csv")

    port = pd.concat(port_all, ignore_index=True)
    port.to_csv(f"{OUT}/{PFX}portfolio_daily.csv", index=False)
    print(f"wrote portfolio_daily.csv ({len(port)} rows)")

    ps = []
    for (m, fi, hz, tg), g in port.groupby(["model", "fold", "horizon", "tag"]):
        ann = ANN if hz in ("oc", "c2c") else ANN / 5.0
        ps.append(dict(model=m, fold=fi, horizon=hz, tag=tg, n_days=len(g),
                       long_gross_sr=sharpe(g["long_gross"], ann),
                       long_net_sr=sharpe(g["long_net"], ann),
                       ls_gross_sr=sharpe(g["ls_gross"], ann),
                       ls_net_sr=sharpe(g["ls_net"], ann),
                       null_ls_net_sr=sharpe(g["null_ls_net"], ann),
                       null_long_net_sr=sharpe(g["null_long_net"], ann),
                       ew_ret_mean=float(g["ew_ret"].mean()),
                       ew_ret_sr=sharpe(g["ew_ret"], ann),
                       long_net_mean=float(g["long_net"].mean()),
                       ls_net_mean=float(g["ls_net"].mean()),
                       long_net_cum=float(cumcurve(g["long_net"])[-1]),
                       ls_net_cum=float(cumcurve(g["ls_net"])[-1])))
    psum = pd.DataFrame(ps)
    psum.to_csv(f"{OUT}/{PFX}portfolio_summary.csv", index=False)
    print("wrote portfolio_summary.csv")


if __name__ == "__main__":
    main()
