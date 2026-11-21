"""Walk-forward training + prediction for the deep cross-sectional benchmark.

Protocol
  * fold-fixed universe: top-500 by mean log market cap over the last 500
    trading days BEFORE the fold's test block (only past information);
  * train / validation / test blocks with H+EMBARGO day purge gaps (from build_dataset);
  * validation = last 15% of the training history, used only for early stopping;
  * test = untouched block; predictions saved for common evaluation.

Models: ridge, mlp, gru, mgru, sm, sm_nomix, sm_pool  (see src/models.py)

Outputs
  out/preds_{model}_{fold}.parquet    date_idx, code_idx, pred, y1, y5, mask
  out/hist_{model}_{fold}.csv         per-epoch loss / val IC
  out/universes.csv                   fold universe membership

Bug fixes applied 2026-11-15 (script had never been executed end-to-end):
  F1 npz member access is lazy: every `d["y1"]` re-decompresses the whole array
     (0.105 s).  The test-prediction row loop hit it ~272k times -> multi-hour
     stall.  All members are now materialised once in load_data().
  F2 Fz is 54.6% NaN (days outside the investable base).  NaNs were copied
     straight into X, propagated through the network and then poisoned the
     loss as NaN*0 -> every batch loss was NaN.  X is now nan_to_num'd at
     build time (masked entries are excluded from the loss anyway).
  F3 labels are cross-sectionally z-scored per day (MASTER convention, lit
     spec S3 / deviation #11).  With raw returns the MSE term is dominated by
     the market direction, which no cross-sectional feature can predict, so
     the ranking signal was diluted.  Rank IC is invariant to this transform.
  F4 `float(loss)` on a grad-tracking tensor -> warning; use .detach().
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models import build_model, sm_loss

OUT = "out"
QL = "../quant-lab/data"
T = 16
H = 1
EMBARGO = 5
MIN_UNIV_DAYS = 0.6
N_UNIV = 500
LOOKBACK_CAP = 500
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
CS_CLIP = 5.0          # clip of the cross-sectional z-scored label


def set_seed(s=12345678):
    np.random.seed(s)
    torch.manual_seed(s)


def load_data():
    """Materialise every npz member once (see bug note F1).

    Only the members the trainer actually touches are kept; y5 / ytrade5 /
    blocked_buy live in the prediction parquet instead, so materialising them
    here would waste ~190 MB per process for nothing.
    """
    z = np.load("data/features.npz", allow_pickle=True)
    d = {k: z[k] for k in ("Fz", "y1", "y5", "MS", "MC", "base", "dates", "codes")}
    z.close()
    meta = pickle.load(open("data/dataset_meta.pkl", "rb"))
    dates = pd.DatetimeIndex([pd.Timestamp(x) for x in d["dates"]])
    codes = np.array([str(c) for c in d["codes"]])
    return d, meta, dates, codes


def cs_zscore(y, mask, clip=CS_CLIP):
    """Cross-sectional z-score of the label within each date (same-day only).

    y: (D, N) float with NaN outside mask; mask: (D, N) bool.
    Returns (D, N) float32, 0 where the day/stock is not usable.
    """
    m = mask.astype(np.float32)
    cnt = m.sum(1)
    yy = np.where(mask, np.nan_to_num(y, nan=0.0), 0.0).astype(np.float64)
    mu = yy.sum(1) / np.maximum(cnt, 1.0)
    dev = np.where(mask, y.astype(np.float64) - mu[:, None], 0.0)
    sd = np.sqrt((dev * dev).sum(1) / np.maximum(cnt, 1.0))
    ok = cnt >= 20
    z = np.zeros(y.shape, dtype=np.float32)
    s = np.where(sd > 1e-8, sd, 1.0)
    z[ok] = (dev[ok] / s[ok][:, None]).astype(np.float32)
    return np.clip(z, -clip, clip)


def build_fold_universe(d, meta, fi):
    """Top-N codes by mean log mktcap over the trailing 500 trading days
    ending right before the fold's purge gap.  Uses only pre-test data."""
    f = meta["folds"][fi]
    i_end = f["test_date_lo"] - EMBARGO - H      # last day usable as 'past' for test
    lo = max(0, i_end - LOOKBACK_CAP)
    MC = d["MC"][lo:i_end]                       # (L, nC)
    base = d["base"][lo:i_end]
    with np.errstate(invalid="ignore"):
        logmc = np.where(base & np.isfinite(MC) & (MC > 0), np.log(MC), np.nan)
    imp = np.nanmean(np.where(base, 1.0, 0.0), axis=0)       # presence share
    meanmc = np.nanmean(logmc, axis=0)
    ok = (imp >= MIN_UNIV_DAYS) & np.isfinite(meanmc)
    rank = np.full(len(meanmc), np.inf)
    rank[ok] = -meanmc[ok]
    cols = np.argsort(rank)[:N_UNIV]
    return np.sort(cols), i_end


def load_industries():
    """证监会行业标签 (panel_wide['industry'], index=code) —— HIST 预定义概念用。"""
    if not os.path.exists(f"{QL}/panel_wide.pkl"):
        return None
    P = pickle.load(open(f"{QL}/panel_wide.pkl", "rb"))
    ind = P["industry"]
    out = {str(k): str(v) for k, v in ind.items()}
    del P, ind
    return out


def build_concept_matrix(industries, codes, cols):
    """每折的 二值 股票×概念 成员矩阵 (N, M)。

    概念 = 证监会行业 (本地面板共 83 类); 成员关系为静态(非时变)。
    """
    labels = [industries.get(str(codes[c]), "未知") for c in cols]
    uniq = sorted(set(labels))
    idx = {l: i for i, l in enumerate(uniq)}
    m = np.zeros((len(cols), len(uniq)), dtype=np.float32)
    for i, l in enumerate(labels):
        m[i, idx[l]] = 1.0
    return m, uniq


def build_day_tensors(Fz, y1, cols, date_hi):
    """X: (date_hi, N, T, C) float16 with window ending at each date; mask (date_hi, N).

    NaN features (stock not in the investable base that day) are replaced by 0
    so they cannot propagate through the network (bug note F2).  The mask marks
    exactly those entries as invalid, so they never enter the loss.
    """
    N = len(cols)
    C = Fz.shape[2]
    X = np.zeros((date_hi, N, T, C), dtype=np.float16)
    ok = np.zeros((date_hi, N), dtype=bool)
    sub = Fz[:, cols, :]                           # (nD, N, C) float16 copy
    for t in range(T - 1, date_hi):
        w = sub[t - T + 1:t + 1]                    # (T, N, C)
        X[t] = np.nan_to_num(w.transpose(1, 0, 2)).astype(np.float16)
        ok[t] = np.isfinite(w).all(axis=(0, 2))
    mask = ok & np.isfinite(y1[:date_hi, cols])
    del sub
    return X, mask


def daily_ic(pred, y, mask):
    """Spearman rank IC per day, averaged. pred/y: (B,N) numpy, mask (B,N) bool."""
    out = []
    for b in range(pred.shape[0]):
        m = mask[b]
        if m.sum() < 20:
            continue
        p, q = pred[b][m], y[b][m]
        if np.std(p) < 1e-12 or np.std(q) < 1e-12:
            out.append(0.0)
            continue
        pr = np.argsort(np.argsort(p))
        qr = np.argsort(np.argsort(q))
        out.append(float(np.corrcoef(pr, qr)[0, 1]))
    return float(np.mean(out)) if out else np.nan


def make_batches(days, bs, rng=None, shuffle=True):
    days = np.asarray(days)
    if shuffle:
        days = rng.permutation(days)
    for i in range(0, len(days), bs):
        yield days[i:i + bs]


def run_fold_model(model_name, fi, d, meta, dates, codes, X, mask, mkt, cols_global,
                   ylab, max_epochs=None, seed=0, suffix="", lr=1e-3, extra=None):
    f = meta["folds"][fi]
    rng = np.random.default_rng(1234 + fi + 1000 * seed)
    tr_days = np.arange(0, f["train_date_hi"] + 1)
    va_days = np.arange(f["val_date_lo"], f["val_date_hi"] + 1)
    te_days = np.arange(f["test_date_lo"], f["test_date_hi"])
    N = X.shape[1]
    C = X.shape[3]
    mdim = mkt.shape[1]
    model = build_model(model_name, N, T, C, mdim, extra=extra).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bs = 16 if model_name.startswith("sm") or model_name in ("gru", "mgru", "hist") else 32
    epochs = 40 if model_name.startswith("sm") or model_name == "hist" else 35
    if max_epochs:
        epochs = max_epochs
    patience = 8
    best_ic, best_state, bad = -9.9, None, 0
    hist = []

    def forward_batch(day_idx):
        x = torch.from_numpy(X[day_idx].astype(np.float32)).to(DEV)
        y = torch.from_numpy(ylab[day_idx].astype(np.float32)).to(DEV)
        m = torch.from_numpy(mask[day_idx]).to(DEV)
        mk = torch.from_numpy(mkt[day_idx]).to(DEV)
        return x, y, m, mk

    @torch.no_grad()
    def evaluate(days):
        model.eval()
        ps, ys, ms = [], [], []
        for b in make_batches(days, 64, shuffle=False):
            x, y, m, mk = forward_batch(b)
            out = model(x, m, mk) if model_name in ("mgru", "mgru_feat") else model(x, m)
            ps.append(out.cpu().numpy())
            ys.append(y.cpu().numpy())
            ms.append(m.cpu().numpy())
        p, y_, m_ = np.concatenate(ps), np.concatenate(ys), np.concatenate(ms)
        return daily_ic(p, y_, m_)

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for b in make_batches(tr_days, bs, rng, shuffle=True):
            x, y, m, mk = forward_batch(b)
            out = model(x, m, mk) if model_name in ("mgru", "mgru_feat") else model(x, m)
            loss = sm_loss(out, y, m)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        vic = evaluate(va_days)
        hist.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_ic=vic))
        mark = ""
        if np.isfinite(vic) and (vic > best_ic):
            best_ic = vic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            mark = " *"
        else:
            bad += 1
        print(f"  [{model_name} f{fi}] ep{ep:02d} loss {tot/max(nb,1):.5f} "
              f"valIC {vic:+.4f} ({time.time()-t0:.0f}s){mark}", flush=True)
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    ps, ds, js = [], [], []
    with torch.no_grad():
        for b in make_batches(te_days, 64, shuffle=False):
            x, y, m, mk = forward_batch(b)
            out = model(x, m, mk) if model_name in ("mgru", "mgru_feat") else model(x, m)
            p = out.cpu().numpy()
            mm = m.cpu().numpy()
            for k, day in enumerate(b):
                idx = np.nonzero(mm[k])[0]
                if len(idx):
                    ps.append(p[k, idx])
                    ds.append(np.full(len(idx), int(day), dtype=np.int32))
                    js.append(idx.astype(np.int32))
    pred = np.concatenate(ps).astype(np.float32)
    day_idx = np.concatenate(ds)
    univ_j = np.concatenate(js)
    df = pd.DataFrame({"date_idx": day_idx, "univ_j": univ_j, "pred": pred})
    df["code"] = codes[cols_global[df["univ_j"].values]]
    df["fold"] = fi
    df["model"] = model_name
    df["y1"] = d["y1"][day_idx, cols_global[univ_j]].astype(np.float32)
    df["y5"] = d["y5"][day_idx, cols_global[univ_j]].astype(np.float32)
    pd.DataFrame(hist).to_csv(f"{OUT}/hist_{model_name}{suffix}_f{fi}.csv", index=False)
    df.to_parquet(f"{OUT}/preds_{model_name}{suffix}_f{fi}.parquet", index=False)
    print(f"  [{model_name} f{fi}] done best_val_ic={best_ic:+.4f} test_rows={len(df)} "
          f"params={npar}", flush=True)
    return best_ic


def run_ridge(fi, d, meta, dates, codes, X, mask, cols_global, ylab, suffix=""):
    """Ridge on the flattened window; same fold universe / folds / masks."""
    from sklearn.linear_model import Ridge
    f = meta["folds"][fi]
    tr_days = np.arange(0, f["train_date_hi"] + 1)
    te_days = np.arange(f["test_date_lo"], f["test_date_hi"])
    N, T_, C = X.shape[1], X.shape[2], X.shape[3]
    Xf = X.reshape(X.shape[0], N, T_ * C).astype(np.float32)

    def gather(days):
        xs, ys, ms, js = [], [], [], []
        for day in days:
            m = mask[day]
            if m.sum() < 20:
                continue
            j = np.nonzero(m)[0]
            xs.append(Xf[day][j])
            ys.append(ylab[day][j])
            ms.append(np.full(len(j), day))
            js.append(j)
        return (np.concatenate(xs), np.concatenate(ys),
                np.concatenate(ms), np.concatenate(js))

    Xtr, ytr, _, _ = gather(tr_days)
    model = Ridge(alpha=10.0)
    model.fit(Xtr, ytr)
    Xte, yte, dte, jte = gather(te_days)
    p = model.predict(Xte).astype(np.float32)
    df = pd.DataFrame(dict(date_idx=dte.astype(int), univ_j=jte.astype(int),
                           pred=p))
    df["code"] = codes[cols_global[df["univ_j"].values]]
    df["fold"] = fi
    df["model"] = "ridge"
    df["y1"] = d["y1"][df["date_idx"].values, cols_global[df["univ_j"].values]]
    df["y5"] = d["y5"][df["date_idx"].values, cols_global[df["univ_j"].values]]
    df.to_parquet(f"{OUT}/preds_ridge{suffix}_f{fi}.parquet", index=False)
    print(f"  [ridge f{fi}] done test_rows={len(df)}", flush=True)
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ridge,mlp,gru,mgru,sm,sm_nomix")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--label-cs", type=int, default=1,
                    help="1 = cross-sectional z-scored label (MASTER convention)")
    ap.add_argument("--resume", action="store_true",
                    help="skip model/fold combos whose preds parquet already exists")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed offset (run the grid with several seeds for variance)")
    ap.add_argument("--out-suffix", default="",
                    help="suffix inserted into the output file names, e.g. '_s1'")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Adam learning rate (StockMixer paper: 1e-3)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    set_seed(12345678 + args.seed)
    torch.manual_seed(12345678 + args.seed)
    np.random.seed(12345678 + args.seed)
    d, meta, dates, codes = load_data()
    mkt_all = d["MS"].astype(np.float32)
    print("device:", DEV, "| nD", len(dates), "| models", args.models,
          "| label_cs", args.label_cs, flush=True)
    uni_rows = []
    industries = None
    for fi in [int(x) for x in args.folds.split(",")]:
        cols_global, i_end = build_fold_universe(d, meta, fi)
        f = meta["folds"][fi]
        print(f"fold {fi} {f['name']}: universe N={len(cols_global)} "
              f"test {f['test_range']} ntrain={f['n_train']}", flush=True)
        date_hi = f["test_date_hi"]
        X, mask = build_day_tensors(d["Fz"], d["y1"], cols_global, date_hi)
        mkt = mkt_all[:date_hi]
        extra = None
        if "hist" in args.models.split(","):
            if industries is None:
                industries = load_industries()
            mem, uniq = build_concept_matrix(industries, codes, cols_global)
            extra = dict(mem=mem)
            print(f"  fold{fi} HIST 概念矩阵: {mem.shape[0]} 只 x {mem.shape[1]} 个概念")
        y1f = d["y1"][:date_hi, cols_global]
        ylab = cs_zscore(y1f, mask) if args.label_cs else \
            np.where(mask, np.nan_to_num(y1f, nan=0.0), 0.0).astype(np.float32)
        for name in args.models.split(","):
            outp = f"{OUT}/preds_{name}{args.out_suffix}_f{fi}.parquet"
            if args.resume and os.path.exists(outp):
                print(f"  skip {name} f{fi} (exists)", flush=True)
                continue
            t0 = time.time()
            if name == "ridge":
                run_ridge(fi, d, meta, dates, codes, X, mask, cols_global, ylab,
                          args.out_suffix)
            else:
                run_fold_model(name, fi, d, meta, dates, codes, X, mask, mkt,
                               cols_global, ylab, args.max_epochs, args.seed,
                               args.out_suffix, args.lr, extra)
            print(f"  fold{fi} {name} wall {time.time()-t0:.0f}s", flush=True)
        for j, c in enumerate(cols_global):
            uni_rows.append(dict(fold=fi, j=j, code=codes[c]))
        del X, mask, mkt, ylab
    if uni_rows:
        pd.DataFrame(uni_rows).to_csv(f"{OUT}/universes.csv", index=False)


if __name__ == "__main__":
    main()
