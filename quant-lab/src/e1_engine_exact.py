"""E1-b: exact mirror of the frozen V5 executor (independent code, same algorithm).

Written by reading CapacityExecutor line-by-line, not by importing it. Purpose: decide whether
the remaining A-vs-B gap is algorithmic (now mirrored) or a real implementation difference.

Mirrored semantics
  * positions are DOLLAR values marked at the adjusted open, return clipped to [-0.99, 10]
  * daily loop; pending orders retried only on non-rebalance days
  * rebalance day: cancel pending buys, sell non-target, trim target above eq/K,
    top-up target below eq/K with cash scaling, blocked sells persist as pending
  * flat 15bp per side (A's four fee components total 15bp)
  * no capacity / no impact (unlimited calibration)
Switches retained for E1-c factorial use: score, price mode, ST disposal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COST = 15e-4
CLIP_HI = 10.0
FILL_TOL = 1e-6
MIN_UNIVERSE = 100


def _topk(score_row, mask_row, k):
    idx = np.where(mask_row)[0]
    if len(idx) == 0:
        return np.array([], dtype=int)
    s = score_row[idx]
    ok = np.isfinite(s)
    idx, s = idx[ok], s[ok]
    if len(idx) <= k:
        return np.sort(idx)
    # A uses argsort(-score)[:K]; replicate order (descending score)
    return idx[np.argsort(-s)[:k]]


def run(env, score, pairs, k=50, cost=COST, price="open", st="hold",
        capital=1e5, track_pnl=False, return_pos=True):
    P = env.ao if price == "open" else env.adjc
    suspended, up_open, dn_open, can_sell, can_buy = env.tradability()
    is_st = env.is_st
    N, M = env.N, env.M
    pos = np.zeros(M, dtype=np.float64)
    cash = float(capital)
    pending = {}
    stale = {}
    frozen = set()
    trades = []
    equity_by_exec = {}
    turn_by_exec = {}
    nh_by_exec = {}
    pos_daily = {}
    pnl_by_code = {} if track_pnl else None
    st_forced = set()
    st_forced_orders = 0
    st_forced_blocked = 0
    pnl_year_code = {} if track_pnl else None
    exec_map = {(i0 + 1): (i0, a) for a, (i0, i1) in enumerate(pairs) if i0 + 1 < N and i1 + 1 < N}
    exec_days = sorted(exec_map)
    if not exec_days:
        return None
    start_e, end_e = exec_days[0], exec_days[-1]

    def apply_fill(e, c, side, v):
        nonlocal cash
        fee = v * cost
        if side == "S":
            pos[c] -= v
            cash += v - fee
        else:
            pos[c] += v
            cash -= v + fee
        return fee

    def attempt(e, c, side, desired, used, kind):
        nonlocal cash
        if desired <= 1e-9:
            return 0.0, 0.0
        v = desired
        if side == "B":
            for _ in range(2):
                if v <= 1e-9:
                    break
                lim = max(cash, 0.0) / (1.0 + cost)
                if v > lim:
                    v = lim
                else:
                    break
        if v <= 1e-9:
            trades.append((e, c, side, kind, "blocked", desired, 0.0, np.nan, 0.0))
            return 0.0, 0.0
        px = P[e, c]
        fee = apply_fill(e, c, side, v)
        trades.append((e, c, side, kind, "filled" if v >= desired * (1 - FILL_TOL) else "partial",
                       desired, v, px, fee))
        return v, fee

    turn = 0.0
    for e in range(start_e, end_e + 1):
        # (0) ST-exclude counterfactual: exit any held ST name at the first sellable day
        if st == "exclude":
            for c in np.where(pos > 1e-9)[0]:
                if env.status[e, c] != 2:
                    continue
                if not can_sell[e, c]:
                    st_forced_blocked += 1
                    continue
                if pos[c] <= 1e-9:
                    continue
                v, _ = attempt(e, int(c), "S", pos[c], None, "st-forced-exit")
                turn += v
                if v > 0:
                    st_forced.add(int(c)); st_forced_orders += 1
        # (1) mark to market, open-to-open, clipped
        if e > start_e:
            cols = np.where(pos > 1e-9)[0]
            if len(cols):
                prev = pos[cols].copy()
                with np.errstate(all="ignore"):
                    r = P[e, cols] / P[e - 1, cols] - 1.0
                good = np.isfinite(r) & np.isfinite(P[e, cols]) & (P[e, cols] > 0)
                r = np.clip(np.where(good, r, 0.0), -0.99, CLIP_HI)
                pos[cols] *= (1.0 + r)
                if track_pnl:
                    for j, c in enumerate(cols):
                        d = prev[j] * r[j]
                        if abs(d) > 0:
                            pnl_by_code[int(c)] = pnl_by_code.get(int(c), 0.0) + d
                            key = (int(c), env.dates[e].year)
                            pnl_year_code[key] = pnl_year_code.get(key, 0.0) + d
                for c in cols[good]:
                    stale[c] = 0
                    frozen.discard(int(c))
                for c in cols[~good]:
                    stale[c] = stale.get(c, 0) + 1
                    if stale[c] >= 20:
                        frozen.add(int(c))
        # (2) non-rebalance day: retry pending
        if e not in exec_map:
            for c in list(pending):
                p = pending[c]
                if p["side"] == "S":
                    if pos[c] <= 1e-9:
                        pending.pop(c, None); continue
                    if not can_sell[e, c]:
                        continue
                    want = pos[c] if p["target"] <= 0 else max(pos[c] - p["target"], 0.0)
                    if want <= 1e-9:
                        pending.pop(c, None); continue
                    v, _ = attempt(e, c, "S", want, None, "retry-sell")
                    turn += v
                    if v >= want * (1 - FILL_TOL):
                        pending.pop(c, None)
                else:
                    rem = p["remaining"]
                    if rem <= 1e-9 or cash <= 1e-9 or not can_buy[e, c]:
                        continue
                    v, _ = attempt(e, c, "B", rem, None, "retry-buy")
                    turn += v
                    p["remaining"] -= v
                    if p["remaining"] <= 1e-9:
                        pending.pop(c, None)
            pos_daily[env.dates[e]] = pos.copy()
            continue
        # (3) rebalance day
        for c in list(pending):
            if pending[c]["side"] == "B":
                pending.pop(c, None)
        i0, a = exec_map[e]
        ok = env.mask[i0] & np.isfinite(score[i0])
        if ok.sum() < MIN_UNIVERSE:
            equity_by_exec[e] = cash + pos.sum()
            turn_by_exec[e] = 0.0
            nh_by_exec[e] = int((pos > 1e-9).sum())
            pos_daily[env.dates[e]] = pos.copy()
            continue
        pick = _topk(score[i0], ok, k)
        target = set(int(c) for c in pick)
        eq = cash + pos.sum()
        tgt_v = eq / k
        turn = 0.0
        # 4a non-target: sell all (blocked -> pending sell)
        for c in np.where(pos > 1e-9)[0]:
            if c in target or int(c) in frozen:
                continue
            if not can_sell[e, c]:
                pending[c] = {"side": "S", "target": 0.0}
                continue
            v, _ = attempt(e, c, "S", pos[c], None, "exit")
            turn += v
            if v < pos[c] * (1 - FILL_TOL):
                pending[c] = {"side": "S", "target": 0.0}
        # 4b trim target above target value
        for c in sorted(target):
            if int(c) in frozen or pos[c] <= tgt_v * (1 + 1e-9):
                continue
            if not can_sell[e, c]:
                continue                      # A: blocked trims are logged, NOT persisted
            want = pos[c] - tgt_v
            v, _ = attempt(e, c, "S", want, None, "trim")
            turn += v
        # 4c top-up target below target value (cash-scaled)
        wants = [(int(c), tgt_v - pos[c]) for c in pick
                 if int(c) not in frozen and pos[c] < tgt_v - 1e-9 and can_buy[e, c]]
        if wants:
            need_tot = sum(w for _, w in wants)
            scale = min(1.0, max(cash, 0.0) / max(need_tot * (1 + cost), 1e-12))
            for c, need in wants:
                v_need = need * scale
                if v_need <= 1e-9:
                    continue
                v, _ = attempt(e, c, "B", v_need, None, "topup")
                turn += v
                if v < v_need * (1 - FILL_TOL):
                    pending[c] = {"side": "B", "remaining": v_need - v}
        equity_by_exec[e] = cash + pos.sum()
        turn_by_exec[e] = turn / max(2.0 * eq, 1e-12)
        nh_by_exec[e] = int((pos > 1e-9).sum())
        pos_daily[env.dates[e]] = pos.copy()

    order_e = sorted(equity_by_exec)
    rows = []
    for i in range(len(order_e) - 1):
        e0, e1 = order_e[i], order_e[i + 1]
        fee_period = sum(t[8] for t in trades if e0 < t[0] <= e1)
        base = max(equity_by_exec[e0], 1e-12)
        rows.append({
            "date": env.dates[e0 - 1], "start_date": env.dates[e0], "end_date": env.dates[e1],
            "net": equity_by_exec[e1] / base - 1.0,
            "gross": equity_by_exec[e1] / base - 1.0 + fee_period / base,
            "turn": turn_by_exec.get(e0, np.nan),
            "n_hold": nh_by_exec.get(e1, 0),
        })
    df = pd.DataFrame(rows)
    tl = pd.DataFrame(trades, columns=["day", "code", "side", "kind", "status",
                                       "desired", "filled", "price", "fee"])
    out = {"df": df, "trades": tl, "equity": equity_by_exec,
           "st_forced": st_forced, "st_forced_orders": st_forced_orders,
           "st_forced_blocked": st_forced_blocked,
           "nh": nh_by_exec, "turn": turn_by_exec}
    if track_pnl:
        out["pnl_by_code"] = pd.Series(pnl_by_code).sort_values(ascending=False)
        out["pnl_year_code"] = pd.Series(pnl_year_code)
    if return_pos:
        out["pos_daily"] = pos_daily
    return out


def bench_for(env, pairs, price="open"):
    P = env.ao if price == "open" else env.adjc
    out = []
    for i0, i1 in pairs:
        r = P[i1 + 1] / P[i0 + 1] - 1.0
        bm = r[env.mask[i0] & np.isfinite(r)]
        out.append(bm.mean() if len(bm) else np.nan)
    return np.asarray(out, dtype=np.float64)
