"""P1 第二轮: 真实持仓状态机执行器 (Portfolio Executor).

与 p1_engine.Executor (旧版) 的差异 —— 这是第二轮审计的核心升级:
  1. 真实持仓跨期延续: 开盘跌停卖不掉的股票继续持有到下一期, 而非"丢掉不算收益"
     或"顺延 5 日强行卖出"(旧版在顺延失败后仍按 close 计价, 隐含不可实现的成交).
  2. 买入失败即放弃(不成交), 资金留在现金, 不虚构成交价.
  3. 涨跌停判定改用真实 up_limit/down_limit 原始价格(而非 ±9.8%/±19.8% 阈值).
  4. 时点对齐: 第 d 日的涨跌停价由第 d-1 日收盘价决定, 开盘前即已知 -> 可用于 d 日
     开盘成交判定; 但绝不使用第 d 日 LimitStatus(=收盘封板信息) 判断 d 日开盘.
  5. 停牌(trdsta != 1)当日完全不可交易.
  6. 权重空间 + 复权价格计价, 天然处理分红除权; 每期再平衡回等权目标.
"""
import numpy as np
import pandas as pd

EPS = 0.0051  # 涨跌停价按 0.01 取整, 比较容差


class PortfolioExecutor:
    def __init__(self, panel, mask, dates, cost_bps=15.0, max_delay=5,
                 limit_panel=None, use_real_limits=True):
        self.dates = dates
        self.mask = mask
        self.N, self.M = mask.shape
        self.cost = cost_bps / 1e4
        self.max_delay = max_delay

        adjc = panel["adjclose"].values.astype(np.float64)
        close = panel["close"].values.astype(np.float64)
        openp = panel["open"].values.astype(np.float64)
        precl = panel["preclose"].values.astype(np.float64)
        self.adjc = adjc
        with np.errstate(all="ignore"):
            self.ao = openp * (adjc / close)
        self.open_raw = openp

        self.ok_open = np.isfinite(self.ao) & np.isfinite(openp) & (openp > 0)

        # 无成交日(停牌/无交易)不可交易: 面板中停牌行 volume==0, open==close==preclose
        vol = panel["volume"].values.astype(np.float64)
        self.no_trade = np.isfinite(vol) & (vol <= 0)
        self.ok_open = self.ok_open & (~self.no_trade | ~np.isfinite(vol))

        self.mode = "real" if (use_real_limits and limit_panel is not None) else "thresh"
        if self.mode == "real":
            def _g(key):
                v = limit_panel[key]
                return (v.values if hasattr(v, "values") else np.asarray(v)).astype(np.float64)

            lu = _g("limit_up")
            ld = _g("limit_down")
            has_lu = np.isfinite(lu) & (lu > 0)
            has_ld = np.isfinite(ld) & (ld > 0)
            self.up_open = has_lu & (openp >= lu - EPS)     # 开盘封涨停 -> 买不进
            self.dn_open = has_ld & (openp <= ld + EPS)     # 开盘封跌停 -> 卖不出
            ts = _g("trdsta")
            self.suspended = np.isfinite(ts) & (ts != 1)
            self.suspended = self.suspended | self.no_trade
            self.ok_open = self.ok_open & (~self.suspended | ~np.isfinite(ts))
            self.limit_up, self.limit_down = lu, ld
        else:
            board = panel["board"]
            lim = np.where(np.asarray(board) == "main", 0.098, 0.198)[None, :]
            with np.errstate(all="ignore"):
                oret = openp / precl - 1.0
            self.up_open = oret >= lim
            self.dn_open = oret <= -lim
            self.suspended = np.zeros_like(self.ok_open)
            self.limit_up = self.limit_down = None

    def grid(self, hold_days=5):
        iso = self.dates.isocalendar()
        wk = self.dates.to_series().groupby([iso.year, iso.week]).max()
        wdates = pd.DatetimeIndex(sorted(set(wk.values)))
        step = {5: 1, 10: 2, 20: 4}[hold_days]
        return np.array([self.dates.get_loc(d) for d in wdates[::step]])

    # ------------------------------------------------------------------
    def simulate(self, score, rebal, exclude=None, k=50, periods=None):
        """score:(N,M) 信号(越大越买); rebal: 调仓信号日索引数组.
        periods: 可选, 显式给定 [(i0,i1), ...] 期间对, 用于跨版本统一口径.
        返回 df[date, gross, net, turn, bench, n_hold] + attrs['diag','equity','trades']"""
        K = k
        w = np.zeros(self.M)        # 各股权重
        cash = 1.0                  # 现金(绝对值)
        equity = 1.0
        prev_day = 0
        rows = []
        diag = {"buy_blocked": 0, "sell_blocked": 0, "trades": 0, "carried": 0,
                "unfilled_target": 0, "excluded": 0, "suspended_hit": 0,
                "max_hold_days": 0}
        trades = []   # (exec_day, code, side, notional)
        held_since = {}
        exec_net, exec_gross, exec_turn, exec_nh = {}, {}, {}, {}

        pairs = periods if periods is not None else [
            (rebal[k], rebal[k + 1]) for k in range(len(rebal) - 1)]
        for kk in range(len(pairs)):
            i, i2 = pairs[kk]
            e = i + 1
            if e >= self.N or i2 + 1 >= self.N:
                continue

            # ---- 从上次执行日开盘 标记到 本次执行日开盘 ----
            if e > prev_day:
                with np.errstate(all="ignore"):
                    r = self.ao[e] / self.ao[prev_day] - 1.0
                r = np.where(np.isfinite(r), r, 0.0)
                port_r = float((w * r).sum())
                denom = 1.0 + port_r
                if denom <= 0:
                    denom = 1e-9
                equity *= denom
                w = w * (1.0 + r) / denom
                cash = cash / denom
            prev_day = e

            # ---- 信号 ----
            sig = score[i]
            ok = self.mask[i] & np.isfinite(sig)
            if exclude is not None:
                ex_m = exclude[i]
                diag["excluded"] += int((ok & ex_m).sum())
                ok = ok & (~ex_m)
            if self.mask[i].sum() < 100:
                continue
            idx = np.where(ok)[0]
            pick = idx[np.argsort(-sig[idx])[:K]]
            target = set(pick.tolist())
            tgt_w = 1.0 / K

            eq_pre_cost = equity
            turnover = 0.0

            # ---------------- 卖出 ----------------
            # (a) 非目标持仓: 全部卖出, 开盘跌停/停牌则保留(持仓延续)
            for c in np.where(w > 1e-9)[0]:
                if c in target:
                    continue
                if self.suspended[e, c]:
                    diag["suspended_hit"] += 1
                    diag["sell_blocked"] += 1
                    diag["carried"] += 1
                    continue
                px_s = self.ao[e, c]
                if not (np.isfinite(px_s) and px_s > 0):
                    # 当日无有效价格(停牌/无成交) -> 无法卖出, 持仓延续
                    diag["sell_blocked"] += 1
                    diag["carried"] += 1
                    held_since[c] = held_since.get(c, e)
                    continue
                if self.dn_open[e, c]:
                    diag["sell_blocked"] += 1
                    diag["carried"] += 1
                    held_since[c] = held_since.get(c, e)   # 延续持有
                    continue
                v = w[c] * equity
                w[c] = 0.0
                cash += v
                turnover += v
                diag["trades"] += 1
                trades.append((e, int(c), "S", v))
                held_since.pop(c, None)

            # (b) 目标内已持有的超额部分: 减到 1/K
            for c in list(target):
                if w[c] <= 1e-9:
                    continue
                if w[c] > tgt_w:
                    px_o = self.ao[e, c]
                    if (self.suspended[e, c] or self.dn_open[e, c]
                            or not (np.isfinite(px_o) and px_o > 0)):
                        diag["sell_blocked"] += 1
                        held_since.setdefault(c, e)
                        continue
                    v = (w[c] - tgt_w) * equity
                    w[c] = tgt_w
                    cash += v
                    turnover += v
                    diag["trades"] += 1

            # ---------------- 买入 ----------------
            # 目标内权重不足/未持有: 补足到 1/K, 受现金与涨停约束
            wants = []
            for c in pick:
                cur = w[c]
                if cur >= tgt_w - 1e-9:
                    continue
                if self.suspended[e, c]:
                    diag["suspended_hit"] += 1
                    diag["buy_blocked"] += 1
                    diag["unfilled_target"] += 1
                    continue
                if self.up_open[e, c]:
                    diag["buy_blocked"] += 1
                    diag["unfilled_target"] += 1
                    continue
                px = self.ao[e, c]
                if not (np.isfinite(px) and px > 0):
                    diag["buy_blocked"] += 1
                    diag["unfilled_target"] += 1
                    continue
                wants.append((c, tgt_w - cur))
            if wants:
                tot_need = sum(dw for _, dw in wants)
                scale = min(1.0, cash / (tot_need * equity)) if tot_need > 0 else 0.0
                if scale < 1.0:
                    diag["unfilled_target"] += int(len(wants) * (1 - scale))
                for c, dw in wants:
                    v = dw * scale * equity
                    if v <= 0:
                        continue
                    w[c] += v / equity
                    cash -= v
                    turnover += v
                    diag["trades"] += 1
                    held_since.setdefault(c, e)

            # ---------------- 成本 ----------------
            cost_amt = turnover * self.cost
            equity = eq_pre_cost - cost_amt
            cash -= cost_amt
            fac = equity / max(eq_pre_cost, 1e-12)
            w = w * fac
            cash = max(cash * fac, 0.0)

            exec_net[e] = equity
            exec_gross[e] = eq_pre_cost
            exec_turn[e] = turnover / max(2.0 * eq_pre_cost, 1e-12)
            exec_nh[e] = int((w > 1e-9).sum())
            for _c in [k for k, v in held_since.items() if w[k] <= 1e-9]:
                held_since.pop(_c, None)
            if held_since:
                ages = [e - v for v in held_since.values()]
                diag["max_hold_days"] = max(diag["max_hold_days"], max(ages) if ages else 0)

        # ---------------- 组装期间收益 ----------------
        order = sorted(exec_net)
        for a in range(len(order) - 1):
            e0, e1 = order[a], order[a + 1]
            if e1 <= e0:
                continue
            d = self.dates[e0 - 1] if e0 - 1 >= 0 else self.dates[0]
            net = exec_net[e1] / exec_net[e0] - 1.0
            # 纯收益(不含任何成本): 用各自期初的税前权益相除
            gross = exec_gross[e1] / exec_gross[e0] - 1.0
            with np.errstate(all="ignore"):
                br = self.ao[e1] / self.ao[e0] - 1.0
            bm = br[self.mask[e0 - 1] & np.isfinite(br)]
            b = bm.mean() if len(bm) else np.nan
            rows.append({"date": d, "gross": gross, "net": net,
                         "turn": exec_turn.get(e0, np.nan), "bench": b,
                         "n_hold": exec_nh.get(e1, 0),
                         "start_date": self.dates[e0], "end_date": self.dates[e1]})

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["date", "gross", "net", "turn", "bench", "n_hold",
                                       "start_date", "end_date"])
        df = df.set_index("date")
        df.attrs["diag"] = diag
        df.attrs["equity"] = exec_net
        df.attrs["trades"] = trades
        df.attrs["held_detail"] = [
            {"code": int(c), "first_buy": int(v), "age_days": int(max(exec_net) - v),
             "weight": float(w[c])}
            for c, v in held_since.items() if w[c] > 1e-9]
        df.attrs["held_detail"].sort(key=lambda x: -x["age_days"])
        return df
