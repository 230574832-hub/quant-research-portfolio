"""P1.5 可信任执行引擎 (TrustedExecutor).

相对 p1_portfolio.PortfolioExecutor 的六处修复, 每一处都对应 p15_audit.py 的取证结论:

  R1 无前视选股池   : mask 只含截至信号日 d 收盘可知的信息; 执行器不再假设 d+1 可交易,
                       d+1 能否成交完全由当日真实行情判定 (build_no_lookahead_mask).
  R2 买卖方向       : 开盘封涨停 -> 买单受阻但**卖单可成交**; 开盘封跌停 -> 卖单受阻但
                       **买单可成交**. 旧报告 2.3 节把涨停列为"卖出受阻"是方向性错误.
                       涨跌停标记在构造时先剔除停牌行(停牌行 LimitUp=PreClosePrice 会误判).
  R3 绝对金额会计   : pos[c]=持仓市值(元), cash=现金(元), equity=cash+sum(pos).
                       每笔订单、每个交易日断言恒等式; 不再用"权重/净值同比缩放"这种
                       会把现金按组合收益缩放、并把成本扣两次的近似.
  R4 逐日重试       : 卖出受阻的订单进入 pending 队列, 后续每个交易日重试, 而非等到
                       下一个调仓日(10 个交易日后)才再试.
    R5 不可退出持仓   : 价格序列中断/长期停牌的持仓按最后有效价冻结估值, 不产生虚假收益,
                       不强制卖出; 连续 writeoff_after 日仍无价格时按 writeoff_recovery
                       计提会计减值(不是成交, 单独记 status='writeoff'), 且可用
                       recovery=1.0 完全关闭以做敏感性对照. 单列 frozen / writeoff 统计.
  R6 逐笔订单日志   : 每笔订单记录 side/status/reason/notional/price/cash/equity,
                       支持资金与持仓会计逐笔核对.

预留(第二阶段接口): participation / max_weight / slice_orders / impact_bps.
"""
import numpy as np
import pandas as pd

EPS = 0.0051


def build_no_lookahead_mask(panel):
    """只用截至 d 日收盘可知的信息构造股票池.

    与 build_factors.py 原 mask 的差异: 删除全部 shift(-1) 条件
      traded_t1 / y_all.notna() / open_.shift(-1)>0 / adj_open.shift(-1)>0
    这些条件要求"次日有成交、次日有收盘价、次日开盘价有效", 在 d 日收盘后并不可知.
    """
    close = panel["close"].values.astype(np.float64)
    vol = panel["volume"].values.astype(np.float64)
    amt = panel["amount"].values.astype(np.float64)
    precl = panel["preclose"].values.astype(np.float64)
    is_st = np.asarray(panel["is_st"]).astype(bool)
    board = np.asarray(panel["board"]).astype(str)
    rm20 = pd.DataFrame(amt).rolling(20, min_periods=10).mean().values
    listed = np.cumsum(np.isfinite(close), axis=0)
    board_ok = np.isin(board, ["main", "chinext", "star"])[None, :]
    return ((listed >= 120) & (~is_st) & board_ok & (close >= 2)
            & (vol > 0) & (amt > 0) & (rm20 >= 2e7) & np.isfinite(precl))


class TrustedExecutor:
    def __init__(self, panel, mask, dates, limit_panel, cost_bps=15.0, k=50,
                 strict=True, stale_limit=20, writeoff_after=60,
                 writeoff_recovery=0.1, participation=None,
                 max_weight=None, impact_bps_fn=None, min_universe=100,
                 retry_daily=True, writeoff_keep_mtm=False,
                 writeoff_event_mask=None,
                 status_mode="v4", status_matrix=None, delist_stage=None,
                 clip_hi=10.0):
        self.dates = dates
        self.mask = np.asarray(mask).astype(bool)
        self.N, self.M = self.mask.shape
        self.cost = cost_bps / 1e4
        self.K = k
        self.strict = strict
        self.min_universe = min_universe
        self.stale_limit = stale_limit
        self.writeoff_after = writeoff_after
        self.writeoff_recovery = writeoff_recovery
        self.participation = participation
        self.max_weight = max_weight
        self.impact_bps_fn = impact_bps_fn
        self.retry_daily = retry_daily
        # P1.5-B 诊断参数: 计提减值后是否仍让该持仓参与盯市.
        # False = 旧 S1/S2 行为(永久移出盯市, 复牌收益一并丢弃);
        # True  = 只把账面价值打折, 保留后续价格变动, 用于归因分解.
        self.writeoff_keep_mtm = writeoff_keep_mtm
        # P1.5-B: (N,M) bool, 仅在"时点可知的官方事件日"允许计提减值.
        # 传入时取代连续无价天数阈值, 用于区分真退市与数据缺口.
        self.writeoff_event_mask = (np.asarray(writeoff_event_mask).astype(bool)
                                    if writeoff_event_mask is not None else None)
        # P1.5-C: 交易状态语义.
        #   v4 = 旧口径 (Trdsta!=1 一律视为停牌) —— 已被证实错误: ST 状态有真实成交.
        #   v5 = 只用"当日无成交 / 无有效价"判定不可交易, Trdsta!=1 不再等于停牌.
        self.status_mode = status_mode
        self.clip_hi = clip_hi
        self.status_matrix = (np.asarray(status_matrix).astype(np.uint8)
                              if status_matrix is not None else None)
        self.delist_stage = (np.asarray(delist_stage).astype(np.uint8)
                             if delist_stage is not None else None)

        close = panel["close"].values.astype(np.float64)
        openp = panel["open"].values.astype(np.float64)
        adjc = panel["adjclose"].values.astype(np.float64)
        vol = panel["volume"].values.astype(np.float64)
        amt = panel["amount"].values.astype(np.float64)
        with np.errstate(all="ignore"):
            self.ao = openp * (adjc / close)
        self.open_raw = openp
        self.adjc = adjc
        self.amt = amt
        self.valid_px = np.isfinite(self.ao) & (self.ao > 0) & np.isfinite(openp) & (openp > 0)
        self.no_trade = np.isfinite(vol) & (vol <= 0)

        lu = np.asarray(limit_panel["limit_up"], dtype=np.float64)
        ld = np.asarray(limit_panel["limit_down"], dtype=np.float64)
        ts = np.asarray(limit_panel["trdsta"], dtype=np.float64)
        has_lu = np.isfinite(lu) & (lu > 0)
        has_ld = np.isfinite(ld) & (ld > 0)
        if self.status_mode == "v5":
            # P1.5-C: Trdsta=2/3/10 是 ST/风险警示状态, 不是停牌.
            # 不可交易只由"当日无成交"或"无有效价(数据缺失/摘牌后)"判定.
            self.suspended = self.no_trade | (~self.valid_px)
            if self.status_matrix is not None:
                self.data_missing = (self.status_matrix == 4)      # DATA_MISSING
                self.delist_risk = (self.status_matrix == 5)       # DELIST_PROC
                if self.delist_stage is not None:
                    self.delist_risk |= (self.delist_stage == 2)   # 已摘牌
            else:
                self.data_missing = ~self.valid_px
                self.delist_risk = np.zeros_like(self.no_trade)
        else:
            # V4 旧口径(保留用于对照): trdsta!=1 一律视为停牌 —— 本轮证实为错误.
            self.suspended = (np.isfinite(ts) & (ts != 1)) | self.no_trade
            self.data_missing = np.zeros_like(self.no_trade)
            self.delist_risk = np.zeros_like(self.no_trade)

        # R2: 涨跌停标记必须先剔除停牌行(CSMAR 停牌行 LimitUp=PreClosePrice)
        tradable = self.valid_px & ~self.suspended
        self.up_open = has_lu & (openp >= lu - EPS) & tradable
        self.dn_open = has_ld & (openp <= ld + EPS) & tradable
        # 开盘封涨停: 买单排不上, 卖单可以立刻成交 -> can_sell 不排除 up_open
        # 开盘封跌停: 卖单排不上, 买单可以立刻成交 -> can_buy 不排除 dn_open
        self.can_sell = tradable & ~self.dn_open
        self.can_buy = tradable & ~self.up_open

    # ------------------------------------------------------------------
    def _assert_accounting(self, pos, cash, equity_expected, day, tag):
        eq = cash + float(pos.sum())
        err = abs(eq - equity_expected)
        if self.strict and err > 1e-8:
            raise AssertionError("会计恒等式破裂 day=%d %s: cash+pos=%.10f vs 期望 %.10f (err=%.2e)"
                                 % (day, tag, eq, equity_expected, err))
        return err

    def simulate(self, score, pairs, bench=None, label=""):
        """score:(N,M) 信号; pairs:[(信号日 i0, 下一信号日 i1)] -> 执行日 e=i0+1."""
        K = self.K
        exec_map = {}
        for a, (i0, i1) in enumerate(pairs):
            e = i0 + 1
            if e < self.N and i1 + 1 < self.N:
                exec_map[e] = (i0, a)
        exec_days = sorted(exec_map)
        if len(exec_days) < 2:
            return pd.DataFrame()
        start_e, end_e = exec_days[0], exec_days[-1]

        pos = np.zeros(self.M, dtype=np.float64)
        cash = 1.0
        orders = []
        diag = {"filled": 0, "blocked": 0, "buy_blocked": 0, "sell_blocked": 0,
                "retry_filled": 0, "pending_max": 0, "frozen_n": 0,
                "frozen_value_end": 0.0, "unfilled_target": 0, "cost_paid": 0.0,
                "max_accounting_err": 0.0, "turnover": 0.0,
                "writeoff_n": 0, "writeoff_loss": 0.0}
        pending = {}            # code -> 首次受阻日
        held_since = {}         # code -> 首次买入日
        stale = {}              # code -> 连续无有效价天数
        frozen = set()
        written_off = set()
        equity_by_exec, turn_by_exec, nh_by_exec = {}, {}, {}
        # P1.5-B: 逐期权益三分类敞口(纯诊断, 不参与任何收益计算)
        #   confirmed  = 现金 + 当日可正常卖出的持仓市值 (可立即变现)
        #   restricted  = 当日不可卖但有有效价(跌停/停牌/无成交)的持仓市值 (延迟变现)
        #   uncertain   = 连续无有效价 >= stale_limit 的冻结持仓市值 (价值依赖未来)
        # 分类只使用 e 日及之前的行情与累计 stale 计数, 不含任何未来信息.
        exposure_by_exec = {}
        sig = np.asarray(score)

        def record_exposure(e):
            eq = cash + float(pos.sum())
            cols = np.where(pos > 1e-12)[0]
            w_conf = w_rest = w_unc = 0.0
            n_conf = n_rest = n_unc = 0
            # P1.5-C: 不可退出权益三分拆 (互斥完备)
            #   data    = 生命周期内原始数据缺失 -> 无法定价
            #   delist  = 时点可知的退市流程 / 已摘牌 -> 回收价值不确定
            #   restrict= 真实交易约束(停牌/涨跌停封板/长期无价冻结)
            w_data = w_delist = w_restrict = w_tradable = 0.0
            n_data = n_delist = n_restrict = n_tradable = 0
            for c in cols:
                v = pos[c]
                if c in frozen:
                    w_unc += v; n_unc += 1
                elif not self.can_sell[e, c]:
                    w_rest += v; n_rest += 1
                else:
                    w_conf += v; n_conf += 1
                if self.delist_risk[e, c]:
                    w_delist += v; n_delist += 1
                elif self.data_missing[e, c]:
                    w_data += v; n_data += 1
                elif c in frozen or not self.can_sell[e, c]:
                    w_restrict += v; n_restrict += 1
                else:
                    w_tradable += v; n_tradable += 1
            w_conf += cash
            w_tradable += cash
            frozen_all = w_data + w_delist + w_restrict
            exposure_by_exec[e] = {
                "equity": float(eq), "cash": float(cash),
                "confirmed": float(w_conf), "restricted": float(w_rest),
                "uncertain": float(w_unc),
                "fcr": float(w_unc / eq) if eq > 0 else 0.0,
                "rcr": float(w_rest / eq) if eq > 0 else 0.0,
                "n_conf": n_conf, "n_rest": n_rest, "n_unc": n_unc,
                "fcr_data": float(w_data / eq) if eq > 0 else 0.0,
                "fcr_delist": float(w_delist / eq) if eq > 0 else 0.0,
                "fcr_restrict": float(w_restrict / eq) if eq > 0 else 0.0,
                "fcr_total": float(frozen_all / eq) if eq > 0 else 0.0,
                "tradable": float(w_tradable),
                "n_data": n_data, "n_delist": n_delist,
                "n_restrict": n_restrict, "n_tradable": n_tradable,
            }

        def book(e, c, side, status, reason, v, px, before=None, fee=0.0):
            eq_after = cash + pos.sum()
            row = {
                "day": int(e), "date": str(self.dates[e].date()),
                "code": str(c), "side": side, "status": status, "reason": reason,
                "notional": float(v), "price": float(px) if np.isfinite(px) else np.nan,
                "fee": float(fee),
                "cash_after": float(cash), "equity_after": float(eq_after),
                "pos_sum_after": float(pos.sum()),
                "cash_before": float(before[0]) if before else np.nan,
                "equity_before": float(before[1]) if before else np.nan,
                "pos_sum_before": float(before[2]) if before else np.nan,
            }
            orders.append(row)
            return row

        def do_trade(e, c, side, v, px, status, reason):
            """v 为含成本前的成交金额; 买卖不改变净值, 只有手续费改变净值."""
            nonlocal cash
            before = (cash, cash + pos.sum(), pos.sum())
            eq_before = before[1]
            extra = 0.0
            if self.impact_bps_fn is not None:
                extra = v * self.impact_bps_fn(e, c, v) / 1e4
            fee = v * self.cost + extra
            if side == "S":
                pos[c] -= v
                cash += v
            else:
                pos[c] += v
                cash -= v
            cash -= fee
            diag["cost_paid"] += fee
            diag["turnover"] += v
            diag["max_accounting_err"] = max(
                diag["max_accounting_err"],
                self._assert_accounting(pos, cash, eq_before - fee, e,
                                        "%s %s" % (side, status)))
            diag["filled"] += 1
            if status == "retry_filled":
                diag["retry_filled"] += 1
            book(e, c, side, status, reason, v, px, before, fee)

        # ------------------------------------------------------------ 逐日事件循环
        for e in range(start_e, end_e + 1):
            # (1) 标记到市: 逐日 open-to-open, 无效价格 -> 收益 0 并累计 stale
            if e > start_e:
                cols = np.array([c for c in np.where(pos > 1e-12)[0]
                                 if c not in written_off or self.writeoff_keep_mtm], dtype=int)
                if len(cols):
                    with np.errstate(all="ignore"):
                        r = self.ao[e, cols] / self.ao[e - 1, cols] - 1.0
                    good = np.isfinite(r) & np.isfinite(self.ao[e, cols]) & (self.ao[e, cols] > 0)
                    r = np.where(good, r, 0.0)
                    r = np.clip(r, -0.99, self.clip_hi)
                    pos[cols] *= (1.0 + r)
                    for c in cols[good]:
                        stale[c] = 0
                        frozen.discard(c)
                    for c in cols[~good]:
                        stale[c] = stale.get(c, 0) + 1
                        if stale[c] >= self.stale_limit:
                            frozen.add(c)

            # (1b) 事实退市/长期无价格: 计提会计减值(不是成交, 不假设能强制卖出)
            #      recovery=1.0 表示完全关闭减值, 只冻结估值(敏感性对照口径)
            if self.writeoff_event_mask is not None:
                # P1.5-B 事件驱动: 只在时点可知的官方退市公告日计提, 且仅针对
                # 该事件对应的股票. 不因"连续无价"就一律减值(避免把数据缺口当退市).
                if self.writeoff_recovery < 1.0:
                    hit = np.where(self.writeoff_event_mask[e] & (pos > 1e-12))[0]
                    for c in hit:
                        if c in written_off:
                            continue
                        before = (cash, cash + pos.sum(), pos.sum())
                        loss = pos[c] * (1.0 - self.writeoff_recovery)
                        pos[c] *= self.writeoff_recovery
                        diag["writeoff_n"] += 1
                        diag["writeoff_loss"] += loss
                        written_off.add(c)
                        pending.pop(c, None)
                        book(e, c, "W", "writeoff", "官方退市公告日, 计提减值",
                             loss, np.nan, before, 0.0)
                        if self.strict:
                            self._assert_accounting(pos, cash, before[1] - loss, e, "事件减值")
            elif self.writeoff_after is not None and self.writeoff_recovery < 1.0:
                for c in list(frozen):
                    if c in written_off or pos[c] <= 1e-12:
                        continue
                    if stale.get(c, 0) < self.writeoff_after:
                        continue
                    before = (cash, cash + pos.sum(), pos.sum())
                    loss = pos[c] * (1.0 - self.writeoff_recovery)
                    pos[c] *= self.writeoff_recovery
                    diag["writeoff_n"] += 1
                    diag["writeoff_loss"] += loss
                    written_off.add(c)
                    pending.pop(c, None)
                    book(e, c, "W", "writeoff", "连续%d日无有效价, 计提减值" % self.writeoff_after,
                         loss, np.nan, before, 0.0)
                    if self.strict:
                        self._assert_accounting(pos, cash, before[1] - loss, e, "减值")

            # (2) 非调仓日: 只重试积压卖单 (retry_daily=False 时关闭, 用于逐项分解)
            if e not in exec_map:
                if self.retry_daily:
                    for c in list(pending):
                        if c in written_off or pos[c] <= 1e-12:
                            pending.pop(c, None)
                            held_since.pop(c, None)
                            continue
                        if self.can_sell[e, c]:
                            v = pos[c]
                            do_trade(e, c, "S", v, self.ao[e, c],
                                     "retry_filled", "积压卖单重试成交")
                            pending.pop(c, None)
                            held_since.pop(c, None)
                        elif c in frozen:
                            pass                   # 不可退出持仓: 保留真实仓位与风险
                eq = cash + pos.sum()
                if self.strict:
                    self._assert_accounting(pos, cash, eq, e, "非调仓日")
                record_exposure(e)
                continue

            # (3) 调仓日: 生成目标
            i0, a = exec_map[e]
            ok = self.mask[i0] & np.isfinite(sig[i0])
            if ok.sum() < self.min_universe:
                equity_by_exec[e] = cash + pos.sum()
                turn_by_exec[e] = 0.0
                nh_by_exec[e] = int((pos > 1e-12).sum())
                record_exposure(e)
                continue
            idx = np.where(ok)[0]
            pick = idx[np.argsort(-sig[i0][idx])[:K]]
            target = set(int(c) for c in pick)
            eq = cash + pos.sum()
            tgt_v = eq / K
            if self.max_weight is not None:
                tgt_v = min(tgt_v, eq * self.max_weight)
            turn = 0.0

            # 3a 非目标持仓: 能卖则全额卖, 否则进 pending
            for c in np.where(pos > 1e-12)[0]:
                if c in target or c in frozen or c in written_off:
                    continue
                if not self.can_sell[e, c]:
                    reason = ("停牌/无成交" if self.suspended[e, c] or self.no_trade[e, c]
                              else "开盘封跌停" if self.dn_open[e, c] else "无有效价格")
                    diag["sell_blocked"] += 1
                    diag["blocked"] += 1
                    pending[c] = pending.get(c, e)
                    book(e, c, "S", "blocked", reason, pos[c], self.ao[e, c])
                    continue
                v = pos[c]
                do_trade(e, c, "S", v, self.ao[e, c], "filled", "非目标调出")
                turn += v
                pending.pop(c, None)
                held_since.pop(c, None)

            # 3b 目标内: 超出 tgt_v 的部分减仓(涨停不妨碍卖出)
            for c in target:
                if c in written_off or c in frozen or pos[c] <= tgt_v * (1 + 1e-9):
                    continue
                if not self.can_sell[e, c]:
                    diag["sell_blocked"] += 1
                    diag["blocked"] += 1
                    book(e, c, "S", "blocked", "减仓受阻", pos[c] - tgt_v, self.ao[e, c])
                    continue
                v = pos[c] - tgt_v
                do_trade(e, c, "S", v, self.ao[e, c], "filled", "再平衡减仓")
                turn += v

            # 3c 目标内不足: 补足到 tgt_v, 受现金与涨停约束
            wants = []
            for c in pick:
                c = int(c)
                if c in written_off or c in frozen or pos[c] >= tgt_v - 1e-12:
                    continue
                if not self.can_buy[e, c]:
                    reason = ("停牌/无成交" if self.suspended[e, c] or self.no_trade[e, c]
                              else "开盘封涨停" if self.up_open[e, c] else "无有效价格")
                    diag["buy_blocked"] += 1
                    diag["blocked"] += 1
                    diag["unfilled_target"] += 1
                    book(e, c, "B", "blocked", reason, tgt_v - pos[c], self.ao[e, c])
                    continue
                need = tgt_v - pos[c]
                if self.participation is not None:
                    cap = self.participation * self.amt[e, c] if np.isfinite(self.amt[e, c]) else 0.0
                    need = min(need, cap)
                    if need <= 1e-12:
                        diag["unfilled_target"] += 1
                        continue
                wants.append((c, need))
            if wants:
                need_tot = sum(v for _, v in wants)
                scale = min(1.0, max(cash, 0.0) / max(need_tot * (1 + self.cost), 1e-12))
                diag["unfilled_target"] += int(round(len(wants) * (1 - scale)))
                for c, need in wants:
                    v = need * scale
                    if v <= 1e-12:
                        continue
                    do_trade(e, c, "B", v, self.ao[e, c], "filled", "建仓/补仓")
                    turn += v
                    held_since.setdefault(c, e)

            diag["pending_max"] = max(diag["pending_max"], len(pending))
            equity_by_exec[e] = cash + pos.sum()
            turn_by_exec[e] = turn / max(2.0 * eq, 1e-12)
            nh_by_exec[e] = int((pos > 1e-12).sum())
            record_exposure(e)

        # ------------------------------------------------------------ 组装
        order_e = sorted(equity_by_exec)
        rows = []
        for a in range(len(order_e) - 1):
            e0, e1 = order_e[a], order_e[a + 1]
            with np.errstate(all="ignore"):
                br = self.ao[e1] / self.ao[e0] - 1.0
            bm = br[self.mask[e0 - 1] & np.isfinite(br)]
            rows.append({
                "date": self.dates[e0 - 1] if e0 - 1 >= 0 else self.dates[0],
                "net": equity_by_exec[e1] / max(equity_by_exec[e0], 1e-12) - 1.0,
                "bench": (bench[a] if bench is not None else (bm.mean() if len(bm) else np.nan)),
                "turn": turn_by_exec.get(e0, np.nan),
                "n_hold": nh_by_exec.get(e1, 0),
                "start_date": self.dates[e0], "end_date": self.dates[e1],
            })
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["date", "net", "bench", "turn", "n_hold",
                                       "start_date", "end_date"])
        df = df.set_index("date")
        df["gross"] = df["net"]          # 调用方用 cost=0 重跑覆盖
        df.attrs["orders"] = pd.DataFrame(orders)
        df.attrs["diag"] = diag
        df.attrs["equity"] = equity_by_exec
        exp = pd.DataFrame.from_dict(exposure_by_exec, orient="index").sort_index()
        exp.index.name = "day"
        exp["date"] = [str(self.dates[int(i)].date()) for i in exp.index]
        df.attrs["exposure"] = exp
        # 逐期订单口径(严格互斥三分类), 供 P1.5-B 澄清成交率定义
        od_raw = df.attrs["orders"]
        n_all = len(od_raw)
        cnt = od_raw["status"].value_counts().to_dict() if n_all else {}
        df.attrs["fill_stats"] = {
            "n_all": n_all,
            "n_filled": int(cnt.get("filled", 0)),
            "n_blocked": int(cnt.get("blocked", 0)),
            "n_retry_filled": int(cnt.get("retry_filled", 0)),
            "n_writeoff": int(cnt.get("writeoff", 0)),
        }
        od = df.attrs["orders"]
        trade_orders = od[od["status"] != "writeoff"] if len(od) else od
        df.attrs["fill_rate"] = (float((trade_orders["status"] != "blocked").mean())
                                 if len(trade_orders) else np.nan)
        fz = [int(c) for c in frozen if pos[c] > 1e-12]
        diag["frozen_n"] = len(fz)
        diag["frozen_value_end"] = float(sum(pos[c] for c in fz))
        df.attrs["held_detail"] = [
            {"code": int(c), "first_buy": int(v),
             "age_days": int(end_e - v), "weight": float(pos[c] / max(cash + pos.sum(), 1e-12)),
             "frozen": c in frozen, "written_off": c in written_off,
             "stale_days": int(stale.get(c, 0))}
            for c, v in held_since.items() if pos[c] > 1e-12]
        df.attrs["held_detail"].sort(key=lambda x: -x["age_days"])
        return df
