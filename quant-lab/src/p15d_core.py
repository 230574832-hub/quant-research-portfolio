"""P1.5-D 核心: 冻结 V5 环境 + 资金规模驱动的成交仿真引擎.

冻结声明 (本模块不提供任何策略参数调节入口):
  * 面板    : panel_v5 / limit_panel_v5 / factors_v5 / p15c_status
  * 股票池  : build_no_lookahead_mask(panel_v5), 与 P1.5-C 逐格一致
  * 因子    : SIZE = -ln(mktcap), PRICE = -ln(close), 等权截面 rank 均值
  * 调仓    : 由 P1.5-C 的 V4 股票池计数决定的 canonical 网格 (见 p15c_universe.csv),
              HOLD=10 真实交易日, K=50, 等权目标, 下一交易日开盘成交
  * 基准    : 同窗口等权全池 open-to-open (make_bench, 与 p15c_run.py 完全相同)
  * 成本    : V5 基准单边 15bp; 本轮只把它拆分成四个互斥费项, 不额外多收一次

P1.5-D 新增的只有三件事, 全部属于"资金规模 -> 成交结果"的执行层, 不是策略层:
  A. 因果 ADTV20 (只用 e-20 .. e-1 的成交额, 绝不使用 e 日全天成交额)
  B. 开盘可用流动性近似 open_share (无集合竞价成交量/盘口数据, 明确标为近似)
  C. 逐笔动态成交: 部分成交 / 未成交延续 / 现金约束 / 参与率上限 / 平方根冲击

成本分解 (防止重复计费):
  单边基础摩擦 15bp = 佣金 3bp + 过户费 0.2bp + 印花税(卖 10bp/5bp, 买 0)
                      + 价差及其他残差 bp
  市场冲击 impact_bps = Y * sigma_daily_bps * sqrt(实际成交额 / ADTV20)
  该冲击是**唯一叠加项**, 只在已成交金额上计一次; 价差项只是 15bp 的残差拆分,
  不另收一次. 因此 Y=0 且无容量约束时, 本引擎与 P1.5-C 的 V5 主口径逐笔一致.
"""
from __future__ import annotations

import gc
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from p1_engine import cs_rank, evaluate  # noqa: E402
from p15_engine import build_no_lookahead_mask  # noqa: E402

EPS = 0.0051
FILL_TOL = 1e-6        # 委托"全额成交"判定容差, 屏蔽浮点残差造成的伪挂单
FROZEN = {
    "HOLD": 10, "K": 50, "COST_BPS_SIDE": 15.0,
    "STALE_LIMIT": 20, "CLIP_HI": 10.0, "MIN_UNIVERSE": 100,
    "CAGR_NET_V5": 24.2982, "EXCESS_V5": 13.4727, "IR_V5": 0.7127,
    "MAXDD_V5": -43.709, "BENCH_V5": 9.5402,
}

# V5 状态码 (p15c_build_v5.py)
S_NOT_EXIST, S_NORMAL, S_ST, S_SUSPENDED, S_DATA_MISSING, S_DELIST_PROC = 0, 1, 2, 3, 4, 5
STATUS_NAME = {0: "上市前/摘牌后", 1: "正常交易", 2: "ST可交易", 3: "真实停牌",
               4: "数据缺失", 5: "退市流程"}


class Env:
    """冻结的 V5 数据环境. 所有面板只加载一次, 供上百次资金规模仿真复用."""

    def __init__(self, verbose=True):
        t0 = time.time()
        if verbose:
            print("[env] loading V5 panels ...", flush=True)
        P = pickle.load(open("data/panel_v5.pkl", "rb"))
        LP = pickle.load(open("data/limit_panel_v5.pkl", "rb"))
        D = pickle.load(open("data/factors_v5.pkl", "rb"))
        S = pickle.load(open("data/p15c_status.pkl", "rb"))

        self.dates = pd.DatetimeIndex(P["dates"])
        self.codes = pd.Index([str(c) for c in P["codes"]])
        self.N, self.M = len(self.dates), len(self.codes)
        f64 = lambda x: np.asarray(x, dtype=np.float64)
        self.openp = f64(P["open"].values)
        self.close = f64(P["close"].values)
        self.adjc = f64(P["adjclose"].values)
        self.amt = f64(P["amount"].values)
        self.vol = f64(P["volume"].values)
        self.precl = f64(P["preclose"].values)
        self.is_st = np.asarray(P["is_st"], dtype=bool)
        self.board = np.asarray(P["board"], dtype=str)
        self.lu = f64(LP["limit_up"])
        self.ld = f64(LP["limit_down"])
        self.trdsta = f64(LP["trdsta"])
        self.pc_csv = f64(LP["preclose_csv"])
        self.one_word_up = np.asarray(LP["one_word_up"], dtype=bool)
        self.one_word_dn = np.asarray(LP["one_word_dn"], dtype=bool)
        self.limit_status = f64(LP["limit_status"])
        self.status = np.asarray(S["status"], dtype=np.uint8)
        self.delist_stage = np.asarray(S["delist_stage"], dtype=np.uint8)
        size = f64(D["factors"]["SIZE"].values)
        price = f64(D["factors"]["PRICE"].values)
        del P, LP, D, S
        gc.collect()

        with np.errstate(all="ignore"):
            self.ao = self.openp * (self.adjc / self.close)
        self.valid_px = (np.isfinite(self.ao) & (self.ao > 0)
                         & np.isfinite(self.openp) & (self.openp > 0))
        self.no_trade = np.isfinite(self.vol) & (self.vol <= 0)

        thin = {"close": pd.DataFrame(self.close), "volume": pd.DataFrame(self.vol),
                "amount": pd.DataFrame(self.amt), "preclose": pd.DataFrame(self.precl),
                "is_st": self.is_st, "board": pd.Series(self.board)}
        self.mask = build_no_lookahead_mask(thin)
        del thin
        self.score = np.nanmean([cs_rank(size, self.mask), cs_rank(price, self.mask)],
                                axis=0).astype(np.float32)
        del size, price
        gc.collect()

        # canonical 调仓网格: 完全复用 P1.5-C 保存的 V4 股票池日计数, 不重挑样本
        u = pd.read_csv("out/p15c_universe.csv")
        v4 = u["V4"].to_numpy()
        assert len(v4) == self.N
        grid = np.arange(0, self.N, FROZEN["HOLD"])
        canon = [int(g) for g in grid if v4[g] >= FROZEN["MIN_UNIVERSE"]]
        self.PAIRS = [(canon[a], canon[a + 1]) for a in range(len(canon) - 1)]

        # 同窗口等权全池 open-to-open 基准 (与 p15c_run.make_bench 逐字一致)
        out = []
        for i0, i1 in self.PAIRS:
            with np.errstate(all="ignore"):
                br = self.ao[i1 + 1] / self.ao[i0 + 1] - 1.0
            bm = br[self.mask[i0] & np.isfinite(br)]
            out.append(bm.mean() if len(bm) else np.nan)
        self.BENCH = np.asarray(out, dtype=np.float64)

        self.adtv, self.sigma, self.sigma_med = self._causal_liquidity()
        if verbose:
            print("[env] N=%d M=%d pairs=%d  ADTV有限格=%d (%.1f%%)  %.1fs"
                  % (self.N, self.M, len(self.PAIRS), np.isfinite(self.adtv).sum(),
                     100 * np.isfinite(self.adtv).mean(), time.time() - t0), flush=True)

    def _causal_liquidity(self):
        """ADTV20 与日波动率: 严格只用 e-1 及之前的信息, e 日成交额不可用于 e 日开盘委托.

        ADTV20[e] = mean(amount[t] for t in [e-20, e-1] if amount[t] 有效且 >0)
                    要求窗口内至少 5 个有效交易日, 否则标记 NaN (容量 = 0).
        sigma[e]  = 过去 60 个网格日 open-to-open 收益率标准差 (至少 20 个有效值),
                    同样不含 e 日及之后的收益. NaN 时用当日横截面中位数替代(亦为时点可知).
        """
        N, M = self.N, self.M
        a = np.where(np.isfinite(self.amt) & (self.amt > 0), self.amt, 0.0)
        ok_a = (a > 0).astype(np.float64)
        ca = np.cumsum(a, axis=0)
        cc_a = np.cumsum(ok_a, axis=0)
        adtv = np.full((N, M), np.nan, dtype=np.float64)
        for e in range(21, N):
            s = ca[e - 1] - ca[e - 21]
            n = cc_a[e - 1] - cc_a[e - 21]
            with np.errstate(all="ignore"):
                v = np.where(n >= 5, s / np.maximum(n, 1), np.nan)
            adtv[e] = v

        with np.errstate(all="ignore"):
            r = np.empty_like(self.ao)
            r[0] = np.nan
            r[1:] = self.ao[1:] / self.ao[:-1] - 1.0
        ok_r = np.isfinite(r) & (np.abs(r) < 0.5)
        r0 = np.where(ok_r, r, 0.0)
        c1 = np.cumsum(r0, axis=0)
        c2 = np.cumsum(r0 * r0, axis=0)
        cn = np.cumsum(ok_r.astype(np.float64), axis=0)
        # 数据缺失 ≠ 流动性为零: 用"截至 e-1 最后一次有效 ADTV20"前向填充,
        # 且最长只保留 60 个网格日. 全程只用 e 之前的信息, 不引入任何未来数据.
        # 停牌/长期无成交后复牌时, 这是唯一时点可知的流动性估计; 超过 60 日
        # 视为无可靠流动性估计, 该日容量保守置 0 并单列统计.
        adtv_ff = np.full((N, M), np.nan, dtype=np.float64)
        age = np.full((N, M), np.nan, dtype=np.float32)
        last = np.full(M, np.nan, dtype=np.float64)
        stale = np.full(M, 9999, dtype=np.int32)
        for e in range(N):
            ok = np.isfinite(adtv[e]) & (adtv[e] > 0)
            last[ok] = adtv[e][ok]
            stale[ok] = 0
            stale[~ok] += 1
            use = np.isfinite(last) & (stale <= 60)
            adtv_ff[e] = np.where(use, last, np.nan)
            age[e] = np.where(use, stale, np.nan)
        self.adtv_raw, self.adtv_age = adtv, age
        adtv = adtv_ff

        sigma = np.full((N, M), np.nan, dtype=np.float64)
        for e in range(21, N):
            w = 60 if e >= 61 else e
            s1 = c1[e - 1] - c1[e - 1 - w]
            s2 = c2[e - 1] - c2[e - 1 - w]
            n = cn[e - 1] - cn[e - 1 - w]
            with np.errstate(all="ignore"):
                var = s2 / np.maximum(n, 1) - (s1 / np.maximum(n, 1)) ** 2
                sd = np.sqrt(np.maximum(var, 0.0))
            sigma[e] = np.where(n >= 20, sd, np.nan)
        med = np.empty(N)
        for e in range(N):
            row = sigma[e]
            med[e] = np.nanmedian(row) if np.isfinite(row).any() else 0.025
        return adtv, sigma, med

    def tradability(self):
        """V5 的开盘可买/可卖矩阵 (与 TrustedExecutor status_mode='v5' 一致)."""
        suspended = self.no_trade | (~self.valid_px)
        tradable = self.valid_px & ~suspended
        has_lu = np.isfinite(self.lu) & (self.lu > 0)
        has_ld = np.isfinite(self.ld) & (self.ld > 0)
        up_open = has_lu & (self.openp >= self.lu - EPS) & tradable
        dn_open = has_ld & (self.openp <= self.ld + EPS) & tradable
        can_sell = tradable & ~dn_open       # 开盘封涨停不阻卖
        can_buy = tradable & ~up_open        # 开盘封跌停不阻买
        return suspended, up_open, dn_open, can_sell, can_buy

    def fcr_masks(self):
        data_missing = (self.status == S_DATA_MISSING)
        delist_risk = (self.status == S_DELIST_PROC) | (self.delist_stage == 2)
        return data_missing, delist_risk


class CapacityExecutor:
    """资金规模直接改变成交结果的 V5 执行器.

    capital      : 初始资金(元). pos/cash 全部以元计价, 便于与 ADTV20 直接比较.
    p_max        : 单笔委托占 ADTV20 的最大参与率上限.
    open_share   : 假定能以开盘价成交的"当日成交额占比"上限 (近似! 见备忘录).
                   有效容量 = ADTV20_known * min(p_max, open_share * 状态乘数).
    Y            : 平方根冲击系数 impact_bps = Y * sigma_daily_bps * sqrt(实际参与率).
    funding      : 'compound' = 盈亏复投, 目标市值 = 当期权益/K;
                   'fixed'    = 目标市值 = min(权益, 初始资金)/K, 盈利以现金闲置.
    st_mult / delist_mult : ST / 退市流程股票开盘流动性的保守折扣.
    unlimited    : 校准档. 关闭全部容量约束和冲击, 应逐笔复现 P1.5-C 的 V5 主口径.
    zero_status  : 反事实归因用, 把这些状态码当日的盯市收益强制归零 (不改交易决策).
    """

    def __init__(self, env: Env, capital=1e5, p_max=0.05, open_share=0.10, Y=1.0,
                 funding="compound", st_mult=0.5, delist_mult=0.25,
                 unlimited=False, zero_status=(), min_comm_yuan=0.0,
                 spread_override=None, base_bps=None, track_pnl=False,
                 save_orders=True, strict=True, snapshot_dates=()):
        self.env = env
        self.N, self.M = env.N, env.M
        self.dates = env.dates
        self.capital = float(capital)
        self.p_max = float(p_max)
        self.open_share = float(open_share)
        self.Y = float(Y)
        self.funding = funding
        self.st_mult = float(st_mult)
        self.delist_mult = float(delist_mult)
        self.unlimited = bool(unlimited)
        self.zero_status = tuple(zero_status)
        self.min_comm_yuan = float(min_comm_yuan)
        self.spread_override = spread_override
        self.track_pnl = bool(track_pnl)
        self.save_orders = bool(save_orders)
        self.strict = bool(strict)
        # D+ 审计用: 指定日期的逐股持仓快照, 纯观测, 不参与任何交易决策或估值
        self.snapshot_dates = {str(x) for x in snapshot_dates}

        self.ao, self.amt, self.adtv = env.ao, env.amt, env.adtv
        self.adtv_age = env.adtv_age
        self.sigma, self.sigma_med = env.sigma, env.sigma_med
        self.status, self.delist_stage = env.status, env.delist_stage
        self.no_trade = env.no_trade
        self.valid_px = env.valid_px
        (self.suspended, self.up_open, self.dn_open,
         self.can_sell, self.can_buy) = env.tradability()
        self.data_missing, self.delist_risk = env.fcr_masks()
        self.mask = env.mask
        self.K = FROZEN["K"]
        self.stale_limit = FROZEN["STALE_LIMIT"]
        self.clip_hi = FROZEN["CLIP_HI"]
        self.base_bps = (FROZEN["COST_BPS_SIDE"] if base_bps is None else float(base_bps))
        # 最小可提交委托(元). 校准档关闭, 避免改变冻结 V5 的逐笔一致; 真实资金档按
        # A 股 100 股整手的量级设 500 元上限, 但不高于初始资金的 1bp.
        self.min_ticket = 0.0 if self.unlimited else min(500.0, 1e-4 * self.capital)
        self.stamp_switch = pd.Timestamp("2023-08-28")
        self._stamp_by_day = (env.dates < self.stamp_switch).astype(np.uint8)

        # 反事实归因的逐日零收益掩码 (只在 track / counterfactual 时构造)
        self.zero_mask = None
        if self.zero_status:
            self.zero_mask = np.isin(self.status, self.zero_status)

    # ---------------------------------------------------------------- 成本
    def _fee(self, e, c, side, v, used=None):
        """返回 (总费用元, 各分项元, impact_bps, 实际参与率). 只在已成交金额上计费."""
        comm = 3.0
        transfer = 0.2
        stamp = 0.0 if side == "B" else (10.0 if self._stamp_by_day[e] else 5.0)
        if self.base_bps <= 0:
            comm = transfer = stamp = spread = 0.0   # 零成本对照档
        else:
            spread = (self.spread_override if self.spread_override is not None
                      else self.base_bps - comm - transfer - stamp)
        a = self.adtv[e, c]
        part = v / a if (np.isfinite(a) and a > 0) else 0.0
        part = float(min(max(part, 0.0), 1.0))
        sd = self.sigma[e, c]
        if not np.isfinite(sd) or sd <= 0:
            sd = self.sigma_med[e]
        impact_bps = 0.0 if (self.unlimited or self.Y == 0.0 or part <= 0) else \
            self.Y * sd * 1e4 * np.sqrt(part)
        bps = {"佣金": comm, "过户费": transfer, "印花税": stamp,
               "价差及其他": spread, "市场冲击": impact_bps}
        total_bps = sum(bps.values())
        yuan = {k: v * b / 1e4 for k, b in bps.items()}
        fee = sum(yuan.values())
        if self.min_comm_yuan > 0 and fee < self.min_comm_yuan:
            delta = self.min_comm_yuan - fee
            yuan["佣金"] += delta
            fee = self.min_comm_yuan
        return fee, yuan, impact_bps, part, total_bps

    def _cap(self, e, c, used=None):
        """该股票当日开盘剩余可成交金额(元). 只用 e 日前已知的 ADTV20.

        used: 当日本股票已消耗的委托金额, 保证同一 (e,c) 的多笔委托不会重复占用
              同一份开盘流动性. 这是"逐笔计算委托金额/参与率"的必要条件.
        """
        if self.unlimited:
            return np.inf, np.inf, 1.0
        a = self.adtv[e, c]
        if not (np.isfinite(a) and a > 0):
            return 0.0, a, 0.0
        mult = 1.0
        st = int(self.status[e, c])
        if st == S_ST:
            mult = self.st_mult
        elif st == S_DELIST_PROC or self.delist_stage[e, c] == 2:
            mult = self.delist_mult
        eff_share = self.open_share * mult
        cap = a * min(self.p_max, eff_share)
        if used is not None:
            cap = max(cap - used.get(c, 0.0), 0.0)
        return cap, a, eff_share

    # ---------------------------------------------------------------- 主循环
    def simulate(self, score=None, pairs=None, bench=None):
        env = self.env
        score = env.score if score is None else score
        pairs = env.PAIRS if pairs is None else pairs
        bench = env.BENCH if bench is None else bench
        K = self.K
        exec_map = {}
        for a, (i0, i1) in enumerate(pairs):
            e = i0 + 1
            if e < self.N and i1 + 1 < self.N:
                exec_map[e] = (i0, a)
        exec_days = sorted(exec_map)
        start_e, end_e = exec_days[0], exec_days[-1]

        pos = np.zeros(self.M, dtype=np.float64)
        cash = self.capital
        orders = []
        pnl_rows = []
        diag = {"filled": 0, "partial": 0, "blocked": 0, "retry_filled": 0,
                "buy_blocked": 0, "sell_blocked": 0, "cap_blocked": 0,
                "desired_notional": 0.0, "traded_notional": 0.0,
                "unfilled_notional": 0.0, "pending_max": 0,
                "backlog_equity_max": 0.0, "backlog_ratio_max": 0.0,
                "max_delay_sell_days": 0, "delay_days": [],
                "delay_cap_days": [], "delay_limit_days": [],
                "backlog_cap_equity_max": 0.0,
                "cost_佣金": 0.0, "cost_过户费": 0.0, "cost_印花税": 0.0,
                "cost_价差及其他": 0.0, "cost_市场冲击": 0.0, "cost_total": 0.0,
                "max_accounting_err": 0.0, "turnover": 0.0,
                "frozen_n": 0, "frozen_value_end": 0.0, "unfilled_target": 0,
                "st_pnl_by_code": {}, "st_days_by_code": {},
                "pnl_by_code_all": {}, "held_ever_st": set()}
        by_status = {}
        snap_rows = []
        # E0 只读观测: 一笔"原始调仓意图" = 某个调仓日针对某只股票首次产生的
        # 买入/卖出决定. 之后所有重试(含跨调仓期)都挂在同一个 intent_id 下,
        # 用于统计 0/1/3/10 日目标委托完成率. 不参与任何交易决策或估值.
        intent_rows, intent_map, intent_seq = [], {}, [0]
        pending = {}          # c -> dict(side, remaining/tgt, first, period, kind)
        block_start = {}       # c -> 首次未完全卖出日 (跨调仓期持续)
        block_reason = {}      # c -> 'capacity' | 'limit' (区分容量延迟与停牌/涨跌停延迟)
        held_since = {}
        stale, frozen = {}, set()
        equity_by_exec, turn_by_exec, nh_by_exec = {}, {}, {}
        exposure_by_exec = {}
        fees_by_day = {}       # 逐日已付费用(元), 用于构造"零成本对照"收益序列

        def mark_delay(c, e):
            if c not in block_start:
                return
            delay = e - block_start[c]
            diag["max_delay_sell_days"] = max(diag["max_delay_sell_days"], delay)
            diag["delay_days"].append(delay)
            key = "delay_cap_days" if block_reason.get(c) == "capacity" else "delay_limit_days"
            diag[key].append(delay)

        def new_intent(e, c, side, kind, amount):
            """新建一笔原始调仓意图, 返回整数 id. 纯观测."""
            i = int(intent_seq[0]); intent_seq[0] += 1
            rec = {"intent_id": i, "first_day": int(e),
                   "date": str(self.dates[e].date()),
                   "period": int(exec_map.get(e, (None, -1))[1]),
                   "code": int(c), "code6": str(env.codes[c]),
                   "side": side, "kind": kind,
                   "original_desired": float(amount), "cancelled": False}
            intent_rows.append(rec); intent_map[i] = rec
            return i

        def intent_for(e, c, side, kind, amount):
            """已有同向积压挂单时沿用原 intent_id (跨调仓期不重置分母)."""
            p = pending.get(c)
            if (p is not None and p.get("side") == side
                    and p.get("intent_id") in intent_map):
                return p["intent_id"]
            return new_intent(e, c, side, kind, amount)

        def record_exposure(e):
            eq = cash + float(pos.sum())
            cols = np.where(pos > 1e-9)[0]
            w_data = w_delist = w_restrict = w_tradable = 0.0
            n_data = n_delist = n_restrict = 0
            w_unc = w_rest = 0.0
            for c in cols:
                v = pos[c]
                if c in frozen:
                    w_unc += v
                elif not self.can_sell[e, c]:
                    w_rest += v
                if self.delist_risk[e, c]:
                    w_delist += v; n_delist += 1
                elif self.data_missing[e, c]:
                    w_data += v; n_data += 1
                elif c in frozen or not self.can_sell[e, c]:
                    w_restrict += v; n_restrict += 1
                else:
                    w_tradable += v
            w_tradable += cash
            frozen_all = w_data + w_delist + w_restrict
            backlog = sum(pos[c] for c in pending if pending[c]["side"] == "S")
            backlog_cap = sum(pos[c] for c in pending
                              if pending[c]["side"] == "S"
                              and block_reason.get(c) == "capacity")
            diag["backlog_equity_max"] = max(diag["backlog_equity_max"], backlog)
            diag["backlog_cap_equity_max"] = max(diag["backlog_cap_equity_max"], backlog_cap)
            diag["backlog_ratio_max"] = max(diag["backlog_ratio_max"],
                                            backlog / eq if eq > 0 else 0.0)
            if self.snapshot_dates and str(self.dates[e].date()) in self.snapshot_dates:
                for c in cols:
                    prev_ao = float(self.ao[e - 1, c]) if e > 0 else np.nan
                    v = float(pos[c])
                    snap_rows.append({
                        "date": str(self.dates[e].date()), "day": int(e),
                        "col": int(c), "code": str(env.codes[c]),
                        "pos_value": v, "weight": v / eq if eq > 0 else np.nan,
                        "status_code": int(self.status[e, c]),
                        "can_sell": bool(self.can_sell[e, c]),
                        "dn_open": bool(self.dn_open[e, c]),
                        "up_open": bool(self.up_open[e, c]),
                        "suspended": bool(self.suspended[e, c]),
                        "valid_px": bool(self.valid_px[e, c]),
                        "data_missing": bool(self.data_missing[e, c]),
                        "delist_risk": bool(self.delist_risk[e, c]),
                        "frozen": bool(c in frozen),
                        "pending_sell": bool(pending.get(c, {}).get("side") == "S"),
                        "open_px": float(self.ao[e, c]),
                        "open_ret": float(self.ao[e, c] / prev_ao - 1.0)
                                    if np.isfinite(prev_ao) and prev_ao > 0 else np.nan,
                        "adtv20_known": float(self.adtv[e, c]),
                    })
            exposure_by_exec[e] = {
                "equity": float(eq), "cash": float(cash),
                "fcr": float(w_unc / eq) if eq > 0 else 0.0,
                "rcr": float(w_rest / eq) if eq > 0 else 0.0,
                "fcr_data": float(w_data / eq) if eq > 0 else 0.0,
                "fcr_delist": float(w_delist / eq) if eq > 0 else 0.0,
                "fcr_restrict": float(w_restrict / eq) if eq > 0 else 0.0,
                "fcr_total": float(frozen_all / eq) if eq > 0 else 0.0,
                "tradable": float(w_tradable), "backlog": float(backlog),
                "n_data": n_data, "n_delist": n_delist, "n_restrict": n_restrict,
            }

        def log(e, c, side, status, reason, desired, filled, before, fee,
                yuan, impact_bps, part, cap, a, eff_share, intent_id=None):
            if not self.save_orders:
                return
            eq_after = cash + pos.sum()
            iid = "" if intent_id is None else int(intent_id)
            orig = (float(intent_map[intent_id]["original_desired"])
                    if intent_id is not None and intent_id in intent_map else np.nan)
            orders.append({
                "intent_id": iid, "intent_original": orig,
                "day": int(e), "date": str(self.dates[e].date()),
                "code": int(c), "side": side, "status": status, "reason": reason,
                "status_code": int(self.status[e, c]),
                "desired": float(desired), "filled": float(filled),
                "price": float(self.ao[e, c]) if np.isfinite(self.ao[e, c]) else np.nan,
                "adtv20_known": float(a) if np.isfinite(a) else np.nan,
                "cap_open": float(cap) if np.isfinite(cap) else np.nan,
                "open_share_eff": float(eff_share),
                "participation": float(part), "impact_bps": float(impact_bps),
                "fee_佣金": yuan["佣金"], "fee_过户费": yuan["过户费"],
                "fee_印花税": yuan["印花税"], "fee_价差及其他": yuan["价差及其他"],
                "fee_市场冲击": yuan["市场冲击"], "fee_total": float(fee),
                "cash_before": float(before[0]), "cash_after": float(cash),
                "equity_before": float(before[1]), "equity_after": float(eq_after),
                "pos_sum_before": float(before[2]), "pos_sum_after": float(pos.sum()),
            })

        def attempt(e, c, side, desired, status_ok, reason,
                    before_kind="filled", used=None, intent_id=None):
            """执行一次委托尝试, 返回实际成交金额."""
            nonlocal cash
            used = {} if used is None else used
            desired = float(max(desired, 0.0))
            if desired <= 1e-9:
                return 0.0
            before = (cash, cash + pos.sum(), pos.sum())
            if not status_ok:
                diag["blocked"] += 1
                diag["desired_notional"] += desired
                diag["unfilled_notional"] += desired
                log(e, c, side, "blocked", reason, desired, 0.0, before,
                    0.0, {k: 0.0 for k in ("佣金", "过户费", "印花税", "价差及其他", "市场冲击")},
                    0.0, 0.0, 0.0, np.nan, 1.0, intent_id)
                return 0.0
            cap, a, eff_share = self._cap(e, c, used)
            cap_limited = desired > cap
            v = min(desired, cap)
            cash_limited = False
            if side == "B":
                # 现金约束: 费用随成交额变化, 迭代两次保证 cash >= 0
                for _ in range(2):
                    if v <= 1e-9:
                        break
                    fee, _, _, _, total_bps = self._fee(e, c, side, v)
                    lim = max(cash, 0.0) / (1.0 + total_bps / 1e4)
                    if v > lim * (1 + 1e-12):
                        v = lim
                        cash_limited = True
                    else:
                        break
            v = max(min(v, desired), 0.0)
            if side == "B" and 0 < v < self.min_ticket:
                v = 0.0
                cash_limited = "tiny"
            diag["desired_notional"] += desired
            if v <= 1e-9:
                diag["blocked"] += 1
                diag["cap_blocked"] += 1
                diag["unfilled_notional"] += desired
                why = ("低于最小委托单位" if cash_limited == "tiny"
                       else "现金不足" if cash_limited and not cap_limited
                       else "容量不足/无可用ADTV20")
                log(e, c, side, "blocked", why, desired, 0.0,
                    before, 0.0, {k: 0.0 for k in ("佣金", "过户费", "印花税", "价差及其他", "市场冲击")},
                    0.0, 0.0, cap, a, eff_share, intent_id)
                return 0.0
            fee, yuan, impact_bps, part, _ = self._fee(e, c, side, v)
            if side == "S":
                pos[c] -= v
                cash += v
            else:
                pos[c] += v
                cash -= v
            cash -= fee
            for k, val in yuan.items():
                diag["cost_" + k] += val
            diag["cost_total"] += fee
            fees_by_day[e] = fees_by_day.get(e, 0.0) + fee
            diag["traded_notional"] += v
            diag["unfilled_notional"] += max(desired - v, 0.0)
            diag["turnover"] += v
            used[c] = used.get(c, 0.0) + v
            if status_ok and before_kind == "retry_filled":
                diag["retry_filled"] += 1 if v >= desired * (1 - FILL_TOL) else 0
            full = v >= desired * (1 - FILL_TOL)
            st = "filled" if (full and before_kind != "retry_filled") else (
                "retry_filled" if full else "partial")
            if full:
                diag["filled"] += 1
            else:
                diag["partial"] += 1
            eq_before = before[1]
            err = abs((cash + pos.sum()) - (eq_before - fee))
            diag["max_accounting_err"] = max(diag["max_accounting_err"], err)
            if self.strict and err > 1e-6:
                raise AssertionError("会计恒等式破裂 e=%d c=%d err=%.3e" % (e, c, err))
            log(e, c, side, st, reason, desired, v, before, fee, yuan,
                impact_bps, part, cap, a, eff_share, intent_id)
            sc = int(self.status[e, c])
            key = (sc, side)
            d = by_status.setdefault(key, {"desired": 0.0, "filled": 0.0,
                                           "n": 0, "partial": 0, "blocked": 0})
            d["desired"] += desired
            d["filled"] += v
            d["n"] += 1
            d["partial"] += 0 if full else 1
            d["blocked"] += 0 if v > 0 else 1
            return v

        # ------------------------------------------------------------ 逐日循环
        for e in range(start_e, end_e + 1):
            used = {}            # 当日各股票已消耗的开盘流动性(元), 每笔委托递减
            # (1) 盯市: 与 V5 完全一致的 open-to-open, 无效价 -> 收益 0
            if e > start_e:
                cols = np.array([c for c in np.where(pos > 1e-9)[0]], dtype=int)
                if len(cols):
                    prev = pos[cols].copy()
                    with np.errstate(all="ignore"):
                        r = self.ao[e, cols] / self.ao[e - 1, cols] - 1.0
                    good = (np.isfinite(r) & np.isfinite(self.ao[e, cols])
                            & (self.ao[e, cols] > 0))
                    r = np.where(good, r, 0.0)
                    r = np.clip(r, -0.99, self.clip_hi)
                    if self.zero_mask is not None:
                        r = np.where(self.zero_mask[e, cols], 0.0, r)
                    pos[cols] *= (1.0 + r)
                    if self.track_pnl:
                        buckets = np.zeros(8)
                        st_n, st_val, st_sellable_val, st_oneword_val = 0, 0.0, 0.0, 0.0
                        for j, c in enumerate(cols):
                            sc = int(self.status[e, c])
                            buckets[min(sc, 5)] += prev[j] * r[j]
                            diag["pnl_by_code_all"][int(c)] = (
                                diag["pnl_by_code_all"].get(int(c), 0.0) + prev[j] * r[j])
                            if sc == S_ST:
                                diag["held_ever_st"].add(int(c))
                                st_n += 1
                                st_val += prev[j]
                                diag["st_pnl_by_code"][int(c)] = (
                                    diag["st_pnl_by_code"].get(int(c), 0.0) + prev[j] * r[j])
                                diag["st_days_by_code"][int(c)] = (
                                    diag["st_days_by_code"].get(int(c), 0) + 1)
                                if self.can_sell[e, c]:
                                    st_sellable_val += prev[j]
                                    buckets[6] += prev[j] * r[j]
                                else:
                                    buckets[7] += prev[j] * r[j]
                                if self.env.one_word_dn[e, c] or self.env.one_word_up[e, c]:
                                    st_oneword_val += prev[j]
                        pnl_rows.append({"day": int(e), "date": str(self.dates[e].date()),
                                         **{"pnl_%d" % k: float(buckets[k]) for k in range(6)},
                                         "st_pnl_sellable": float(buckets[6]),
                                         "st_pnl_nonsellable": float(buckets[7]),
                                         "st_pos_n": st_n, "st_pos_value": float(st_val),
                                         "st_sellable_value": float(st_sellable_val),
                                         "st_oneword_value": float(st_oneword_val),
                                         "st_codes": ",".join(
                                             str(env.codes[int(c)]) for c in cols
                                             if self.status[e, c] == S_ST)})
                    for c in cols[good]:
                        stale[c] = 0
                        frozen.discard(c)
                    for c in cols[~good]:
                        stale[c] = stale.get(c, 0) + 1
                        if stale[c] >= self.stale_limit:
                            frozen.add(c)

            # (2) 非调仓日重试未成交延续委托 (买单与卖单都重试).
            #     调仓日不预先重试: V5 在 3a 一次性按目标重新推导, 且 tgt_v 是在
            #     当日任何手续费扣除之前以 eq/K 计算的. 若提前重试会先扣费,
            #     使目标市值相差 fee/K, 破坏与冻结 V5 的逐笔一致性.
            for c in (list(pending) if e not in exec_map else []):
                p = pending[c]
                if p["side"] == "S":
                    if pos[c] <= 1e-9:
                        pending.pop(c, None); block_start.pop(c, None)
                        block_reason.pop(c, None)
                        held_since.pop(c, None)
                        continue
                    if not self.can_sell[e, c]:
                        continue
                    want = pos[c] if p.get("target", 0.0) <= 0 else max(pos[c] - p["target"], 0.0)
                    if want <= 1e-9:
                        pending.pop(c, None); block_start.pop(c, None)
                        continue
                    v = attempt(e, c, "S", want, True, "未成交卖单延续",
                               "retry_filled", used, p.get("intent_id"))
                    if v >= want * (1 - FILL_TOL):
                        mark_delay(c, e)
                        pending.pop(c, None); block_start.pop(c, None)
                        block_reason.pop(c, None)
                        held_since.pop(c, None)
                else:
                    rem = p["remaining"]
                    if rem <= 1e-9:
                        pending.pop(c, None)
                        continue
                    # 满仓且无可用现金时不可能提交买单, 保留挂单但不生成伪"受阻订单"
                    if cash <= max(self.min_ticket, 1e-9):
                        continue
                    if not self.can_buy[e, c]:
                        continue
                    v = attempt(e, c, "B", rem, True, "未成交买单延续",
                               "retry_filled", used, p.get("intent_id"))
                    p["remaining"] -= v
                    if p["remaining"] <= 1e-9:
                        pending.pop(c, None)

            # (3) 非调仓日结束
            if e not in exec_map:
                eq = cash + pos.sum()
                if self.strict:
                    err = abs(eq - (cash + pos.sum()))
                    diag["max_accounting_err"] = max(diag["max_accounting_err"], err)
                record_exposure(e)
                continue

            # (4) 调仓日: 目标完全由冻结的 V5 因子/股票池决定
            i0, a = exec_map[e]
            # 上一期遗留的 pending 在本期由目标重新推导, 避免同一笔委托被重复下单
            for c in list(pending):
                if pending[c]["side"] == "B":
                    iid = pending[c].get("intent_id")
                    if iid in intent_map:
                        intent_map[iid]["cancelled"] = True
                    pending.pop(c, None)
            ok = self.mask[i0] & np.isfinite(score[i0])
            if ok.sum() < FROZEN["MIN_UNIVERSE"]:
                equity_by_exec[e] = cash + pos.sum()
                turn_by_exec[e] = 0.0
                nh_by_exec[e] = int((pos > 1e-9).sum())
                record_exposure(e)
                continue
            idx = np.where(ok)[0]
            pick = idx[np.argsort(-score[i0][idx])[:K]]
            target = set(int(c) for c in pick)
            eq = cash + pos.sum()
            if self.funding == "fixed":
                tgt_v = min(eq, self.capital) / K
            else:
                tgt_v = eq / K
            turn = 0.0

            # 4a 非目标持仓: 能卖则卖(受容量限制), 受阻或不足进 pending
            for c in np.where(pos > 1e-9)[0]:
                if c in target or c in frozen:
                    continue
                if not self.can_sell[e, c]:
                    reason = ("停牌/无成交" if self.suspended[e, c] or self.no_trade[e, c]
                              else "开盘封跌停" if self.dn_open[e, c] else "无有效价格")
                    diag["sell_blocked"] += 1
                    diag["blocked"] += 1
                    diag["desired_notional"] += pos[c]
                    diag["unfilled_notional"] += pos[c]
                    before = (cash, cash + pos.sum(), pos.sum())
                    iid = intent_for(e, c, "S", "非目标调出", pos[c])
                    log(e, c, "S", "blocked", reason, pos[c], 0.0, before, 0.0,
                        {k: 0.0 for k in ("佣金", "过户费", "印花税", "价差及其他", "市场冲击")},
                        0.0, 0.0, 0.0, np.nan, 1.0, iid)
                    pending.setdefault(c, {"side": "S", "target": 0.0, "first": e,
                                           "period": a, "intent_id": iid})
                    block_start.setdefault(c, e)
                    block_reason.setdefault(c, "limit")
                    continue
                want = pos[c]
                iid = intent_for(e, c, "S", "非目标调出", want)
                v = attempt(e, c, "S", want, True, "非目标调出",
                            used=used, intent_id=iid)
                turn += v
                if v < want * (1 - FILL_TOL):
                    pending[c] = {"side": "S", "target": 0.0, "first": e,
                                  "period": a, "intent_id": iid}
                    block_start.setdefault(c, e)
                    block_reason.setdefault(c, "capacity")   # 只记首次阻塞原因
                else:
                    mark_delay(c, e)
                    pending.pop(c, None); block_start.pop(c, None)
                    block_reason.pop(c, None)
                    held_since.pop(c, None)

            # 4b 目标内减仓 (涨停不妨碍卖出)
            for c in target:
                if c in frozen or pos[c] <= tgt_v * (1 + 1e-9):
                    continue
                if not self.can_sell[e, c]:
                    diag["sell_blocked"] += 1
                    diag["blocked"] += 1
                    before = (cash, cash + pos.sum(), pos.sum())
                    diag["desired_notional"] += pos[c] - tgt_v
                    diag["unfilled_notional"] += pos[c] - tgt_v
                    iid = intent_for(e, c, "S", "再平衡减仓", pos[c] - tgt_v)
                    log(e, c, "S", "blocked", "减仓受阻", pos[c] - tgt_v, 0.0, before,
                        0.0, {k: 0.0 for k in ("佣金", "过户费", "印花税", "价差及其他", "市场冲击")},
                        0.0, 0.0, 0.0, np.nan, 1.0, iid)
                    continue
                want = pos[c] - tgt_v
                iid = intent_for(e, c, "S", "再平衡减仓", want)
                v = attempt(e, c, "S", want, True, "再平衡减仓",
                            used=used, intent_id=iid)
                turn += v
                if v < want * (1 - FILL_TOL):
                    pending.setdefault(c, {"side": "S", "target": tgt_v,
                                          "first": e, "period": a,
                                          "intent_id": iid})
                    block_start.setdefault(c, e)
                    block_reason.setdefault(c, "capacity")

            # 4c 目标内建仓/补仓
            wants = []
            for c in pick:
                c = int(c)
                if c in frozen or pos[c] >= tgt_v - 1e-9:
                    continue
                if not self.can_buy[e, c]:
                    reason = ("停牌/无成交" if self.suspended[e, c] or self.no_trade[e, c]
                              else "开盘封涨停" if self.up_open[e, c] else "无有效价格")
                    diag["buy_blocked"] += 1
                    diag["blocked"] += 1
                    diag["unfilled_target"] += 1
                    before = (cash, cash + pos.sum(), pos.sum())
                    diag["desired_notional"] += tgt_v - pos[c]
                    diag["unfilled_notional"] += tgt_v - pos[c]
                    iid = intent_for(e, c, "B", "建仓/补仓", tgt_v - pos[c])
                    log(e, c, "B", "blocked", reason, tgt_v - pos[c], 0.0, before,
                        0.0, {k: 0.0 for k in ("佣金", "过户费", "印花税", "价差及其他", "市场冲击")},
                        0.0, 0.0, 0.0, np.nan, 1.0, iid)
                    continue
                needs = tgt_v - pos[c]
                iid = intent_for(e, c, "B", "建仓/补仓", needs)
                cap, adv, sh = self._cap(e, c, used)
                if not self.unlimited and cap <= 1e-9:
                    needs = min(needs, cap)
                    if needs <= 1e-9:
                        diag["unfilled_target"] += 1
                        diag["desired_notional"] += tgt_v - pos[c]
                        diag["unfilled_notional"] += tgt_v - pos[c]
                        diag["cap_blocked"] += 1
                        diag["blocked"] += 1
                        before = (cash, cash + pos.sum(), pos.sum())
                        log(e, c, "B", "blocked", "容量不足/无ADTV20", tgt_v - pos[c],
                            0.0, before, 0.0,
                            {k: 0.0 for k in ("佣金", "过户费", "印花税", "价差及其他", "市场冲击")},
                            0.0, 0.0, cap, adv, sh, iid)
                        continue
                wants.append((c, needs, iid))
            if wants:
                need_tot = sum(v for _, v, _ in wants)
                scale = min(1.0, max(cash, 0.0) / max(need_tot * (1 + self.base_bps / 1e4), 1e-12))
                diag["unfilled_target"] += int(round(len(wants) * (1 - scale)))
                for c, need, iid in wants:
                    v = need * scale
                    if v <= max(1e-9, self.min_ticket):
                        continue
                    got = attempt(e, c, "B", v, True, "建仓/补仓",
                                  used=used, intent_id=iid)
                    turn += got
                    if got > 0:
                        held_since.setdefault(c, e)
                    if got < v * (1 - FILL_TOL):
                        pending[c] = {"side": "B", "remaining": v - got,
                                      "first": e, "period": a, "intent_id": iid}

            diag["pending_max"] = max(diag["pending_max"], len(pending))
            equity_by_exec[e] = cash + pos.sum()
            turn_by_exec[e] = turn / max(2.0 * eq, 1e-12)
            nh_by_exec[e] = int((pos > 1e-9).sum())
            record_exposure(e)

        # ---------------------------------------------------------------- 组装
        order_e = sorted(equity_by_exec)
        rows = []
        for a in range(len(order_e) - 1):
            e0, e1 = order_e[a], order_e[a + 1]
            fee_period = sum(v for d, v in fees_by_day.items() if e0 < d <= e1)
            base_eq = max(equity_by_exec[e0], 1e-12)
            rows.append({
                "date": self.dates[e0 - 1] if e0 - 1 >= 0 else self.dates[0],
                "net": equity_by_exec[e1] / base_eq - 1.0,
                "gross": equity_by_exec[e1] / base_eq - 1.0 + fee_period / base_eq,
                "bench": bench[a] if a < len(bench) else np.nan,
                "turn": turn_by_exec.get(e0, np.nan),
                "n_hold": nh_by_exec.get(e1, 0),
                "start_date": self.dates[e0], "end_date": self.dates[e1],
            })
        df = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
            columns=["net", "gross", "bench", "turn", "n_hold", "start_date", "end_date"])
        od = pd.DataFrame(orders) if orders else pd.DataFrame()
        df.attrs["orders"] = od
        df.attrs["intents"] = pd.DataFrame(intent_rows)
        df.attrs["equity"] = equity_by_exec
        df.attrs["diag"] = diag
        df.attrs["by_status"] = by_status
        exp = pd.DataFrame.from_dict(exposure_by_exec, orient="index").sort_index()
        exp.index.name = "day"
        exp["date"] = [str(self.dates[int(i)].date()) for i in exp.index]
        df.attrs["exposure"] = exp
        df.attrs["pnl"] = pd.DataFrame(pnl_rows)
        df.attrs["held_snapshots"] = pd.DataFrame(snap_rows)
        fz = [int(c) for c in frozen if pos[c] > 1e-9]
        diag["frozen_n"] = len(fz)
        diag["frozen_value_end"] = float(sum(pos[c] for c in fz))
        df.attrs["held_detail"] = [
            {"code": int(c), "first_buy": int(v),
             "age_days": int(end_e - v),
             "weight": float(pos[c] / max(cash + pos.sum(), 1e-12)),
             "frozen": c in frozen, "stale_days": int(stale.get(c, 0))}
            for c, v in held_since.items() if pos[c] > 1e-9]
        df.attrs["held_detail"].sort(key=lambda x: -x["age_days"])
        df.attrs["final_equity"] = cash + float(pos.sum())
        df.attrs["final_cash"] = float(cash)
        return df


def metrics(df, label=""):
    """统一评价口径: 完全复用 p1_engine.evaluate, 并补充 P1.5-D 的容量指标."""
    d = df.copy()
    if "gross" not in d:
        d["gross"] = d["net"]
    m = evaluate(d, label=label)
    dg = df.attrs["diag"]
    exp = df.attrs["exposure"]
    od = df.attrs["orders"]
    n_all = len(od)
    cnt = od["status"].value_counts().to_dict() if n_all else {}
    n_full = int(cnt.get("filled", 0)) + int(dg["retry_filled"])
    n_partial = int(cnt.get("partial", 0))
    n_blocked = int(cnt.get("blocked", 0))
    delay = np.asarray(dg["delay_days"], dtype=float)
    out = {
        "标签": label, "期数": m["periods"],
        "净CAGR%": round(m["cagr_net"] * 100, 4),
        "基准CAGR%": round(m["cagr_bench"] * 100, 4),
        "几何超额%": round(m["excess_geo"] * 100, 4),
        "IR": round(m["ir"], 4), "Sharpe": round(m["sharpe"], 4),
        "最大回撤%": round(m["maxDD"] * 100, 3),
        "年换手": round(m["turnover_annual"], 3),
        "成本拖累%": round(m["cost_drag"] * 100, 4),
        "平均持仓数": round(m["avg_hold_n"], 2),
        "期末权益": round(df.attrs["final_equity"], 2),
        "期末现金": round(df.attrs["final_cash"], 2),
        "订单数": n_all, "全额成交": int(cnt.get("filled", 0)),
        "部分成交": n_partial, "受阻": n_blocked,
        "重试全额成交": int(dg["retry_filled"]),
        "严格首次成交率%": round(100 * int(cnt.get("filled", 0)) / n_all, 4) if n_all else np.nan,
        "含重试成交率%": round(100 * n_full / n_all, 4) if n_all else np.nan,
        "金额成交率%": round(100 * dg["traded_notional"] / dg["desired_notional"], 4)
                     if dg["desired_notional"] > 0 else np.nan,
        "委托总额元": round(dg["desired_notional"], 2),
        "成交总额元": round(dg["traded_notional"], 2),
        "未成交总额元": round(dg["unfilled_notional"], 2),
        "最大延迟卖出天数": int(dg["max_delay_sell_days"]),
        "延迟卖出天数中位": round(float(np.median(delay)), 1) if len(delay) else 0.0,
        "延迟卖出天数p95": round(float(np.percentile(delay, 95)), 1) if len(delay) else 0.0,
        "容量导致最大延迟天数": int(max(dg["delay_cap_days"] or [0])),
        "容量导致延迟天数p95": round(float(np.percentile(dg["delay_cap_days"], 95)), 1)
                          if dg["delay_cap_days"] else 0.0,
        "停牌涨跌停导致最大延迟天数": int(max(dg["delay_limit_days"] or [0])),
        "最大积压持仓数": int(dg["pending_max"]),
        "最大积压权益元": round(dg["backlog_equity_max"], 2),
        "其中容量积压权益元": round(dg["backlog_cap_equity_max"], 2),
        "最大积压权益占比%": round(100 * dg["backlog_ratio_max"], 4),
        "FCR_data期末%": round(100 * exp["fcr_data"].iloc[-1], 4),
        "FCR_delist期末%": round(100 * exp["fcr_delist"].iloc[-1], 4),
        "FCR_restrict期末%": round(100 * exp["fcr_restrict"].iloc[-1], 4),
        "FCR_total期末%": round(100 * exp["fcr_total"].iloc[-1], 4),
        "FCR_total均值%": round(100 * exp["fcr_total"].mean(), 4),
        "FCR_total峰值%": round(100 * exp["fcr_total"].max(), 4),
        "会计最大误差": float(f"{dg['max_accounting_err']:.3e}"),
        "费用_佣金元": round(dg["cost_佣金"], 2),
        "费用_过户费元": round(dg["cost_过户费"], 2),
        "费用_印花税元": round(dg["cost_印花税"], 2),
        "费用_价差及其他元": round(dg["cost_价差及其他"], 2),
        "费用_市场冲击元": round(dg["cost_市场冲击"], 2),
        "费用_合计元": round(dg["cost_total"], 2),
        "期末冻结持仓数": int(dg["frozen_n"]),
        "期末冻结市值元": round(dg["frozen_value_end"], 2),
    }
    return out
