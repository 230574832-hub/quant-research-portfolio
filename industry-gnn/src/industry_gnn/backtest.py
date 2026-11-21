from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostScenario:
    commission_bps: float
    slippage_bps: float
    stamp_duty_policy: str = "fixed"
    stamp_duty_sell_bps: float | None = None

    @classmethod
    def from_dict(cls, values: dict) -> "CostScenario":
        return cls(
            commission_bps=float(values.get("commission_bps", 0.0)),
            slippage_bps=float(values.get("slippage_bps", 0.0)),
            stamp_duty_policy=str(values.get("stamp_duty_policy", "fixed")),
            stamp_duty_sell_bps=(
                None if values.get("stamp_duty_sell_bps") is None else float(values["stamp_duty_sell_bps"])
            ),
        )

    @property
    def buy_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10_000.0

    def stamp_duty_bps(self, date, schedule: list[dict] | None = None) -> float:
        if self.stamp_duty_policy == "fixed":
            return float(self.stamp_duty_sell_bps or 0.0)
        if self.stamp_duty_policy != "historical":
            raise ValueError(f"Unknown stamp_duty_policy: {self.stamp_duty_policy}")
        timestamp = pd.Timestamp(date).normalize()
        for interval in schedule or []:
            start = pd.Timestamp(interval["effective_start"])
            end = pd.Timestamp(interval["effective_end"]) if interval.get("effective_end") else pd.Timestamp.max
            if start <= timestamp <= end:
                return float(interval["stamp_duty_sell_bps"])
        raise ValueError(f"No historical stamp-duty interval covers {timestamp.date()}.")

    def sell_rate(self, date, schedule: list[dict] | None = None) -> float:
        return (self.commission_bps + self.slippage_bps + self.stamp_duty_bps(date, schedule)) / 10_000.0


def _cost_components(notional: float, costs: CostScenario, side: str, date, schedule):
    commission = notional * costs.commission_bps / 10_000.0
    slippage = notional * costs.slippage_bps / 10_000.0
    duty_bps = costs.stamp_duty_bps(date, schedule) if side == "SELL" else 0.0
    stamp = notional * duty_bps / 10_000.0
    return commission, stamp, slippage, duty_bps


def run_execution_backtest(
    model_predictions: pd.DataFrame,
    execution_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    top_k: int,
    scenario_name: str,
    scenario: dict,
    stamp_duty_schedule: list[dict] | None = None,
    initial_cash: float = 1_000_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long-only weekly Top-K simulation with point-in-time rankings and blocked-trade carry-forward.

    The target list is selected only from signal-date predictions. Entry-day quotes and tradeability
    decide execution after ranking; unavailable/locked Top-K buys are logged as BLOCKED and are not
    silently replaced by lower-ranked names. Fractional adjusted shares isolate signal quality from
    lot-size effects. Historical stamp duty is selected by execution date when configured.
    """
    predictions = model_predictions.loc[model_predictions["split"] == "test"].copy()
    if predictions.empty:
        raise ValueError("No test predictions available for backtesting.")
    for column in ["signal_date", "entry_date", "exit_date"]:
        predictions[column] = pd.to_datetime(predictions[column])
    execution = execution_prices.copy()
    execution["trade_date"] = pd.to_datetime(execution["trade_date"])
    benchmark = benchmark_prices.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = benchmark.sort_values("date")
    benchmark["benchmark_return"] = benchmark["benchmark_open"].pct_change(fill_method=None)
    benchmark_return_map = benchmark.set_index("date")["benchmark_return"].to_dict()

    start = predictions["entry_date"].min()
    end = max(predictions["exit_date"].max(), predictions["entry_date"].max())
    calendar = pd.DatetimeIndex(sorted(execution.loc[execution["trade_date"].between(start, end), "trade_date"].unique()))
    if not len(calendar):
        raise RuntimeError("No execution dates cover the test prediction interval.")
    investment_start = calendar[0]
    execution_by_date = {date: section.set_index("ts_code") for date, section in execution.groupby("trade_date", sort=True)}
    predictions_by_entry = {date: section for date, section in predictions.groupby("entry_date", sort=True)}
    costs = CostScenario.from_dict(scenario)

    cash = float(initial_cash)
    shares: dict[str, float] = defaultdict(float)
    marks: dict[str, float] = {}
    previous_equity = initial_cash
    value_rows = []
    trade_rows = []
    model_name = str(predictions["model"].iloc[0])

    def record_trade(date, code, side, status, notional, reason, components=(0.0, 0.0, 0.0, 0.0)):
        commission, stamp, slippage, duty_bps = components
        trade_rows.append({
            "date": date, "model": model_name, "top_k": int(top_k), "cost_scenario": scenario_name,
            "ts_code": code, "side": side, "status": status, "notional": float(notional),
            "commission_cost": float(commission), "stamp_duty_cost": float(stamp),
            "slippage_cost": float(slippage), "applied_stamp_duty_bps": float(duty_bps),
            "cost": float(commission + stamp + slippage), "reason": reason,
        })

    for date in calendar:
        day = execution_by_date.get(date, pd.DataFrame())
        if not day.empty:
            for code, row in day.iterrows():
                price = row.get("adj_open")
                if pd.notna(price) and price > 0:
                    marks[str(code)] = float(price)
        marked_holdings = sum(quantity * marks.get(code, 0.0) for code, quantity in shares.items())
        pretrade_equity = cash + marked_holdings
        day_cost = 0.0
        traded_notional = 0.0

        if date in predictions_by_entry:
            candidates = predictions_by_entry[date].sort_values(["y_pred", "ts_code"], ascending=[False, True])
            targets = set(candidates.head(int(top_k))["ts_code"].astype(str))
            desired_value = pretrade_equity / len(targets) if targets else 0.0

            # Sales happen first. Untradeable names stay as locked holdings.
            for code in list(shares):
                if shares[code] <= 0 or code not in marks:
                    continue
                current_value = shares[code] * marks[code]
                wanted = desired_value if code in targets else 0.0
                sell_notional = max(0.0, current_value - wanted)
                if sell_notional <= 1e-8:
                    continue
                sellable = False
                if not day.empty and code in day.index:
                    quote = day.loc[code]
                    if isinstance(quote, pd.DataFrame):
                        quote = quote.iloc[-1]
                    sellable = bool(quote.get("sellable", False))
                if not sellable:
                    record_trade(date, code, "SELL", "BLOCKED", sell_notional,
                                 "suspended_or_open_at_lower_limit_or_missing_quote")
                    continue
                components = _cost_components(sell_notional, costs, "SELL", date, stamp_duty_schedule)
                fee = sum(components[:3])
                shares[code] -= sell_notional / marks[code]
                cash += sell_notional - fee
                day_cost += fee
                traded_notional += sell_notional
                record_trade(date, code, "SELL", "EXECUTED", sell_notional, "rebalance", components)
                if shares[code] <= 1e-12:
                    shares.pop(code, None)

            # Compute cash scaling only for executable targets; blocked Top-K names remain explicit logs.
            executable_needs = {}
            for code in sorted(targets):
                current_value = shares.get(code, 0.0) * marks.get(code, 0.0)
                requested = max(0.0, desired_value - current_value)
                if requested <= 1e-8:
                    continue
                quote = None
                if not day.empty and code in day.index:
                    quote = day.loc[code]
                    if isinstance(quote, pd.DataFrame):
                        quote = quote.iloc[-1]
                if quote is None or code not in marks or not bool(quote.get("buyable", False)):
                    record_trade(date, code, "BUY", "BLOCKED", requested,
                                 "suspended_or_open_at_upper_limit_or_missing_quote")
                else:
                    executable_needs[code] = requested

            total_cash_needed = sum(executable_needs.values()) * (1.0 + costs.buy_rate)
            scale = min(1.0, cash / total_cash_needed) if total_cash_needed > 0 else 0.0
            for code, requested in executable_needs.items():
                buy_notional = requested * scale
                if buy_notional <= 1e-8:
                    continue
                components = _cost_components(buy_notional, costs, "BUY", date, stamp_duty_schedule)
                fee = sum(components[:3])
                if buy_notional + fee > cash + 1e-8:
                    buy_notional = cash / (1.0 + costs.buy_rate)
                    components = _cost_components(buy_notional, costs, "BUY", date, stamp_duty_schedule)
                    fee = sum(components[:3])
                shares[code] += buy_notional / marks[code]
                cash -= buy_notional + fee
                day_cost += fee
                traded_notional += buy_notional
                record_trade(date, code, "BUY", "EXECUTED", buy_notional, "rebalance", components)

        equity = cash + sum(quantity * marks.get(code, 0.0) for code, quantity in shares.items())
        net_return = equity / previous_equity - 1.0 if previous_equity else 0.0
        gross_return = (equity + day_cost) / previous_equity - 1.0 if previous_equity else 0.0
        value_rows.append({
            "date": date, "model": model_name, "top_k": int(top_k), "cost_scenario": scenario_name,
            "portfolio_value": equity, "gross_return": gross_return, "net_return": net_return,
            # Both legs start investing at this opening print.  The benchmark's
            # prior-open-to-first-open move happened before the strategy existed.
            "benchmark_return": 0.0 if date == investment_start else benchmark_return_map.get(date, np.nan),
            "turnover": traded_notional / pretrade_equity if pretrade_equity else 0.0,
            "transaction_cost": day_cost, "cash_weight": cash / equity if equity else 1.0,
            "holdings": len([value for value in shares.values() if value > 0]),
        })
        previous_equity = equity

    return pd.DataFrame(value_rows), pd.DataFrame(trade_rows)
