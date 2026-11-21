# Backtest audit

## Execution sequence

Stocks are ranked on the signal date using all decision-eligible names. At the next open the simulator attempts sales first, then buys the fixed Top-K list. A blocked Top-K purchase remains blocked and is not replaced by a lower-ranked stock. A blocked sale is carried and marked using the latest available adjusted open. Fractional shares intentionally remove board-lot effects so the experiment isolates signal quality.

## Tradeability

Missing/non-positive quotes and suspension records block both sides. A one-price session at the upper limit blocks buying; a one-price session at the lower limit blocks selling. Synthetic tests cover suspension, upper-limit buying, lower-limit selling, and future missing exit quotes. `tables/execution_constraint_audit.csv` reports whether Full Run actually exercised each branch.

## Costs

Commission and slippage apply to buys and sells. Stamp duty applies to sells only and is selected by execution date: 10 bps through 2023-08-27, 5 bps from 2023-08-28. A fixed 5 bps scenario is explicitly counterfactual and is not the historical Base result.

## Metric definitions

- Net daily return: change in marked portfolio equity after executed costs.
- Annualized return: portfolio CAGR using `number of trading observations / 252` years.
- Benchmark annualized return: benchmark open-to-open CAGR on the same calendar.
- Annualized excess return: strategy CAGR minus benchmark CAGR.
- Volatility: daily net-return standard deviation times \(\sqrt{252}\).
- Sharpe: daily mean divided by daily standard deviation times \(\sqrt{252}\), zero risk-free rate.
- Maximum drawdown: minimum cumulative net wealth divided by its running maximum minus one.
- Turnover: buy plus sell notional divided by pre-trade equity; it is double-sided.
- Transaction cost: realized commission + stamp duty + modeled slippage in currency units.

## Known simplifications

The simulator does not model round lots, order-book depth, partial fills, price impact beyond slippage, dividends as cash, shorting, or intraday signal availability. Adjusted fractional shares are a research abstraction. Those limitations must accompany any economic claim.
