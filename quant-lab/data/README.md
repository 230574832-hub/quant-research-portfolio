# 数据说明

本目录只保存数据契约和构建说明，**不包含日线 sqlite、面板 pickle、因子文件、订单明细或逐笔成交文件**。

## 需要准备的数据

- 本地日线行情 sqlite（约 6.9GB）：包含股票代码、交易日、开高低收、成交量/成交额、复权因子、市值、行业、ST 标记等。
- 由 `src/build_panel.py` 生成的 `data/panel_wide.pkl`。
- 由 `src/build_factors.py` 生成的 `data/factors.pkl`。

## 构建顺序

```bash
python src/build_panel.py       # sqlite + ST/行业/股本 → 宽表面板
python src/build_factors.py     # 24 个因子 + 三种标签 + 股票池掩码
python src/evaluate2.py         # RankIC / 十分位 / 期限结构
python src/realistic.py         # T+1 可执行周频组合
```

## 最小字段契约

| 字段 | 含义 |
|---|---|
| `date` | 交易日 |
| `stock` | 股票代码 |
| `open/high/low/close` | 复权后价格 |
| `volume/amount` | 成交量/成交额 |
| `market_cap` | 流通市值 |
| `industry` | 行业分类 |
| `is_st` | ST 标记 |
| `is_suspended` | 停牌标记 |

## 注意

- 原始行情数据库受数据源授权约束，不随仓库分发。
- 报告中的 22,319 笔逐笔对账来自本地模拟订单与独立实现，不包含真实账户或券商数据。
- 不要把 TuShare token、账户信息或本地绝对路径写进数据文件。
