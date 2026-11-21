# 数据说明

本目录只保存数据契约和运行说明，**不包含原始行情数据、缓存、训练产物或 TuShare token**。

## 数据来源

- 日线、复权、市值、行业、指数成分等数据通过合法的 TuShare 权限获取。
- 具体字段、缓存键、PIT 证据和下载流程见 `docs/setup.md`、`docs/data_leakage_audit.md` 与 `docs/experiment_protocol.md`。

## 仓库中保留的审计材料

`artifacts/` 保存的是**数据指纹和审计表**，不是原始数据：

- `data_file_fingerprints.json`
- `download_data_quality.json`
- `industry_taxonomy_audit.json`
- `reference_data_audit.json`
- `universe_timing_audit.json`

这些文件用于说明数据版本、字段缺失和 PIT 时间戳的证据等级。

## 最小字段契约

| 字段 | 含义 |
|---|---|
| `trade_date` | 交易日 |
| `ts_code` | 股票代码 |
| `open/high/low/close` | 复权后价格 |
| `vol/amount` | 成交量/成交额 |
| `total_mv/circ_mv` | 总市值/流通市值 |
| `industry` | 行业分类 |
| `index_weight` | 沪深 300 动态成分权重 |

## 注意

- 不要把 TuShare token、缓存、`.env` 或本地绝对路径提交到仓库。
- 不要把原始行情数据上传到公开仓库；如确需共享，请先确认数据源授权。
- 重新运行前先跑 `SMOKE_TEST` 和 PIT 审计，不要直接跳到 `FULL_RUN`。
