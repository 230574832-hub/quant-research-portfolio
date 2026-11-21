# 数据说明

本目录只保存数据契约和构建说明，**不包含原始行情数据、特征文件或预测结果**。

## 需要准备的数据

- A 股日线面板（本地 sqlite）：包含股票代码、交易日、开高低收、成交量/成交额、复权因子、市值、行业、ST 标记等。
- 由 `quant-lab` 构建的上游面板：`data/panel_wide.pkl`、`data/factors.pkl`。
- 本项目需要的点时序数据：`data/features.npz`、`data/dataset_top800.npz`、`data/dataset_meta.pkl`。

## 数据从哪来

`quant-dl-roam` 的 A 股数据来自本地合法渠道的日线行情库，经 `quant-lab/src/build_panel.py` 和 `quant-lab/src/build_factors.py` 构建；由于数据源授权与体积限制，仓库不附带原始数据。

论文复现部分使用作者发布的公开数据集：

- StockMixer 官方仓库：https://github.com/SJTU-DMTai/StockMixer
- NASDAQ / S&P500：官方仓库随附的预处理数据
- NYSE：官方发布的是 0 字节占位文件，本项目使用作者自己的 `load_EOD_data` 从上游 CSV 重建

## 最小字段契约

| 字段 | 含义 |
|---|---|
| `date` | 交易日（必须剔除法定休市日） |
| `stock` | 股票代码 |
| `open/high/low/close` | 复权后价格 |
| `volume/amount` | 成交量/成交额 |
| `market_cap` | 流通市值 |
| `industry` | 行业分类 |
| `is_st` | ST 标记 |

## 注意

- 不要把 TuShare token、券商账号、本地绝对路径写进数据文件。
- 不要把原始行情数据上传到公开仓库；如确需共享，请先确认数据源授权。
- 本项目报告中的数字来自本地研究环境，重新运行时应先做交易日历与复权一致性检查。
