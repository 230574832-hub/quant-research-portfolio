# quant-dl-roam

**深度学习横截面选股的复现与失效诊断**  
复现 AAAI-24 的 StockMixer 与 MASTER，并把它们迁移到 A 股做 walk-forward 样本外评测；重点不是“再跑一个模型”，而是定位模型为什么失效。

[主报告](docs/main_report.pdf) · [StockMixer 论文复现报告](docs/stockmixer_reproduction.pdf) · [文献规格](docs/lit_spec.md) · [数据缺陷通告](docs/data_defect_calendar.md)

---

## 研究问题

1. AAAI-24 的 StockMixer / MASTER，在作者代码与作者数据上能否复现？
2. 把同一套结构迁移到 A 股、使用 8 个量价特征和每折 Top-500 股票池时，论文里的“结构增量”是否还成立？
3. 如果不成立，失效点是数据、训练协议，还是模型的核心模块？

## 数据与评测协议

- **A 股**：每折固定 Top-500（按测试段之前 500 个交易日的平均对数市值选），5 折 walk-forward；训练段逐折增长，标签为次日收盘—收盘收益，5 日 purge/embargo。
- **模型**：Ridge、MLP、GRU、MarketGRU（特征门控/隐藏态门控）、StockMixer（完整版/去混合块/置换不变池化）。
- **种子**：3 个；早停只看训练史最后 15% 的验证 RankIC。
- **成本**：单边 15bp；可成交性剔除收盘涨跌停、停牌等。
- **论文复现**：作者发布的代码 + 作者数据集，NASDAQ / NYSE / S&P500 三个数据集；NYSE 的官方 pkl 为 0 字节占位，使用作者自己的 `load_EOD_data` 从上游 CSV 重建。

## 主要结论

1. **StockMixer 在 A 股上不成立，失效点可定位到横截面“股票混合”模块。**  
   完整模型测试 RankIC **0.0025**（t=0.94，等同随机）；移除混合块后恢复至 **0.0248**；替换为置换不变的市场均值池化同样为 0.0248。训练损失明显低于其他模型、验证 RankIC 首轮后转负，指向过拟合。

2. **MASTER 的特征门控在 A 股上没有增量，隐藏态门控才有效。**  
   纯 GRU RankIC 0.0502；论文原式特征门控 0.0494（与纯 GRU 基本无差异）；将市场状态接入隐藏态后为 **0.0584**。

3. **信号存在，但只有多空组合收得下。**  
   收盘→收盘、扣 15bp 单边成本后，`mgru` 多空净 Sharpe 2.05；同一股票池随机打分的零假设为 −9.38。多头腿净 Sharpe 仍为负，降到 5 日调仓后才转正。

4. **论文复现：架构与协议跑通，数值层面部分复现。**  
   prec@10 在三个数据集上复现到 **96%–97%**；IC 类指标只复现到 **25%–56%**，且作者代码上不同种子的 IC 可相差约 4 倍。另查出论文 Table 1 的交易日数与发布代码不一致，以及官方 `RIC` 实际计算的是 **ICIR** 而非 Spearman 秩相关。

5. **上游数据缺陷：休市日被当成交易日。**  
   源面板混入 271 个法定休市日（价格结转、成交量为 0），导致 43% 的日期窗口失效；修复后测试期有效天数由 298/521 恢复到 **488/488**。

## 仓库结构

```text
quant-dl-roam/
├── README.md
├── requirements.txt
├── src/
│   ├── build_dataset.py          # 构建点时序数据集、PIT 股票池、交易日轴
│   ├── models.py                 # Ridge / MLP / GRU / MarketGRU / StockMixer
│   ├── train_wf.py               # 5 折 walk-forward 训练
│   ├── evaluate_wf.py            # RankIC / ICIR / 组合 / 随机零假设
│   ├── repro_paper.py            # 论文日志解析与 Table 2 对拍
│   └── ...
├── figs/                         # 报告图表
├── docs/                         # 报告、文献规格、数据缺陷通告
└── data/                         # 数据说明；不包含原始数据
```

建议先看：`src/models.py` → `src/train_wf.py` → `src/evaluate_wf.py` → `src/repro_paper.py`。

## 快速开始

```bash
pip install -r requirements.txt

# 1) 构建 A 股数据集（需要上游 quant-lab 的面板）
python src/build_dataset.py

# 2) walk-forward 训练与评测
python src/train_wf.py --seed 0 --models ridge,mlp,gru,mgru,mgru_feat,sm,sm_nomix,sm_pool
python src/evaluate_wf.py
```

论文复现部分需要额外准备官方 StockMixer 仓库与作者发布的数据集，放到 `vendor/StockMixer/` 后运行 `src/repro_paper.py`。本仓库不包含该上游仓库与数据集。

## 数据说明

原始行情数据、特征面板、模型预测 parquet、日志与虚拟环境均不随仓库分发。数据字段、构建顺序和可公开的论文数据集说明见 [`data/README.md`](data/README.md)。

## 限制

- A 股结果基于 8 个量价特征和 Top-500 股票池，不能外推到全市场或基本面模型。
- 未建模冲击成本、融券约束与真实涨跌停排队；15bp 是简化成本假设。
- 论文复现的 IC 缺口可能来自协议、数据版本和随机性，报告只陈述可复核的差异，不把缺口归因于单一原因。

## 参考工作

- Jinyong Fan, Yanyan Shen. **StockMixer: A Simple Yet Strong MLP-Based Architecture for Stock Price Forecasting.** AAAI 2024.
- Tong Li, et al. **MASTER: Market-Guided Stock Transformer for Stock Price Forecasting.** AAAI 2024.
