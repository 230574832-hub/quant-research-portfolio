# 量化研究项目作品集

> 三个以**可复现性、样本外验证、数据质量**为核心的本地研究项目。  
> 仓库只保留代码、报告与图表；原始行情数据、缓存、虚拟环境和运行日志不随仓库分发。

作者：吴宇彤 · 上海交通大学（金融学—计算机科学与技术双学士）  
联系方式：zhzxwuyutong@sjtu.edu.cn

---

## 项目导航

| 项目 | 研究问题 | 代表性结果 | 入口 |
|---|---|---|---|
| **quant-dl-roam** | AAAI-24 的 StockMixer / MASTER 在 A 股上是否成立？失效点在哪里？ | 完整 StockMixer 的测试 RankIC 仅 **0.0025**；移除横截面“股票混合”模块后恢复到 **0.0248**。MASTER 将市场状态接入隐藏层优于特征门控（**0.0584 vs 0.0494**）。 | [项目说明](quant-dl-roam/README.md) · [主报告](quant-dl-roam/docs/main_report.pdf) |
| **industry-gnn** | 行业关系图能否带来稳定的、超越匹配随机图的经济增益？ | 2026 预注册 Holdout（41 期）中，跨行业图相对匹配随机图的周均超额 **+0.53%**（Holm **p=0.80**）；风险暴露匹配后缩至 **+0.12%**（不显著）。 | [项目说明](industry-gnn/README.md) · [结论](industry-gnn/docs/final_conclusions.md) |
| **quant-lab** | A 股量价因子的预测力来自哪里？在 T+1 与容量约束下能否兑现？ | 24 个量价因子中，低特质波动因子的隔夜 RankIC 为 **11.96%**，日内仅 **1.76%**；完成 **22,319/22,319** 笔订单逐笔对账。 | [项目说明](quant-lab/README.md) · [主报告](quant-lab/docs/factor_health_report.pdf) |

---

## 研究重点

三个项目共享同一套研究习惯：

1. **先固定评价协议，再看结果。** 时间序列采用 walk-forward 或预注册 holdout；不使用未来信息；对股票池、成本、purge/embargo 和早停规则做显式约定。
2. **把数据问题当作一等公民。** 交易日历混入休市日、PIT 字段缺失、停牌与涨跌停、复权与市值对齐，都会先审计再建模。
3. **用消融定位失效点，而不是只报总分。** 例如把 StockMixer 的横截面混合模块单独移除，观察 RankIC 从 0.0025 回到 0.0248。
4. **报告负结果和限制。** `industry-gnn` 没有识别出稳定的图特异性经济增益；这被作为结论保留，而不是被调参掩盖。
5. **尽量做到订单级可复核。** `quant-lab` 的模拟订单与独立实现逐笔对账，22,319 笔订单零差异。

## 建议阅读顺序

1. 每个项目的 `README.md`：研究问题、数据协议、主要结论、代码结构。
2. 每个项目的 `docs/`：完整报告或关键审计备忘录。
3. `src/`：核心实现。每个项目 README 都标出了建议先看的文件。

## 复现说明

本仓库**不包含原始行情数据**，原因包括数据体量、数据源授权和隐私合规。各项目的数据依赖不同：

- `quant-dl-roam`：需要上游 `quant-lab` 的 A 股面板；论文复现部分需要自行下载官方 StockMixer 仓库与作者发布的数据集。
- `industry-gnn`：需要合法的 TuShare 数据权限；仓库不保存 token，运行时通过环境变量或隐藏输入读取。
- `quant-lab`：需要本地日线行情数据库；仓库只保留数据 schema 与构建脚本，不分发原始数据。

具体步骤见各项目 `README.md` 和 `data/README.md`。

## 数据、代码与引用

- 仓库不包含任何 TuShare token、API key、账户信息或本地绝对路径。
- 第三方论文、数据与上游代码版权归原作者；本仓库只保存个人实现、实验报告和图表。
- 如果使用本仓库中的分析思路或代码，请同时引用对应论文（见各项目 README 的“参考工作”）。

## 参考工作

- Jinyong Fan, Yanyan Shen. **StockMixer: A Simple Yet Strong MLP-Based Architecture for Stock Price Forecasting.** AAAI 2024.
- Tong Li, et al. **MASTER: Market-Guided Stock Transformer for Stock Price Forecasting.** AAAI 2024.
- Lifan Zhao, Shuming Kong, Yanyan Shen. **DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting.** KDD 2023.
