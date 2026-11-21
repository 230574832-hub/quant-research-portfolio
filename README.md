# 量化研究项目作品集

> 聚焦**可复现性、样本外验证与数据审计**的金融机器学习与量化实证研究集合。  
> 仓库仅保留核心源码、实验报告与结果图表；原始行情数据、本地缓存、虚拟环境及运行日志不随仓库分发。

**作者**：吴宇彤 · 上海交通大学（金融学—计算机科学与技术双学士学位）  
**联系**：zhzxwuyutong@sjtu.edu.cn

---

## 项目导航

| 项目 | 核心研究问题 | 代表性发现 | 入口 |
|---|---|---|---|
| **quant-dl-roam** | AAAI-24 的 *StockMixer* 与 *MASTER* 在 A 股是否有效？失效点在哪里？ | 完整 *StockMixer* 测试 RankIC 仅 **0.0025**，移除横截面“股票混合”模块后恢复至 **0.0248**；*MASTER* 将市场状态接入隐藏层优于特征门控（**0.0584 vs 0.0494**）。 | [项目说明](quant-dl-roam/README.md) · [主报告](quant-dl-roam/docs/main_report.pdf) |
| **industry-gnn** | 行业关系图能否带来超越匹配随机图的稳定经济增益？ | 2026 年预注册 Holdout（41 周）中，跨行业图相对匹配随机图的周均超额收益为 **+0.53%**（Holm *p* = 0.80），控制风格与风险暴露后缩至 **+0.12%**（不显著）。 | [项目说明](industry-gnn/README.md) · [结论](industry-gnn/docs/final_conclusions.md) |
| **quant-lab** | A 股量价因子的预测力来源及 T+1、容量约束下的可执行性。 | 24 个量价因子中，低特质波动因子的隔夜 RankIC 达 **11.96%**，日内仅 **1.76%**；完成 **22,319 笔**模拟订单的跨实现零差异对账。 | [项目说明](quant-lab/README.md) · [主报告](quant-lab/docs/factor_health_report.pdf) |

---

## 实验与工程规范

三个项目均遵循同一套实证与审计流程：

1. **先锁定协议，再执行评测**：统一采用 Walk-Forward 或预注册 Holdout 划分，显式约定股票池、交易成本、Purge/Embargo 与早停规则，避免事后调参。
2. **前置数据审计**：在建模前优先排查并修复交易日历混入休市日、Point-in-Time（PIT）字段滞后、停牌/涨跌停不可成交及复权市值对齐等问题。
3. **以消融定位失效机制**：不仅汇报汇总指标，而是通过模块消融（如移除 *StockMixer* 横截面混合块）与门控对比，明确模型增益或过拟合的具体来源。
4. **如实保留负结果与适用边界**：`industry-gnn` 在多重检验校正与风险匹配后未识别出稳健的图特异性经济增益，该结果予以完整保留，不做选择性筛选。
5. **交易账本逐笔可核对**：`quant-lab` 完成 22,319 笔模拟订单的跨实现零差异对账，确保回测收益与底层执行逻辑严格一致。

## 建议阅读顺序

1. 各项目根目录的 `README.md`：了解研究问题、数据协议、核心结论与代码结构。
2. 各项目的 `docs/` 目录：查阅完整实验报告或关键审计备忘录。
3. 各项目的 `src/` 目录：查看核心实现（具体入口文件已在对应项目 README 中标注）。

## 复现与合规说明

受限于数据体量、第三方数据源授权及隐私合规要求，本仓库不包含任何原始行情数据、TuShare Token、API Key 或本地路径配置：

- `quant-dl-roam`：A 股实验依赖上游 `quant-lab` 生成的特征面板；论文复现部分需自行获取官方 *StockMixer* 仓库及作者公开数据集。
- `industry-gnn`：运行数据构建脚本需具备合法 TuShare 权限，Token 仅通过环境变量或本地隔离配置读取。
- `quant-lab`：依赖本地日线行情数据库，仓库仅提供 Schema 定义与特征/回测构建脚本。

详细复现步骤与目录说明见各项目 `README.md` 及 `data/README.md`。第三方论文、数据集与上游开源代码的版权归原作者所有，引用本仓库思路或代码时请同时引用对应原始文献。

## 参考工作

- Jinyong Fan, Yanyan Shen. **StockMixer: A Simple Yet Strong MLP-Based Architecture for Stock Price Forecasting.** *AAAI*, 2024.
- Tong Li, et al. **MASTER: Market-Guided Stock Transformer for Stock Price Forecasting.** *AAAI*, 2024.
- Lifan Zhao, Shuming Kong, Yanyan Shen. **DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting.** *KDD*, 2023.
