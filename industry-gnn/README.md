# industry-gnn

**行业图神经网络的预注册样本外检验**  
在沪深 300 周频 Top-10 选股框架下，比较行业关系图、跨行业图与度数匹配的随机图，检验“行业关系”是否带来稳定的经济增益。

[最终结论](docs/final_conclusions.md) · [研究设计](docs/research_design.md) · [实验协议](docs/experiment_protocol.md) · [预注册文件](docs/preregistration_2026_holdout.md) · [环境与运行](docs/setup.md)

---

## 研究问题

行业关系图是股票预测中常见的 inductive bias，但“图结构有效”需要同时回答三个问题：

1. 跨行业图相对**同行业图**是否有增量？
2. 跨行业图相对**度数匹配的随机跨行业图**是否有增量？
3. 这种增量在**未见过的 2026 年样本外窗口**中是否仍然存在？

项目把 2026 年作为预注册 holdout，先冻结协议，再打开测试窗口；2024–2025 的结果只作为探索性证据。

## 数据与协议

- **股票池**：沪深 300 历史动态成分股，周频调仓，Top-10 等权。
- **模型**：Ridge、Shared MLP、Industry MLP、Residual-GCN、Residual-GAT（Complete / Sparse）、HIST-inspired 因子模型。
- **图结构对照**：同行业图、跨行业图、度数匹配的随机跨行业图；参数量严格对齐（1,637 参数）。
- **执行**：保守开盘执行、显式涨停/停牌不可成交测试、历史印花税、与组合持有区间对齐的沪深 300 基准。
- **预注册**：2026 holdout 共 41 期；协议、偏离和冻结配置见 `docs/`。
- **稳健性**：3 个种子 × 多次重训；风险暴露匹配（行业、市值、Beta、波动率）。

## 主要结论

1. **没有识别出稳定的图特异性经济增益。**  
   2026 预注册 holdout（41 期）中，跨行业图相对匹配随机图的周均超额为 **+0.53%/期**（95% CI [−0.62, +1.87]），Holm 校正 **p=0.80**。

2. **风险暴露解释了一部分差异。**  
   匹配行业、市值、Beta 与波动率后，2026 年 C−E 缩至 **+0.12%/期**，不显著；2024–2025 的方向也不稳定。

3. **头部持仓对训练初始化和集成方式高度敏感。**  
   同一数据、代码、种子重跑时预测 Spearman 0.99988、Top-10 重合 98.99%；换种子后 Top-10 逐日重合仅 0.04–0.21。单次训练的 Top-10 回测排名，对模型经济优劣的代表性有限。

4. **这不是“图没有价值”的证明。**  
   更准确的表述是：在当前数据、模型、投资规则与检验功效下，尚未获得真实跨行业图稳定优于匹配随机图的统计证据；真实增益存在但样本不足以识别，仍是可能解释。

## 仓库结构

```text
industry-gnn/
├── README.md
├── requirements.txt
├── environment.yml
├── run_local.py
├── src/industry_gnn/          # 数据、特征、图、模型、训练、回测、PIT 审计
├── configs/                   # 冻结配置与实验配置
├── notebooks/                 # 从零运行的主 Notebook
├── scripts/                   # setup / preflight / full run
├── tests/                     # 完整性、合成数据、评测契约测试
├── artifacts/                 # 数据指纹与审计表
├── docs/                      # 结论、设计、协议、预注册、偏离记录
└── data/                      # 数据说明；不包含原始数据
```

建议先看：`docs/final_conclusions.md` → `docs/research_design.md` → `src/industry_gnn/models.py` → `src/industry_gnn/graphs.py` → `src/industry_gnn/training.py`。

## 快速开始

```bash
pip install -r requirements.txt
# 或：conda env create -f environment.yml

# 先跑 GPU-gated 的 SMOKE_TEST preflight，确认 PIT 表和成交测试通过
python run_local.py preflight

# 正式全量运行需要显式确认，并准备 TuShare 权限与 GPU
python run_local.py full --confirm-full-run
```

更完整的 Windows / Linux / VS Code 流程见 [`docs/setup.md`](docs/setup.md)。主实验 Notebook 为 `notebooks/Industry_GNN_Research_PIT_HIST_V4_From_Zero.ipynb`。

## 数据与密钥

- 原始行情、缓存、训练产物、虚拟环境不随仓库分发。
- 不保存 TuShare token；运行时通过环境变量或隐藏输入读取，代码中有脱敏检查。
- 数据字段与缓存契约见 [`data/README.md`](data/README.md)。

## 限制

- 2024–2025 是探索性证据，2026 才是预注册 confirmatory holdout；两者不能混在一起讲。
- 风险匹配和稳定性分析是 post-hoc descriptive，不构成新的因果识别。
- 3 个种子只能说明初始化敏感性，不能精确估计种子分布。
- 结果依赖沪深 300 动态成分股、周频 Top-10 和当前成本假设，不能直接外推到其他股票池。

## 参考工作

- Yanyan Shen, et al. **Entity-Centric Data Management for the Ubiquitous Computing Era.** SCIS 2026.
- Xin Zhang, Yanyan Shen, et al. **Efficient GNN Training on Giant Graphs with Collective Batching and Scheduling.** PVLDB 2026.
- Jiale Deng, Yanyan Shen, et al. **DeMix: Debugging Training Data with Mixed Data Error Types by Investigating Influence Vectors.** KDD 2026.
- Lifan Zhao, Yanyan Shen. **Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting.** KDD 2025.
