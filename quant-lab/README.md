# quant-lab

**A 股量价因子体检与可交易性审计（2011–2026）**  
从本地行情库出发，构造 24 个纯量价因子，统一评测“次日全收益 / 隔夜 / 日内”三种标签，并回答三个问题：因子在预测什么、能不能交易、有没有衰减。

[主报告](docs/factor_health_report.pdf) · [跨实现一致性审计](docs/e1_consistency_audit.pdf) · [逐笔对账与集中度审计](docs/e1bc_order_reconciliation.pdf)

---

## 研究问题

1. 量价因子的预测力主要落在隔夜还是日内？
2. 在 A 股 T+1、涨跌停、停牌和交易成本约束下，这些预测力能不能兑现？
3. 因子的衰减是普遍现象，还是只在特定股票池/特定因子上出现？

## 数据与协议

- **样本**：2011-03-28 至 2026-03-25；剔除北交所/B 股/ST/上市不足 120 日/价格低于 2 元/20 日均额低于 2,000 万元的股票。
- **因子**：24 个纯量价因子，涵盖动量、反转、波动率、换手率、流动性、特质波动等。
- **标签**：次日全收益、隔夜收益、日内收益。
- **评测**：RankIC（raw / 中性化 / 去个股均值）、十分位组合、期限结构；walk-forward Ridge / 梯度提升树。
- **可交易性**：T+1、周频 Top-50、单边 15bp 成本、涨跌停与停牌近似处理；千万级资金容量网格。
- **对账**：策略引擎与独立实现逐笔订单对账。

## 主要结论

1. **因子主要在预测隔夜。**  
   低特质波动因子的隔夜 RankIC 为 **11.96%**，日内仅 **1.76%**；低换手因子为 10.16% vs 0.35%。A 股隔夜均值约 −7.0bp、日内约 +13.0bp。

2. **T+1 让日内腿不可交易。**  
   日内腿（开盘买、收盘卖）多头收益普遍为正且很大，但 T+1 禁止；可执行的隔夜腿多头普遍为负，alpha 主要集中在空头腿。日频调仓下最好的因子净 Sharpe 仅约 0.14。

3. **可执行的周频 Top-50 组合里，只有小市值与低价赢过基准。**  
   全样本小市值超额约 **+8.4%/年**（IR 0.48），低价约 **+6.3%/年**（IR 0.39）；IC 最高的低特质波动因子反而跑输基准。

4. **因子没有普遍衰减，而是分化。**  
   5 日反转在大市值池明显衰减；波动率/换手类因子反而增强。梯度提升树相对最优单因子只提升约 11% 的 OOS IC，且逐年不稳定。

5. **订单级可复核。**  
   22,319/22,319 笔模拟订单与独立实现一致；成交额最大差 0.000000，权益最大相对差 5.4e-15。容量（千万元级）而非信号本身，是策略落地的主要约束。

## 仓库结构

```text
quant-lab/
├── README.md
├── requirements.txt
├── src/
│   ├── build_panel.py            # 日线 sqlite + ST/行业/股本 → 宽表面板
│   ├── build_factors.py          # 24 个因子 + 三种标签 + 股票池掩码
│   ├── evaluate2.py              # RankIC / 十分位 / 期限结构
│   ├── realistic.py              # T+1 可执行周频组合与固定池衰减
│   ├── ml_baseline.py            # walk-forward Ridge / 梯度提升树
│   ├── portfolio_ml.py           # ML 与单因子同窗口组合对比
│   ├── p1_engine.py              # 组合执行引擎
│   ├── e1_engine_exact.py        # 精确执行引擎（逐笔对账）
│   └── ...
├── figs/                         # 报告图表
├── docs/                         # 主报告与关键审计备忘录
└── data/                         # 数据说明；不包含原始数据
```

建议先看：`src/build_factors.py` → `src/evaluate2.py` → `src/realistic.py` → `src/e1_engine_exact.py`。

## 快速开始

```bash
pip install -r requirements.txt

# 需要先准备本地日线面板；字段与构建顺序见 data/README.md
python src/build_panel.py
python src/build_factors.py
python src/evaluate2.py
python src/realistic.py
python src/ml_baseline.py
python src/portfolio_ml.py
```

原始行情数据库（约 6.9GB）不随仓库分发；仓库只保留构建脚本、因子定义和报告结果。

## 数据说明

仓库不包含日线 sqlite、面板 pickle、因子 pickle、订单明细和逐笔成交文件。字段契约、样本区间与清洗规则见 [`data/README.md`](data/README.md)。

## 限制

- 未做行业中性化，因子可能含行业/风格暴露；没有财务、分析师、文本因子。
- 涨跌停判定使用价格阈值近似；本地 `stk_limit` 只覆盖 2026 年。
- 未建模冲击成本、真实排队与融券约束；15bp 是简化成本假设。
- 容量结论基于千万级资金网格，不能直接外推到更大资金。

## 参考工作

- 项目主报告：`docs/factor_health_report.pdf`
- 执行引擎与逐笔对账：`docs/e1_consistency_audit.pdf`、`docs/e1bc_order_reconciliation.pdf`
