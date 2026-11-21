"""逐层数值等价测试:本项目的 StockMixer vs 官方 model.py 的 StockMixer。

把官方模型的权重逐个映射进本项目的实现(本项目省略了 NoGraphMixer 里
nn.LayerNorm 的可学习仿射参数, 官方有 weight/bias 两组), 然后在同一批输入上
比较两者的输出。

若数值一致, 则本项目关于 StockMixer 的 A 股结论建立在**同一个机制**上,
只是换了市场与特征; 若不一致, 差异点会被逐层打印出来。

    ./.venv/bin/python src/test_equiv.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "vendor/StockMixer/src")
sys.path.insert(0, os.path.dirname(__file__))

from model import StockMixer as OfficialStockMixer   # noqa: E402
from models import StockMixer as MyStockMixer        # noqa: E402


def copy_weights(off, mine):
    """官方 state_dict -> 本项目 state_dict"""
    sd = off.state_dict()
    new = {}

    def g(k):
        return sd[k]

    # 时间混合: 官方 mixer.mix_layer.{LN_1,LN_2,timeMixer,channelMixer}
    #           本项目 mixer.main.{ln1,ln2,time,chan}
    for a, b in [("mixer.mix_layer.LN_1", "mixer.main.ln1"),
                 ("mixer.mix_layer.LN_2", "mixer.main.ln2"),
                 ("mixer.scale_mix_layer.LN_1", "mixer.scale.ln1"),
                 ("mixer.scale_mix_layer.LN_2", "mixer.scale.ln2")]:
        new[b + ".weight"] = g(a + ".weight")
        new[b + ".bias"] = g(a + ".bias")
    # TriU: 官方 mixer.mix_layer.timeMixer.triU.{i}.{weight,bias}
    for i in range(16):            # 主分支 T=16
        new[f"mixer.main.time.triU.{i}.weight"] = g(f"mixer.mix_layer.timeMixer.triU.{i}.weight")
        new[f"mixer.main.time.triU.{i}.bias"] = g(f"mixer.mix_layer.timeMixer.triU.{i}.bias")
    for i in range(8):             # 尺度分支 T/2=8
        new[f"mixer.scale.time.triU.{i}.weight"] = g(f"mixer.scale_mix_layer.timeMixer.triU.{i}.weight")
        new[f"mixer.scale.time.triU.{i}.bias"] = g(f"mixer.scale_mix_layer.timeMixer.triU.{i}.bias")
    # MixerBlock(通道混合): 官方 channelMixer.dense_1/2 -> 本项目 chan.d1/d2
    for src, dst in [("mixer.mix_layer.channelMixer", "mixer.main.chan"),
                     ("mixer.scale_mix_layer.channelMixer", "mixer.scale.chan")]:
        for a, b in [("dense_1", "d1"), ("dense_2", "d2")]:
            new[f"{dst}.{b}.weight"] = g(f"{src}.{a}.weight")
            new[f"{dst}.{b}.bias"] = g(f"{src}.{a}.bias")
    # 头部与卷积
    new["channel_fc.weight"] = g("channel_fc.weight")
    new["channel_fc.bias"] = g("channel_fc.bias")
    new["time_fc.weight"] = g("time_fc.weight")
    new["time_fc.bias"] = g("time_fc.bias")
    new["time_fc_.weight"] = g("time_fc_.weight")
    new["time_fc_.bias"] = g("time_fc_.bias")
    new["conv.weight"] = g("conv.weight")
    new["conv.bias"] = g("conv.bias")
    # 股票混合: 官方 stock_mixer.dense1/2 (+ LayerNorm 仿射, 本项目无)
    new["stock_mixer.d1.weight"] = g("stock_mixer.dense1.weight")
    new["stock_mixer.d1.bias"] = g("stock_mixer.dense1.bias")
    new["stock_mixer.d2.weight"] = g("stock_mixer.dense2.weight")
    new["stock_mixer.d2.bias"] = g("stock_mixer.dense2.bias")
    return new


def main():
    N, T, C, m = 64, 16, 5, 20
    torch.manual_seed(0)
    off = OfficialStockMixer(stocks=N, time_steps=T, channels=C, market=m, scale=3)
    mine = MyStockMixer(stocks=N, T=T, C=C, market=m, mix="graph_nomask")
    missing = mine.load_state_dict(copy_weights(off, mine), strict=True)
    off.eval()
    mine.eval()

    x = torch.randn(N, T, C) * 0.3 + 0.5
    with torch.no_grad():
        y_off = off(x)
        # 本项目 forward 期望 (B,N,T,C) 与 (B,N) 掩码; 用 graph_nomask 时
        # 股票混合层的统计量用全部 N 只(与官方一致)
        y_mine = mine(x.unsqueeze(0), torch.ones(1, N))[0]
    y_off = y_off.reshape(-1)
    y_mine = y_mine.reshape(-1)
    d = (y_off - y_mine).abs()
    print(f"输入 {tuple(x.shape)}  输出 {tuple(y_off.shape)}")
    print(f"官方输出   前 5 个: {y_off[:5].numpy().round(6)}")
    print(f"本项目输出 前 5 个: {y_mine[:5].numpy().round(6)}")
    print(f"最大绝对误差 {d.max().item():.3e}   均方误差 {(d**2).mean().item():.3e}")
    print(f"官方输出量级 std {y_off.std().item():.4f}")
    ok = d.max().item() < 1e-4
    print()
    print("结论:", "两者**数值等价**(差异仅来自未拷贝的 LayerNorm 仿射参数, 其初值为 1/0)" if ok
          else "存在实质差异 —— 本项目实现与官方不等价, A 股结论需重新审视")
    if not ok:
        raise SystemExit(1)

    # 再验: 掩码版本在"全部有效"时与官方一致(masked 分支的统计量也用全部 N)
    mine_masked = MyStockMixer(stocks=N, T=T, C=C, market=m, mix="graph")
    mine_masked.load_state_dict(copy_weights(off, mine_masked), strict=True)
    mine_masked.eval()
    with torch.no_grad():
        y_mm = mine_masked(x.unsqueeze(0), torch.ones(1, N))[0].reshape(-1)
    print(f"带掩码版本(掩码全 1)最大绝对误差 {(y_off-y_mm).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
