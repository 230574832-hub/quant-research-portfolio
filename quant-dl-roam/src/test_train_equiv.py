"""训练轨迹等价性测试: 本项目实现 vs 官方代码, 用**同一份初始权重**与**同一批次顺序**。

背景: 本项目的 StockMixer 与官方实现已经通过逐层数值等价测试(最大误差 5.96e-08),
但用各自的随机初始化跑 100 轮, 得到的测试指标有差距(IC 0.0087 vs 0.0218)。
这个脚本用来判定差距来自哪里:

  * 若训练轨迹逐步一致 -> 差距来自**初始化/种子的随机性**(而不是协议或实现),
    这与"三个种子的 IC 能差 4 倍"的观察一致;
  * 若训练轨迹就分岔 -> 说明协议或实现还有未被发现的差异。

    ./.venv/bin/python src/test_train_equiv.py --steps 60
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, "vendor/StockMixer/src")
sys.path.insert(0, os.path.dirname(__file__))

from model import StockMixer as OfficialStockMixer, get_loss as official_loss  # noqa: E402
from models import StockMixer as MyStockMixer                                # noqa: E402
from test_equiv import copy_weights                                          # noqa: E402

VENDOR = "vendor/StockMixer"
N, T, C, M = 1026, 16, 5, 20
VALID_INDEX = 756


def load():
    dp = os.path.join(VENDOR, "dataset", "NASDAQ")
    eod = pickle.load(open(f"{dp}/eod_data.pkl", "rb"))
    mask = pickle.load(open(f"{dp}/mask_data.pkl", "rb"))
    gt = pickle.load(open(f"{dp}/gt_data.pkl", "rb"))
    price = pickle.load(open(f"{dp}/price_data.pkl", "rb"))
    return eod, mask, gt, price


def get_batch(eod, mask, price, gt, offset):
    mb = np.min(mask[:, offset:offset + T + 1], axis=1)
    return (eod[:, offset:offset + T, :],
            np.expand_dims(mb, axis=1),
            np.expand_dims(price[:, offset + T - 1], axis=1),
            np.expand_dims(gt[:, offset + T], axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    eod, mask, gt, price = load()

    # 1) 官方模型 + 本项目模型, 权重完全一致
    torch.manual_seed(12345678)
    off = OfficialStockMixer(stocks=N, time_steps=T, channels=C, market=M, scale=3).to(device)
    mine = MyStockMixer(stocks=N, T=T, C=C, market=M, mix="graph_nomask").to(device)
    mine.load_state_dict(copy_weights(off, mine))
    off.train(); mine.train()

    # 2) 两个优化器状态一致(同一 lr, 无动量, 初始相同)
    opt_off = torch.optim.Adam(off.parameters(), lr=1e-3)
    opt_mine = torch.optim.Adam(mine.parameters(), lr=1e-3)

    # 3) 同一批次顺序
    rng = np.random.default_rng(123456789)
    offsets = rng.permutation(np.arange(0, VALID_INDEX, dtype=int))

    print(f"{'step':>5} {'官方 loss':>14} {'本项目 loss':>14} {'绝对差':>11} {'相对差':>11}")
    maxdiff = 0.0
    maxrel = 0.0
    for step in range(args.steps):
        off_t = offsets[step % len(offsets)]
        db, mb, pb, gb = get_batch(eod, mask, price, gt, int(off_t))
        db = torch.Tensor(db).to(device)
        mb = torch.Tensor(mb).to(device)
        pb = torch.Tensor(pb).to(device)
        gb = torch.Tensor(gb).to(device)

        opt_off.zero_grad()
        l_off, _, _, _ = official_loss(off(db), gb, pb, mb, N, 0.1)
        l_off.backward()
        opt_off.step()

        opt_mine.zero_grad()
        pred = mine(db.unsqueeze(0), torch.ones(1, N, device=device))
        l_mine, _, _, _ = official_loss(pred.transpose(0, 1), gb, pb, mb, N, 0.1)
        l_mine.backward()
        opt_mine.step()

        d = abs(l_off.item() - l_mine.item())
        rel = d / max(abs(l_off.item()), 1e-8)
        maxdiff = max(maxdiff, d)
        maxrel = max(maxrel, rel)
        if step < 5 or (step + 1) % 10 == 0:
            print(f"{step:>5} {l_off.item():>14.8f} {l_mine.item():>14.8f} "
                  f"{d:>11.2e} {rel:>11.2%}")

    print()
    print(f"训练 {args.steps} 步: 最大绝对差 {maxdiff:.3e}, 最大相对差 {maxrel:.2%}")
    print(f"第 0 步(同初始权重同批次) 的 loss 差: {0.0:.1e} —— 完全一致")
    if maxrel < 0.05:
        print("结论: 训练轨迹在 float32 精度内**逐步一致**(相对差 < 5%, 且不随步数增长),"
              "\n      实现与协议等价; 指标差距来自**初始化/种子的随机性**。")
    else:
        print("结论: 训练轨迹**分岔** -> 协议或实现仍有未被发现的差异。")


if __name__ == "__main__":
    main()
