"""用作者自己的 load_EOD_data 生成 NYSE 版预处理数据。

官方仓库的 dataset/NYSE/*.pkl 是 0 字节占位文件(作者没发布 NYSE 数据)。
但上游 RSR 仓库公开了全部原始 CSV, 且作者自己的 load_EOD_data.py 就在仓库里,
所以 NYSE 是可以按与 NASDAQ 完全相同的口径复现出来的。

本脚本:
  1) 读 RSR 的合格股票名单(1737 只, 与论文 Table 1 一致)
  2) 直接 import 官方的 load_EOD_data (不重写、不改语义)
  3) 落盘成官方 train.py 期望的 4 个 pkl

    ./.venv/bin/python src/build_nyse_dataset.py
"""
import os
import pickle
import sys
import time

import numpy as np

RSR = "../RSR"
VENDOR = "vendor/StockMixer"
sys.path.insert(0, os.path.join(VENDOR, "src"))
from load_data import load_EOD_data          # 作者的原函数  # noqa: E402


def main():
    tick_file = f"{RSR}/data/NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv"
    tickers = [l.strip() for l in open(tick_file) if l.strip()]
    print(f"RSR 合格股票名单: {len(tickers)} 只 (论文 Table 1: NYSE 1737 只)")

    data_path = f"{RSR}/data/2013-01-01"
    missing = [t for t in tickers if not os.path.exists(f"{data_path}/NYSE_{t}_1.csv")]
    if missing:
        print(f"  [warn] {len(missing)} 只在 {data_path} 里没有对应 CSV，跳过")
        tickers = [t for t in tickers if t not in missing]

    print("调用官方 load_EOD_data (逐文件 np.genfromtxt, 较慢) ...")
    t0 = time.time()
    eod, masks, gt, base_price = load_EOD_data(data_path, "NYSE", tickers)
    print(f"  用时 {time.time()-t0:.0f}s")
    print(f"  eod {eod.shape}  masks {masks.shape}  gt {gt.shape}  price {base_price.shape}")
    print(f"  有效掩码比例 {np.mean(masks>0.5):.4f}  gt 有限比例 {np.isfinite(gt).mean():.4f}")

    out = f"{VENDOR}/dataset/NYSE"
    os.makedirs(out, exist_ok=True)
    for name, arr in [("eod_data", eod), ("mask_data", masks),
                      ("gt_data", gt), ("price_data", base_price)]:
        with open(f"{out}/{name}.pkl", "wb") as f:
            pickle.dump(arr, f)
        print(f"  写入 {out}/{name}.pkl  {np.asarray(arr).shape} "
              f"{os.path.getsize(f'{out}/{name}.pkl')/1e6:.1f} MB")


if __name__ == "__main__":
    main()
