"""把 out/*.csv 生成的表格刷新进主报告。

报告里的表格全部来自 src/report_tables.py。本脚本把生成块拆成
「标题 + 表」的序列, 再在报告里按顺序定位并逐块替换, 因此:
  * 重复运行是幂等的;
  * 标题文字也跟着产物走, 不会脱节;
  * 支持一个生成键对应多张表(表 5a / 表 5b)。

    ./.venv/bin/python src/refresh_report_tables.py
"""
import subprocess

REPORT = "reports/深度学习横截面模型评测报告.md"

# 报告里的表标题前缀 -> report_tables.py 的生成键
PLAN = [
    ("### 表 1", "summary"),
    ("### 表 2", "folds"),
    ("### 表 3", "year"),
    ("### 表 4", "decomp"),
    ("### 表 5a", "port"),
    ("### 表 6", "lr"),
]


def gen(key):
    out = subprocess.run(["./.venv/bin/python", "src/report_tables.py", key],
                         capture_output=True, text=True).stdout.strip()
    if not out:
        raise SystemExit(f"表格生成失败: {key}")
    return out.split("\n")


def split_blocks(lines):
    """把生成块拆成 [(标题行或 None, [表行...]), ...]"""
    blocks, i = [], 0
    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue
        if lines[i].startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            blocks.append((None, lines[i:j]))
            i = j
            continue
        head = lines[i]
        i += 1
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        tbl = []
        while i < len(lines) and lines[i].startswith("|"):
            tbl.append(lines[i])
            i += 1
        blocks.append((head, tbl))
    return blocks


def main():
    lines = open(REPORT, encoding="utf-8").read().split("\n")

    for marker, key in PLAN:
        idx = [i for i, l in enumerate(lines) if l.startswith(marker)]
        if not idx:
            print(f"  [skip] 报告里没有 {marker}")
            continue
        blocks = split_blocks(gen(key))
        i = idx[0]
        # 第一个块替换报告里的标题(连同其后的表)
        head, tbl = blocks[0]
        t = i + 1
        while t < len(lines) and not lines[t].startswith("|"):
            t += 1
        end = t
        while end < len(lines) and lines[end].startswith("|"):
            end += 1
        lines[i:end] = ([head] if head else []) + tbl
        # 后续块(表 5b 之类): 依次在后面找下一张表替换
        cursor = i + (1 if head else 0) + len(tbl)
        for head2, tbl2 in blocks[1:]:
            t2 = cursor
            while t2 < len(lines) and not lines[t2].startswith("|"):
                t2 += 1
            if t2 >= len(lines):
                break
            e2 = t2
            while e2 < len(lines) and lines[e2].startswith("|"):
                e2 += 1
            # 把表前(可能堆叠的)旧标题与空行一并吃掉, 避免留下孤儿标题
            s2 = t2
            while s2 > 0:
                if lines[s2 - 1].strip() == "":
                    s2 -= 1
                elif lines[s2 - 1].startswith("### 表"):
                    s2 -= 1
                else:
                    break
            lines[s2:e2] = ([head2] if head2 else []) + tbl2
            cursor = s2 + (1 if head2 else 0) + len(tbl2)
        print(f"  刷新 {marker}  ({key}, {len(blocks)} 个块)")

    open(REPORT, "w", encoding="utf-8").write("\n".join(lines))
    print("done")


if __name__ == "__main__":
    main()
