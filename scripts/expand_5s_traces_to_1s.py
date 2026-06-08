#!/usr/bin/env python3
"""
将 5s 间隔带宽轨迹按 60 行分块，每块扩成 300 行（1s 间隔）写入 1s/ 目录。

- 分块命名: {源文件名}_{起始行号}，起始行号为 0, 60, 120, ...
- 仅处理完整 60 行块
- 每 5s 原值扩为 5 个 1s 值：±1%~2% 随机扰动，块内均值等于原值
- 输出格式与 fcc-train 一致: time<TAB>bandwidth，从 0.0 起每秒一行
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_trace(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), float(parts[1])))
    return rows


def expand_five_seconds(bw: float, rng: random.Random) -> list[float]:
    """5 个 1s 带宽，均值等于 bw，每个相对 bw 有 1%~2% 扰动（可正可负）。"""
    if bw == 0.0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    multipliers = []
    for _ in range(5):
        sign = rng.choice([-1.0, 1.0])
        pct = rng.uniform(0.01, 0.02)
        multipliers.append(1.0 + sign * pct)

    values = [bw * m for m in multipliers]
    mean_val = sum(values) / 5.0
    scale = bw / mean_val
    return [v * scale for v in values]


def expand_chunk(chunk: list[tuple[float, float]], rng: random.Random) -> list[tuple[float, float]]:
    """每个输出文件时间戳从 0.0 起，步长 1.0，共 300 行（0.0 … 299.0）。"""
    out: list[tuple[float, float]] = []
    for block_idx, (_t5, bw) in enumerate(chunk):
        sub = expand_five_seconds(bw, rng)
        base = block_idx * 5
        for i, v in enumerate(sub):
            out.append((float(base + i), v))
    return out


def write_trace(path: Path, rows: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t, bw in rows:
            f.write(f"{t:.1f}\t{bw}\n")


def process_file(src: Path, out_dir: Path, rng: random.Random) -> int:
    rows = parse_trace(src)
    n_written = 0
    stem = src.name

    for start in range(0, len(rows) - 59, 60):
        chunk = rows[start : start + 60]
        if len(chunk) < 60:
            break
        expanded = expand_chunk(chunk, rng)
        if len(expanded) != 300:
            raise RuntimeError(f"{src}: chunk @{start} expanded to {len(expanded)} lines, expected 300")

        out_name = f"{stem}_{start}"
        write_trace(out_dir / out_name, expanded)
        n_written += 1

    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand 5s bandwidth traces to 1s traces.")
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "5s",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "1s",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_dir = args.src_dir
    out_dir = args.out_dir
    files = sorted(p for p in src_dir.iterdir() if p.is_file() and not p.name.startswith("."))

    total_files = 0
    total_chunks = 0
    for i, src in enumerate(files):
        file_rng = random.Random(args.seed + i)
        n = process_file(src, out_dir, file_rng)
        total_chunks += n
        total_files += 1

    print(f"源文件: {total_files} 个 ({src_dir})")
    print(f"输出块: {total_chunks} 个 ({out_dir})")


if __name__ == "__main__":
    main()
