#!/usr/bin/env python3
"""
审计 fcc-test / fcc-train / fcc-valid 中每条轨迹在 data/traces 其余目录中的来源。

策略：
  1. 先按文件名（basename）在参考库中查找，若存在则比对轨迹数据是否完全一致；
  2. 若无同名文件，用轨迹内容签名（time,bw 序列）在参考库中查找；
  3. 若签名无命中，对参考库逐条做浮点容差比对（较慢，仅对未命中项执行）。

输出 TSV：每条查询轨迹对应 0~N 条参考轨迹（相对 data/traces 的路径），无匹配则标注「未找到」。

用法：
  cd adaptive_bitrate_streaming
  python run/audit_fcc_trace_provenance.py
  python run/audit_fcc_trace_provenance.py --output artifacts/trace_provenance_fcc_audit.tsv
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

_ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ABR_ROOT not in sys.path:
    sys.path.insert(0, _ABR_ROOT)

# 待审计的三个目录（相对 data/traces）
QUERY_REL_DIRS = (
    "test/fcc-test",
    "train/fcc-train",
    "valid/fcc-valid",
)

# 额外排除（非 cooked 轨迹）
EXTRA_EXCLUDE_REL = ("train/eval_trajectories",)

SKIP_SUFFIXES = (".pkl", ".pt", ".json", ".md", ".txt", ".csv")


@dataclass(frozen=True)
class TraceSignature:
    points: tuple[tuple[float, float], ...]

    @property
    def length(self) -> int:
        return len(self.points)

    def content_hash(self) -> str:
        return hashlib.sha256(repr(self.points).encode("utf-8")).hexdigest()


def is_trace_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    base = os.path.basename(path)
    if base.startswith("."):
        return False
    lower = base.lower()
    return not any(lower.endswith(s) for s in SKIP_SUFFIXES)


def read_trace_signature(path: str) -> TraceSignature | None:
    points: list[tuple[float, float]] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    points.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    except OSError:
        return None
    if not points:
        return None
    return TraceSignature(tuple(points))


def signatures_equal(a: TraceSignature, b: TraceSignature, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    if a.length != b.length:
        return False
    for (t1, bw1), (t2, bw2) in zip(a.points, b.points):
        if abs(t1 - t2) > atol and abs(t1 - t2) > rtol * max(abs(t1), abs(t2), 1e-15):
            return False
        if abs(bw1 - bw2) > atol and abs(bw1 - bw2) > rtol * max(abs(bw1), abs(bw2), 1e-15):
            return False
    return True


def normpath_under(root: str, path: str) -> str:
    return os.path.normpath(os.path.join(root, path))


def collect_trace_files(traces_root: str) -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(traces_root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if is_trace_file(fp):
                out.append(fp)
    return sorted(out)


def is_excluded(rel_path: str, exclude_prefixes: tuple[str, ...]) -> bool:
    rel = rel_path.replace("\\", "/")
    for ex in exclude_prefixes:
        ex_slash = ex.replace("\\", "/")
        if rel == ex_slash or rel.startswith(ex_slash + "/"):
            return True
    return False


def dataset_label(rel_path: str) -> str:
    """例如 test/fcc16-test 或 train/Puffer22-train。"""
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else rel_path


def build_reference_index(
    ref_files: list[str],
    traces_root: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[int, list[str]]]:
    """返回 (by_basename, by_content_hash, by_length)。"""
    by_name: dict[str, list[str]] = defaultdict(list)
    by_hash: dict[str, list[str]] = defaultdict(list)
    by_len: dict[int, list[str]] = defaultdict(list)

    n = len(ref_files)
    t0 = time.time()
    for i, fp in enumerate(ref_files):
        rel = os.path.relpath(fp, traces_root).replace("\\", "/")
        by_name[os.path.basename(fp)].append(rel)
        sig = read_trace_signature(fp)
        if sig is None:
            continue
        by_hash[sig.content_hash()].append(rel)
        by_len[sig.length].append(rel)
        if (i + 1) % 500 == 0 or i + 1 == n:
            elapsed = time.time() - t0
            print(f"  索引参考轨迹: {i + 1}/{n} ({elapsed:.1f}s)", flush=True)
    return by_name, by_hash, by_len


def match_query_trace(
    query_fp: str,
    traces_root: str,
    ref_by_name: dict[str, list[str]],
    ref_by_hash: dict[str, list[str]],
    ref_by_len: dict[int, list[str]],
    ref_files_by_rel: dict[str, str],
    *,
    rtol: float,
    atol: float,
    do_bruteforce: bool,
) -> tuple[str, list[str]]:
    """
    返回 (match_method, matched_rel_paths)。
    match_method: filename+content | content_hash | bruteforce | not_found
    """
    basename = os.path.basename(query_fp)
    qsig = read_trace_signature(query_fp)
    if qsig is None:
        return "invalid_query", []

    # 1) 同名文件
    name_hits = ref_by_name.get(basename, [])
    name_matches: list[str] = []
    for rel in name_hits:
        ref_fp = ref_files_by_rel.get(rel)
        if ref_fp is None:
            continue
        rsig = read_trace_signature(ref_fp)
        if rsig is not None and signatures_equal(qsig, rsig, rtol=rtol, atol=atol):
            name_matches.append(rel)
    if name_matches:
        return "filename+content", sorted(set(name_matches))

    # 2) 内容哈希
    h = qsig.content_hash()
    hash_hits = ref_by_hash.get(h, [])
    if hash_hits:
        # 哈希碰撞时再严格比对
        hash_matches: list[str] = []
        for rel in hash_hits:
            ref_fp = ref_files_by_rel.get(rel)
            if ref_fp is None:
                continue
            rsig = read_trace_signature(ref_fp)
            if rsig is not None and signatures_equal(qsig, rsig, rtol=rtol, atol=atol):
                hash_matches.append(rel)
        if hash_matches:
            return "content_hash", sorted(set(hash_matches))

    # 3) 逐条比对（仅同长度候选 + 可选全库）
    if not do_bruteforce:
        return "not_found", []

    candidates: list[str] = list(ref_by_len.get(qsig.length, []))
    if not candidates:
        candidates = list(ref_files_by_rel.keys())

    brute_matches: list[str] = []
    for rel in candidates:
        ref_fp = ref_files_by_rel.get(rel)
        if ref_fp is None:
            continue
        rsig = read_trace_signature(ref_fp)
        if rsig is not None and signatures_equal(qsig, rsig, rtol=rtol, atol=atol):
            brute_matches.append(rel)
    if brute_matches:
        return "bruteforce", sorted(set(brute_matches))
    return "not_found", []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="审计 fcc-test/train/valid 轨迹来源")
    parser.add_argument(
        "--traces-root",
        default=os.path.join(_ABR_ROOT, "data", "traces"),
        help="data/traces 根目录",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(_ABR_ROOT, "artifacts", "trace_provenance_fcc_audit.tsv"),
    )
    parser.add_argument(
        "--no-bruteforce",
        action="store_true",
        help="签名未命中时不做全库逐条比对（更快，可能漏掉浮点格式不同但数值相同的轨迹）",
    )
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args(argv)

    traces_root = os.path.abspath(args.traces_root)
    exclude = tuple(QUERY_REL_DIRS) + tuple(EXTRA_EXCLUDE_REL)

    all_files = collect_trace_files(traces_root)
    query_files: list[str] = []
    ref_files: list[str] = []
    for fp in all_files:
        rel = os.path.relpath(fp, traces_root).replace("\\", "/")
        if is_excluded(rel, exclude):
            query_files.append(fp)
        else:
            ref_files.append(fp)

    ref_files_by_rel = {
        os.path.relpath(fp, traces_root).replace("\\", "/"): fp for fp in ref_files
    }

    print(f"轨迹根目录: {traces_root}")
    print(f"待审计: {len(query_files)} 条（来自 {', '.join(QUERY_REL_DIRS)}）")
    print(f"参考库: {len(ref_files)} 条（其余 data/traces 目录）")
    print("构建参考索引…")
    ref_by_name, ref_by_hash, ref_by_len = build_reference_index(ref_files, traces_root)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    stats = defaultdict(int)

    with open(args.output, "w", encoding="utf-8") as out:
        out.write(
            "query_set\tquery_relpath\tquery_basename\tmatch_method\t"
            "num_matches\tmatched_relpaths\tmatched_datasets\n"
        )

        for qdir in QUERY_REL_DIRS:
            qdir_files = sorted(
                fp for fp in query_files if os.path.relpath(fp, traces_root).replace("\\", "/").startswith(qdir + "/")
            )
            print(f"\n审计 {qdir} ({len(qdir_files)} 条)…")
            for j, qfp in enumerate(qdir_files):
                qrel = os.path.relpath(qfp, traces_root).replace("\\", "/")
                method, matches = match_query_trace(
                    qfp,
                    traces_root,
                    ref_by_name,
                    ref_by_hash,
                    ref_by_len,
                    ref_files_by_rel,
                    rtol=args.rtol,
                    atol=args.atol,
                    do_bruteforce=not args.no_bruteforce,
                )
                if matches:
                    stats["matched"] += 1
                    stats[f"matched_{method}"] += 1
                    matched_paths = ";".join(matches)
                    matched_ds = ";".join(sorted({dataset_label(m) for m in matches}))
                else:
                    stats["not_found"] += 1
                    if method == "invalid_query":
                        stats["invalid_query"] += 1
                    matched_paths = "未找到"
                    matched_ds = "未找到"

                out.write(
                    f"{qdir}\t{qrel}\t{os.path.basename(qfp)}\t{method}\t"
                    f"{len(matches)}\t{matched_paths}\t{matched_ds}\n"
                )
                if (j + 1) % 50 == 0:
                    print(f"  {qdir}: {j + 1}/{len(qdir_files)}", flush=True)

    summary_path = args.output.replace(".tsv", "_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write(f"total_queries\t{len(query_files)}\n")
        sf.write(f"reference_traces\t{len(ref_files)}\n")
        for k in sorted(stats):
            sf.write(f"{k}\t{stats[k]}\n")

    print(f"\n完成。明细: {args.output}")
    print(f"汇总: {summary_path}")
    print(f"  匹配: {stats.get('matched', 0)}，未找到: {stats.get('not_found', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
