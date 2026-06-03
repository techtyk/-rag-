"""Reranker batch_size 实验脚本（使用真实法规文档）。

用法：conda run -n retrieval python -m tests.bench_reranker_batch
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import torch
from rerank.reranker import Qwen3Reranker
from config import RERANKER_MODEL_PATH, RERANKER_DEVICE, INDEX_DIR

QUERY = "安全生产责任"


def load_real_documents(n=15):
    """从 metadatas.json 加载真实法规文档。"""
    meta_path = Path(INDEX_DIR) / "metadatas.json"
    metas = json.loads(meta_path.read_text(encoding="utf-8"))
    # 按 content 长度排序，取不同长度的文档，模拟 RRF 召回结果
    metas_sorted = sorted(metas, key=lambda x: len(x["content"]))
    # 取短、中、长各一些
    step = max(1, len(metas_sorted) // n)
    selected = [metas_sorted[i] for i in range(0, len(metas_sorted), step)][:n]
    docs = [m["content"] for m in selected]
    lengths = [len(d) for d in docs]
    print(f"文档长度范围: {min(lengths)}-{max(lengths)} 字符, 平均 {sum(lengths)//len(lengths)} 字符")
    return docs


def get_vram():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        return allocated, reserved, total
    return 0, 0, 0


def bench_batch_size(reranker, query, documents, batch_size, n_runs=3):
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reranker.rerank(query, documents, batch_size=batch_size)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
    avg = sum(times) / len(times)
    mn = min(times)
    return avg, mn


def main():
    print("=" * 70)
    print("Reranker batch_size 实验（真实法规文档）")
    print("=" * 70)

    documents = load_real_documents(15)

    print(f"\n加载 Reranker 模型...")
    reranker = Qwen3Reranker(RERANKER_MODEL_PATH, device=RERANKER_DEVICE)
    alloc1, reserved1, total = get_vram()
    print(f"模型显存: allocated={alloc1:.0f}MB, reserved={reserved1:.0f}MB / total={total:.0f}MB")

    # Warmup
    print("\nWarmup...")
    reranker.rerank(QUERY, documents[:3], batch_size=3)
    reranker.rerank(QUERY, documents[:5], batch_size=5)

    # 测试
    batch_sizes = [1, 2, 4, 8, 12, 15]
    print(f"\n{'batch_size':>10} | {'avg (ms)':>10} | {'min (ms)':>10} | {'peak_alloc MB':>14} | {'peak_reserved MB':>16} | {'OOM?':>5}")
    print("-" * 85)

    results = []
    for bs in batch_sizes:
        try:
            torch.cuda.reset_peak_memory_stats()
            avg, mn = bench_batch_size(reranker, QUERY, documents, bs, n_runs=3)
            peak_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024
            peak_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
            print(f"{bs:>10} | {avg:>10.1f} | {mn:>10.1f} | {peak_alloc:>14.0f} | {peak_reserved:>16.0f} | {'No':>5}")
            results.append((bs, avg, mn, peak_alloc, peak_reserved))
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{bs:>10} | {'---':>10} | {'---':>10} | {'---':>14} | {'---':>16} | {'OOM':>5}")
                torch.cuda.empty_cache()
                results.append((bs, None, None, None, None))
            else:
                raise

    # 分析
    print(f"\n总显存: {total:.0f}MB")
    valid = [(bs, avg, mn, pa, pr) for bs, avg, mn, pa, pr in results if avg is not None]
    if valid:
        for bs, avg, mn, pa, pr in valid:
            pct = pa / total * 100
            print(f"  batch_size={bs:>2}: peak={pa:.0f}MB ({pct:.1f}%), avg={avg:.1f}ms")
        # 找最大安全 batch
        max_safe = max(valid, key=lambda x: x[0])
        print(f"\n  最大安全 batch_size: {max_safe[0]} (peak={max_safe[3]:.0f}MB)")


if __name__ == "__main__":
    main()
