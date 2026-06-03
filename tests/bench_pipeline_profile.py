"""Pipeline 完整性能 Profile 脚本。

测试各阶段耗时，使用最优配置（Dense GPU 编码 + Reranker batch_size=16）。

用法：conda run -n retrieval python -m tests.bench_pipeline_profile
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import torch
from config import (INDEX_DIR, BM25_K1, BM25_B, BM25_BACKEND, BM25_RECALL_K,
                    RERANK_TOP_K, RERANKER_MODEL, RERANKER_MODEL_PATH, RERANKER_DEVICE,
                    RERANKER_BATCH_SIZE,
                    DENSE_MODEL_PATH, DENSE_DEVICE, DENSE_BATCH_SIZE,
                    DENSE_RECALL_K, RRF_METHOD, RRF_K, RRF_TOP_K, RRF_2WAY_AXIS)
from retrieval.retrieve import Retriever, _check_index_complete
from rerank.reranker import RERANKER_REGISTRY

QUERIES = [
    "安全生产责任",
    "机动车驾驶证申领",
    "道路交通安全违法行为处罚",
    "建设工程安全生产管理",
    "环境保护责任追究",
]


def profile_pipeline():
    print("=" * 70)
    print("Pipeline 完整性能 Profile")
    print(f"设备: Dense={DENSE_DEVICE}, Reranker={RERANKER_DEVICE}")
    print(f"配置: RERANKER_BATCH_SIZE={RERANKER_BATCH_SIZE}, RRF_TOP_K={RRF_TOP_K}")
    print("=" * 70)

    # 加载索引
    config = {
        "index_dir": str(INDEX_DIR),
        "bm25_k1": BM25_K1, "bm25_b": BM25_B, "bm25_backend": BM25_BACKEND,
        "bm25_recall_k": BM25_RECALL_K,
        "dense_model_path": DENSE_MODEL_PATH, "dense_device": DENSE_DEVICE,
        "dense_batch_size": DENSE_BATCH_SIZE, "dense_recall_k": DENSE_RECALL_K,
        "rrf_method": RRF_METHOD, "rrf_k": RRF_K,
        "rrf_top_k": RRF_TOP_K, "rrf_2way_axis": RRF_2WAY_AXIS,
    }
    retriever = Retriever.load(config["index_dir"], config)

    reranker_cls = RERANKER_REGISTRY[RERANKER_MODEL]
    reranker = reranker_cls(RERANKER_MODEL_PATH, device=RERANKER_DEVICE)

    # 显示显存
    alloc = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    print(f"\n模型加载后显存: allocated={alloc:.0f}MB, reserved={reserved:.0f}MB / total={total:.0f}MB")

    # Warmup
    print("\nWarmup...")
    for q in QUERIES[:2]:
        candidates = retriever.retrieve(q)
        if candidates:
            docs = [r["content"] for r in candidates]
            reranker.rerank(q, docs, batch_size=RERANKER_BATCH_SIZE)

    # 正式测试
    print(f"\n{'Query':>30} | {'BM25(ms)':>8} | {'Dense(ms)':>9} | {'RRF(ms)':>7} | {'Rerank(ms)':>10} | {'Total(ms)':>9} | {'候选数':>6} | {'输出数':>6}")
    print("-" * 110)

    all_bm25, all_dense, all_rrf, all_rerank, all_total = [], [], [], [], []

    for query in QUERIES:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # BM25
        t1 = time.perf_counter()
        bm25_results = retriever.bm25.search(query, top_k=DENSE_RECALL_K)
        t_bm25 = (time.perf_counter() - t1) * 1000

        # Dense
        t2 = time.perf_counter()
        dense_results = retriever.dense.search(query, top_k=DENSE_RECALL_K)
        t_dense = (time.perf_counter() - t2) * 1000

        # RRF
        t3 = time.perf_counter()
        if RRF_METHOD == "2way":
            fused = retriever._fuse_2way(bm25_results, dense_results, RRF_2WAY_AXIS)
        else:
            fused = retriever._fuse_4way(bm25_results, dense_results)
        t_rrf = (time.perf_counter() - t3) * 1000

        recall_count = len(fused)

        # Rerank
        t4 = time.perf_counter()
        documents = [r["content"] for r in fused]
        rerank_scores = reranker.rerank(query, documents, batch_size=RERANKER_BATCH_SIZE)
        for r, s in zip(fused, rerank_scores):
            r["rerank_score"] = s
        fused.sort(key=lambda x: x["rerank_score"], reverse=True)
        display = fused[:RERANK_TOP_K]
        t_rerank = (time.perf_counter() - t4) * 1000

        torch.cuda.synchronize()
        t_total = (time.perf_counter() - t0) * 1000

        all_bm25.append(t_bm25)
        all_dense.append(t_dense)
        all_rrf.append(t_rrf)
        all_rerank.append(t_rerank)
        all_total.append(t_total)

        print(f"{query:>30} | {t_bm25:>8.1f} | {t_dense:>9.1f} | {t_rrf:>7.1f} | {t_rerank:>10.1f} | {t_total:>9.1f} | {recall_count:>6} | {len(display):>6}")

    n = len(QUERIES)
    print("-" * 110)
    print(f"{'平均':>30} | {sum(all_bm25)/n:>8.1f} | {sum(all_dense)/n:>9.1f} | {sum(all_rrf)/n:>7.1f} | {sum(all_rerank)/n:>10.1f} | {sum(all_total)/n:>9.1f}")

    # 显存报告
    peak_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024
    peak_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
    print(f"\n峰值显存: allocated={peak_alloc:.0f}MB, reserved={peak_reserved:.0f}MB / total={total:.0f}MB")
    print(f"显存利用率: {peak_alloc/total*100:.1f}%")

    # 百分比分析
    avg_total = sum(all_total) / n
    avg_bm25 = sum(all_bm25) / n
    avg_dense = sum(all_dense) / n
    avg_rrf = sum(all_rrf) / n
    avg_rerank = sum(all_rerank) / n

    print(f"\n耗时占比:")
    print(f"  BM25 检索:    {avg_bm25:>6.1f}ms  ({avg_bm25/avg_total*100:>5.1f}%)")
    print(f"  Dense 检索:   {avg_dense:>6.1f}ms  ({avg_dense/avg_total*100:>5.1f}%)")
    print(f"  RRF 融合:     {avg_rrf:>6.1f}ms  ({avg_rrf/avg_total*100:>5.1f}%)")
    print(f"  Rerank 精排:  {avg_rerank:>6.1f}ms  ({avg_rerank/avg_total*100:>5.1f}%)")


if __name__ == "__main__":
    profile_pipeline()
