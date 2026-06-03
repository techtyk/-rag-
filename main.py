from config import (INDEX_DIR, BM25_K1, BM25_B, BM25_BACKEND, BM25_RECALL_K,
                    RERANK_TOP_K, RERANKER_MODEL, RERANKER_MODEL_PATH, RERANKER_DEVICE,
                    RERANKER_BATCH_SIZE,
                    DENSE_MODEL_PATH, DENSE_DEVICE, DENSE_BATCH_SIZE,
                    DENSE_RECALL_K, RRF_METHOD, RRF_K, RRF_2WAY_AXIS)
from retrieval.retrieve import Retriever, _check_index_complete
from rerank.reranker import RERANKER_REGISTRY


def _build_retriever_config():
    return {
        "index_dir": str(INDEX_DIR),
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "bm25_backend": BM25_BACKEND,
        "bm25_recall_k": BM25_RECALL_K,
        "dense_model_path": DENSE_MODEL_PATH,
        "dense_device": DENSE_DEVICE,
        "dense_batch_size": DENSE_BATCH_SIZE,
        "dense_recall_k": DENSE_RECALL_K,
        "rrf_method": RRF_METHOD,
        "rrf_k": RRF_K,
        "rrf_2way_axis": RRF_2WAY_AXIS,
    }


def rag_pipeline():
    config = _build_retriever_config()

    if config.get("index_dir") and _check_index_complete(config["index_dir"]):
        retriever = Retriever.load(config["index_dir"], config)
    else:
        print("错误：索引未找到。请先运行以下命令构建索引：")
        print("  cd /home/moga/project/dense_training/app && python index.py")
        return

    # 构建重排器（精排阶段）
    reranker_cls = RERANKER_REGISTRY[RERANKER_MODEL]
    reranker = reranker_cls(RERANKER_MODEL_PATH, device=RERANKER_DEVICE)

    # 交互式查询循环
    print("\n" + "=" * 80)
    print("RAG Pipeline 已启动（输入 q 退出）")
    print("=" * 80)
    while True:
        query = input("\n查询：").strip()
        if query.lower() in ("q", "quit", "exit"):
            print("退出。")
            break
        if not query:
            continue

        # 阶段一：多路召回 + RRF 融合
        candidates = retriever.retrieve(query)
        if not candidates:
            print("未召回任何结果。")
            continue

        # 阶段二：Reranker 精排
        documents = [r["content"] for r in candidates]
        rerank_scores = reranker.rerank(query, documents, batch_size=RERANKER_BATCH_SIZE)
        for r, s in zip(candidates, rerank_scores):
            r["rerank_score"] = s
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        display_results = candidates[:RERANK_TOP_K]

        print(f"\n查询：「{query}」召回 {len(candidates)} 条，精排展示前 {len(display_results)} 条")
        print("-" * 80)
        for i, r in enumerate(display_results, 1):
            matched_str = " | ".join(f"{k}={v}" for k, v in r['matched_tokens'].items()) if r['matched_tokens'] else "无"
            print(f"【第 {i} 名】Rerank={r['rerank_score']:.4f} | RRF={r['score']:.6f} | "
                  f"来源：{r['source']} | 匹配词元：{matched_str}")
            print(f"  出处：{r['file_name']} · {r['chapter']} · {r['article_no']}")
            content = r['content']
            print(f"  内容：{content[:200]}{'...' if len(content) > 200 else ''}")
            print("-" * 80)


if __name__ == "__main__":
    rag_pipeline()
