import numpy as np

from config import (INDEX_DIR, BM25_K1, BM25_B, BM25_BACKEND, BM25_RECALL_K,
                    RERANK_TOP_K, RERANKER_MODEL, RERANKER_MODEL_PATH, RERANKER_DEVICE,
                    RERANKER_BATCH_SIZE,
                    DENSE_MODEL_PATH, DENSE_DEVICE, DENSE_BATCH_SIZE,
                    DENSE_RECALL_K, RRF_METHOD, RRF_K, RRF_2WAY_AXIS,
                    QA_INDEX_DIR, QA_BM25_K1, QA_BM25_B, QA_BM25_BACKEND, QA_BM25_RECALL_K,
                    QA_INDEX_DEVICE, QA_INDEX_BATCH_SIZE, QA_DENSE_RECALL_K,
                    QA_RRF_K, QA_RRF_TOP_K, QA_RERANK_TOP_K, QA_SIMILARITY_THRESHOLD)
from retrieval.retrieve import Retriever, _check_index_complete
from retrieval.qa_retrieve import QARetriever, _check_qa_index_complete
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


def _build_qa_retriever_config():
    return {
        "index_dir": str(QA_INDEX_DIR),
        "bm25_k1": QA_BM25_K1,
        "bm25_b": QA_BM25_B,
        "bm25_backend": QA_BM25_BACKEND,
        "bm25_recall_k": QA_BM25_RECALL_K,
        "dense_model_path": DENSE_MODEL_PATH,
        "dense_device": QA_INDEX_DEVICE,
        "dense_batch_size": QA_INDEX_BATCH_SIZE,
        "dense_recall_k": QA_DENSE_RECALL_K,
        "rrf_k": QA_RRF_K,
        "rrf_top_k": QA_RRF_TOP_K,
    }


def rag_pipeline():
    config = _build_retriever_config()
    qa_config = _build_qa_retriever_config()

    if config.get("index_dir") and _check_index_complete(config["index_dir"]):
        retriever = Retriever.load(config["index_dir"], config)
    else:
        print("错误：索引未找到。请先运行以下命令构建索引：")
        print("  cd /home/moga/project/dense_training/app && python index.py")
        return

    # 加载 QA 索引，共享法规 Dense 模型
    qa_retriever = None
    shared_model = retriever.dense.model if retriever.dense else None
    if qa_config.get("index_dir") and _check_qa_index_complete(qa_config["index_dir"]):
        qa_retriever = QARetriever.load(qa_config["index_dir"], qa_config,
                                         shared_model=shared_model)
    else:
        print("QA 索引未找到，QA RAG 不可用。请运行 python index.py --qa 构建索引。")

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

        # 阶段一：法规多路召回 + RRF 融合
        reg_candidates = retriever.retrieve(query)
        if not reg_candidates and qa_retriever is None:
            print("未召回任何结果。")
            continue

        # 阶段一（QA）：QA 多路召回 + 2 路 RRF
        qa_candidates = []
        if qa_retriever is not None:
            query_emb = np.array(
                retriever.dense.model.encode([query], normalize_embeddings=True,
                                             device=DENSE_DEVICE),
                dtype="float32")
            qa_candidates = qa_retriever.retrieve(query, query_emb=query_emb)

        # 阶段二：合并 Rerank
        all_docs = [r["content"] for r in reg_candidates] + [q["question"] for q in qa_candidates]

        if not all_docs:
            print("未召回任何结果。")
            continue

        all_scores = reranker.rerank(query, all_docs, batch_size=RERANKER_BATCH_SIZE)

        # 拆分分数回法规组和 QA 组
        reg_scores = all_scores[:len(reg_candidates)]
        qa_scores = all_scores[len(reg_candidates):]

        # 法规组
        for r, s in zip(reg_candidates, reg_scores):
            r["rerank_score"] = s
        reg_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        display_results = reg_candidates[:RERANK_TOP_K]

        # QA 组
        for c, s in zip(qa_candidates, qa_scores):
            c["rerank_score"] = s
        qa_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 展示法规结果
        print(f"\n查询：「{query}」法规召回 {len(reg_candidates)} 条，精排展示前 {len(display_results)} 条")
        print("-" * 80)
        for i, r in enumerate(display_results, 1):
            matched_str = " | ".join(f"{k}={v}" for k, v in r['matched_tokens'].items()) if r['matched_tokens'] else "无"
            print(f"【第 {i} 名】Rerank={r['rerank_score']:.4f} | RRF={r['score']:.6f} | "
                  f"来源：{r['source']} | 匹配词元：{matched_str}")
            print(f"  出处：{r['file_name']} · {r['chapter']} · {r['article_no']}")
            content = r['content']
            print(f"  内容：{content[:200]}{'...' if len(content) > 200 else ''}")
            print("-" * 80)

        # 展示 QA 结果
        if qa_candidates:
            qa_top = qa_candidates[:QA_RERANK_TOP_K]
            filtered = [c for c in qa_top if c["rerank_score"] >= QA_SIMILARITY_THRESHOLD]
            if filtered:
                print(f"\n【QA 匹配】{len(filtered)} 条")
                for i, c in enumerate(filtered, 1):
                    print(f"  [{i}] 置信度={c['rerank_score']:.4f} | Q：{c['question']}")
                    answer = c['answer']
                    print(f"      A：{answer[:200]}{'...' if len(answer) > 200 else ''}")
                print("-" * 80)
            else:
                best = qa_candidates[0]
                print(f"\n【QA 匹配】最佳候选置信度 {best['rerank_score']:.4f} 低于阈值 {QA_SIMILARITY_THRESHOLD}，未输出")
                print("-" * 80)


if __name__ == "__main__":
    rag_pipeline()
