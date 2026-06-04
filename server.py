"""FastAPI 服务入口：将 RAG Pipeline 暴露为 HTTP API。"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import (INDEX_DIR, BM25_K1, BM25_B, BM25_BACKEND, BM25_RECALL_K,
                    RERANK_TOP_K, RERANKER_MODEL, RERANKER_MODEL_PATH, RERANKER_DEVICE,
                    RERANKER_BATCH_SIZE,
                    DENSE_MODEL_PATH, DENSE_DEVICE, DENSE_BATCH_SIZE,
                    DENSE_RECALL_K, RRF_METHOD, RRF_K, RRF_TOP_K, RRF_2WAY_AXIS,
                    SERVER_HOST, SERVER_PORT,
                    QA_INDEX_DIR, QA_BM25_K1, QA_BM25_B, QA_BM25_BACKEND, QA_BM25_RECALL_K,
                    QA_INDEX_DEVICE, QA_INDEX_BATCH_SIZE, QA_DENSE_RECALL_K,
                    QA_RRF_K, QA_RRF_TOP_K, QA_RERANK_TOP_K, QA_SIMILARITY_THRESHOLD,
                    QA_RERANKER_BATCH_SIZE)
from retrieval.retrieve import Retriever, _check_index_complete
from retrieval.qa_retrieve import QARetriever, _check_qa_index_complete
from rerank.reranker import RERANKER_REGISTRY


# ─── 全局对象（在 lifespan 中初始化） ───
retriever: Retriever | None = None
qa_retriever: QARetriever | None = None
reranker = None


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
        "rrf_top_k": RRF_TOP_K,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时加载模型，关闭时清理。"""
    global retriever, qa_retriever, reranker

    config = _build_retriever_config()
    qa_config = _build_qa_retriever_config()
    t0 = time.time()

    if config.get("index_dir") and _check_index_complete(config["index_dir"]):
        retriever = Retriever.load(config["index_dir"], config)
    else:
        print("错误：法规索引未找到。请先运行 python index.py 构建索引。")
        retriever = None

    # 加载 QA 索引，共享法规 Dense 模型
    shared_model = None
    if retriever is not None and retriever.dense is not None:
        retriever.dense._ensure_model()
        shared_model = retriever.dense.model
    if qa_config.get("index_dir") and _check_qa_index_complete(qa_config["index_dir"]):
        qa_retriever = QARetriever.load(qa_config["index_dir"], qa_config,
                                         shared_model=shared_model)
    else:
        print("QA 索引未找到，QA RAG 不可用。请运行 python index.py --qa 构建索引。")
        qa_retriever = None

    reranker_cls = RERANKER_REGISTRY[RERANKER_MODEL]
    reranker = reranker_cls(RERANKER_MODEL_PATH, device=RERANKER_DEVICE)

    print(f"模型加载完成，耗时 {time.time()-t0:.1f}s")
    yield


app = FastAPI(title="法规 RAG Pipeline", lifespan=lifespan)


# ─── 请求 / 响应模型 ───

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="查询文本")
    top_k: int | None = Field(None, ge=1, le=50, description="精排返回条数，默认用配置值")
    debug: bool = Field(False, description="True 则返回中间召回/融合信息")


class ResultItem(BaseModel):
    rerank_score: float
    rrf_score: float | None = None
    source: str | None = None
    matched_tokens: dict | None = None
    file_name: str | None = None
    chapter: str | None = None
    article_no: str | None = None
    content: str | None = None


class QAResultItem(BaseModel):
    qa_id: int
    matched_question: str
    answer: str
    rerank_score: float


class QueryResponse(BaseModel):
    query: str
    count: int
    results: list[ResultItem]
    qa_results: list[QAResultItem] | None = None
    elapsed_ms: float
    debug: dict | None = None


class HealthResponse(BaseModel):
    status: str
    reranker_model: str


# ─── 端点 ───

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ready" if retriever and reranker else "loading",
        reranker_model=RERANKER_MODEL,
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if retriever is None:
        return JSONResponse(
            status_code=503,
            content={"error": "索引未就绪，请先运行 python index.py 构建索引"},
        )

    t0 = time.time()
    top_k = req.top_k or RERANK_TOP_K

    # 阶段一：法规多路召回 + RRF 融合
    reg_candidates = retriever.retrieve(req.query)
    reg_count = len(reg_candidates)

    # 阶段一（QA）：QA 多路召回 + 2 路 RRF 融合
    qa_candidates = []
    if qa_retriever is not None:
        # 复用法规 Dense 的 query embedding
        query_emb = None
        if retriever.dense is not None and retriever.dense.model is not None:
            import numpy as np
            query_emb = np.array(
                retriever.dense.model.encode([req.query], normalize_embeddings=True,
                                             device=DENSE_DEVICE),
                dtype="float32")
        qa_candidates = qa_retriever.retrieve(req.query, query_emb=query_emb)
    qa_count = len(qa_candidates)

    # 阶段二：Rerank（法规和 QA 分别独立精排）
    # QA 问题文本极短（~20-40字符），与法规内容（~数百-数千字符）长度差异过大，
    # 合并 batch 会导致 QA 被无效 pad 到法规长度，故分开精排
    qa_results_out = []

    # 法规 Rerank
    reg_docs = [r["content"] for r in reg_candidates]
    if reg_docs:
        reg_scores = reranker.rerank(req.query, reg_docs, batch_size=RERANKER_BATCH_SIZE)
        for r, s in zip(reg_candidates, reg_scores):
            r["rerank_score"] = s
    reg_candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    display = reg_candidates[:top_k]

    # QA Rerank
    qa_docs = [q["question"] for q in qa_candidates]
    if qa_docs:
        qa_scores = reranker.rerank(req.query, qa_docs, batch_size=QA_RERANKER_BATCH_SIZE)
        for c, s in zip(qa_candidates, qa_scores):
            c["rerank_score"] = s
        qa_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        qa_top = qa_candidates[:QA_RERANK_TOP_K]
        qa_results_out = [
            QAResultItem(
                qa_id=c["qa_id"],
                matched_question=c["question"],
                answer=c["answer"],
                rerank_score=c["rerank_score"],
            )
            for c in qa_top
            if c["rerank_score"] >= QA_SIMILARITY_THRESHOLD
        ]

    elapsed_ms = (time.time() - t0) * 1000

    results = [
        ResultItem(
            rerank_score=r["rerank_score"],
            rrf_score=r.get("score"),
            source=r.get("source"),
            matched_tokens=r.get("matched_tokens"),
            file_name=r.get("file_name"),
            chapter=r.get("chapter"),
            article_no=r.get("article_no"),
            content=r.get("content"),
        )
        for r in display
    ]

    debug_info = None
    if req.debug:
        debug_info = {
            "recall_count": reg_count,
            "qa_recall_count": qa_count,
            "rrf_top_k": RRF_TOP_K,
            "qa_rrf_top_k": QA_RRF_TOP_K,
            "rerank_model": RERANKER_MODEL,
            "rrf_method": RRF_METHOD,
            "qa_similarity_threshold": QA_SIMILARITY_THRESHOLD,
        }

    return QueryResponse(
        query=req.query,
        count=len(results),
        results=results,
        qa_results=qa_results_out if qa_results_out else None,
        elapsed_ms=round(elapsed_ms, 1),
        debug=debug_info,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=SERVER_HOST, port=SERVER_PORT)
