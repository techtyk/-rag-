"""FastAPI 服务入口：将 RAG Pipeline 暴露为 HTTP API。"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from config import (INDEX_DIR, KB_PATH, BM25_K1, BM25_B, BM25_BACKEND, BM25_RECALL_K,
                    RERANK_TOP_K, RERANKER_MODEL, RERANKER_MODEL_PATH, RERANKER_DEVICE,
                    DENSE_MODEL_PATH, DENSE_DEVICE, DENSE_BATCH_SIZE,
                    DENSE_RECALL_K, RRF_METHOD, RRF_K, RRF_TOP_K, RRF_2WAY_AXIS,
                    SERVER_HOST, SERVER_PORT)
from utils.doc_parser import parse_regulation
from retrieval.retrieve import Retriever, _check_index_complete
from rerank.reranker import RERANKER_REGISTRY


# ─── 全局对象（在 lifespan 中初始化） ───
retriever: Retriever | None = None
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时加载模型，关闭时清理。"""
    global retriever, reranker

    config = _build_retriever_config()
    t0 = time.time()

    if config.get("index_dir") and _check_index_complete(config["index_dir"]):
        retriever = Retriever.load(config["index_dir"], config)
    else:
        print("索引未找到，正在解析文档并构建索引...")
        docs_loc, docs_content, metadatas = parse_regulation(str(KB_PATH))
        print(f"共解析出 {len(metadatas)} 个条款")
        retriever = Retriever(docs_loc, docs_content, metadatas, config=config)

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


class QueryResponse(BaseModel):
    query: str
    count: int
    results: list[ResultItem]
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
    t0 = time.time()

    top_k = req.top_k or RERANK_TOP_K

    # 阶段一：多路召回 + RRF 融合
    candidates = retriever.retrieve(req.query)
    recall_count = len(candidates)

    # 阶段二：Reranker 精排
    documents = [r["content"] for r in candidates]
    rerank_scores = reranker.rerank(req.query, documents)
    for r, s in zip(candidates, rerank_scores):
        r["rerank_score"] = s
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    display = candidates[:top_k]
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
            "recall_count": recall_count,
            "rrf_top_k": RRF_TOP_K,
            "rerank_model": RERANKER_MODEL,
            "rrf_method": RRF_METHOD,
        }

    return QueryResponse(
        query=req.query,
        count=len(results),
        results=results,
        elapsed_ms=round(elapsed_ms, 1),
        debug=debug_info,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=SERVER_HOST, port=SERVER_PORT)
