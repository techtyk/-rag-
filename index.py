"""索引构建入口：解析知识库文档并构建 BM25 + Dense 索引。

用法:
    cd /home/moga/project/dense_training/app
    conda activate retrieval
    python index.py            # 构建所有索引
    python index.py --qa       # 仅构建 QA 索引
    python index.py --reg      # 仅构建法规索引
"""
import json
import time
from pathlib import Path

from config import (KB_PATH, INDEX_DIR,
                    BM25_K1, BM25_B, BM25_BACKEND, BM25_RECALL_K,
                    DENSE_MODEL_PATH, DENSE_RECALL_K,
                    INDEX_DEVICE, INDEX_BATCH_SIZE,
                    RRF_METHOD, RRF_K, RRF_TOP_K, RRF_2WAY_AXIS,
                    CHUNK_SIZE, CHUNK_OVERLAP,
                    QA_DATA_PATH, QA_INDEX_DIR,
                    QA_BM25_K1, QA_BM25_B, QA_BM25_BACKEND, QA_BM25_RECALL_K,
                    QA_INDEX_DEVICE, QA_INDEX_BATCH_SIZE, QA_DENSE_RECALL_K,
                    QA_RRF_K, QA_RRF_TOP_K)
from utils.doc_parser import parse_regulation
from utils.chunker import chunk_documents
from retrieval.retrieve import Retriever, _check_index_complete
from retrieval.qa_retrieve import QARetriever, _check_qa_index_complete
from utils.qa_data_loader import load_qa_pairs


def build_index():
    if _check_index_complete(str(INDEX_DIR)):
        print(f"索引已存在于 {INDEX_DIR}，如需重建请先删除该目录。")
        return

    t0 = time.time()
    print("正在解析知识库文档...")
    docs_loc, docs_content, metadatas = parse_regulation(str(KB_PATH))
    print(f"共解析出 {len(metadatas)} 个条款")

    if len(metadatas) == 0:
        print("错误：未解析到任何条款，请检查知识库 JSON 结构。")
        return

    # 智能切分：对超过 CHUNK_SIZE 的文档进行分块
    docs_loc, docs_content, metadatas = chunk_documents(
        docs_loc, docs_content, metadatas,
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )

    # 切分结果持久化（不修改原始 metadatas.json）
    temp_dir = Path(__file__).parent / "artifacts" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "split_metadatas.json").write_text(
        json.dumps(metadatas, ensure_ascii=False), encoding="utf-8")
    print(f"切分结果已保存到 {temp_dir / 'split_metadatas.json'}")

    config = {
        "index_dir": str(INDEX_DIR),
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "bm25_backend": BM25_BACKEND,
        "bm25_recall_k": BM25_RECALL_K,
        "dense_model_path": DENSE_MODEL_PATH,
        "dense_device": INDEX_DEVICE,
        "dense_batch_size": INDEX_BATCH_SIZE,
        "dense_recall_k": DENSE_RECALL_K,
        "rrf_method": RRF_METHOD,
        "rrf_k": RRF_K,
        "rrf_top_k": RRF_TOP_K,
        "rrf_2way_axis": RRF_2WAY_AXIS,
    }

    print("正在构建索引...")
    Retriever.build(docs_loc, docs_content, metadatas, config=config)
    elapsed = time.time() - t0
    print(f"\n索引构建完成，保存到 {INDEX_DIR}，总耗时 {elapsed:.1f}s")


def build_qa_index():
    """从 Excel 构建 QA 索引。"""
    if _check_qa_index_complete(str(QA_INDEX_DIR)):
        print(f"QA 索引已存在于 {QA_INDEX_DIR}，如需重建请先删除该目录。")
        return

    t0 = time.time()
    print("正在加载 QA 数据...")
    qa_data = load_qa_pairs(str(QA_DATA_PATH))
    print(f"共加载 {len(qa_data)} 条 QA 对")

    config = {
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

    print("正在构建 QA 索引...")
    QARetriever.build(qa_data, config)
    elapsed = time.time() - t0
    print(f"\nQA 索引构建完成，保存到 {QA_INDEX_DIR}，总耗时 {elapsed:.1f}s")


if __name__ == "__main__":
    import sys
    args = set(sys.argv[1:])
    if "--qa" in args:
        build_qa_index()
    elif "--reg" in args:
        build_index()
    else:
        build_index()
        build_qa_index()
