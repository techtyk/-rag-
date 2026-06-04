"""QA 检索器：对问题文本进行 BM25 + Dense 双路检索 + 2 路 RRF 融合。"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import bm25s
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.tokenizer import tokenize_for_doc, tokenize_for_query


# QA 索引文件完整性检查清单
QA_INDEX_FILES = [
    "bm25/qa_bm25",
    "bm25/tokenized_questions.json",
    "dense/faiss_questions.index",
    "qa_data.json",
]


def _check_qa_index_complete(index_dir: str) -> bool:
    d = Path(index_dir)
    return all((d / f).exists() for f in QA_INDEX_FILES)


class QARetriever:
    """QA 检索器：单问题索引上的 BM25 + Dense 2 路 RRF。"""

    def __init__(self, qa_data: List[Dict], bm25_index, faiss_index,
                 tokenized_questions: List[List[str]], model: Optional[SentenceTransformer],
                 config: dict):
        self.qa_data = qa_data
        self.bm25_index = bm25_index
        self.faiss_index = faiss_index
        self.tokenized_questions = tokenized_questions
        self.model = model
        self.model_path = config.get("dense_model_path", "")
        self._device = config.get("dense_device", "cuda")
        self._apply_config(config)

    def _apply_config(self, config: dict):
        self.bm25_recall_k = config.get("bm25_recall_k", 10)
        self.dense_recall_k = config.get("dense_recall_k", 10)
        self.rrf_k = config.get("rrf_k", 60)
        self.rrf_top_k = config.get("rrf_top_k", 5)

    @classmethod
    def build(cls, qa_data: List[Dict], config: dict) -> "QARetriever":
        """从 QA 数据构建 BM25 + Dense 索引并持久化。"""
        index_dir = config.get("index_dir")
        questions = [item["question"] for item in qa_data]

        # BM25
        tokenized = [tokenize_for_doc(q) for q in questions]
        k1 = config.get("bm25_k1", 1.5)
        b = config.get("bm25_b", 0.75)
        backend = config.get("bm25_backend", "numpy")
        bm25_index = bm25s.BM25(method="lucene", k1=k1, b=b, backend=backend)
        bm25_index.index(tokenized)
        print(f"QA BM25 索引构建完成，文档数：{len(questions)}")

        # Dense
        model_path = config.get("dense_model_path")
        device = config.get("dense_device", "cuda")
        batch_size = config.get("dense_batch_size", 4)
        model = SentenceTransformer(model_path, device=device,
                                    model_kwargs={"attn_implementation": "sdpa"})
        dim = model.get_embedding_dimension()
        max_seq = model[0].max_seq_length if hasattr(model[0], 'max_seq_length') else 512
        truncated = [q[:max_seq * 4] for q in questions]
        embs = model.encode(truncated, batch_size=batch_size,
                            show_progress_bar=False, normalize_embeddings=True)
        embs = np.array(embs, dtype="float32")
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embs)
        print(f"QA Dense 索引构建完成，维度={dim}，文档数：{len(questions)}")

        obj = cls(qa_data, bm25_index, faiss_index, tokenized, model, config)

        if index_dir:
            obj.save(index_dir)

        return obj

    @classmethod
    def load(cls, index_dir: str, config: dict,
             shared_model: Optional[SentenceTransformer] = None) -> "QARetriever":
        """从磁盘加载索引。可接受外部共享的 SentenceTransformer 模型实例。"""
        d = Path(index_dir)
        qa_data = json.loads((d / "qa_data.json").read_text(encoding="utf-8"))
        tokenized = json.loads(
            (d / "bm25" / "tokenized_questions.json").read_text(encoding="utf-8"))
        bm25_index = bm25s.BM25.load(str(d / "bm25" / "qa_bm25"), load_corpus=False)
        faiss_index = faiss.read_index(str(d / "dense" / "faiss_questions.index"))

        model = shared_model
        if model is None:
            model_path = config.get("dense_model_path")
            if model_path:
                device = config.get("dense_device", "cuda")
                model = SentenceTransformer(
                    model_path, device=device,
                    model_kwargs={"attn_implementation": "sdpa"})
                print(f"QA Dense 模型已独立加载")

        print(f"QA 索引已从 {index_dir} 加载（{len(qa_data)} 条 QA）")
        return cls(qa_data, bm25_index, faiss_index, tokenized, model, config)

    def save(self, index_dir: str):
        d = Path(index_dir)
        d.mkdir(parents=True, exist_ok=True)

        # QA 数据
        (d / "qa_data.json").write_text(
            json.dumps(self.qa_data, ensure_ascii=False), encoding="utf-8")

        # BM25
        bm25_dir = d / "bm25"
        bm25_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_index.save(str(bm25_dir / "qa_bm25"), corpus=None)
        (bm25_dir / "tokenized_questions.json").write_text(
            json.dumps(self.tokenized_questions, ensure_ascii=False), encoding="utf-8")

        # Dense
        dense_dir = d / "dense"
        dense_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(dense_dir / "faiss_questions.index"))

        print(f"QA 索引已保存到 {d}")

    def retrieve(self, query: str, query_emb: Optional[np.ndarray] = None) -> List[Dict]:
        """BM25 + Dense 2 路 RRF 融合检索。可接受预编码的 query embedding。"""
        k = max(self.bm25_recall_k, self.dense_recall_k)
        n = len(self.qa_data)

        # BM25 检索
        query_tokens = tokenize_for_query(query)
        bm25_results, bm25_scores = self.bm25_index.retrieve([query_tokens], k=min(k, n))
        bm25_ranked = []
        for rank, (idx, score) in enumerate(zip(bm25_results[0], bm25_scores[0]), start=1):
            idx = int(idx)
            if float(score) <= 0:
                continue
            bm25_ranked.append({"qa_id": idx, "bm25_rank": rank})

        # Dense 检索
        if query_emb is None and self.model is not None:
            query_emb = np.array(
                self.model.encode([query], normalize_embeddings=True,
                                  device=self._device),
                dtype="float32")
        dense_ranked = []
        if query_emb is not None:
            scores, indices = self.faiss_index.search(query_emb, k=min(k, n))
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
                if idx == -1 or float(score) <= 0:
                    continue
                dense_ranked.append({"qa_id": int(idx), "dense_rank": rank})

        # 2 路 RRF 融合
        rrf_scores: Dict[int, float] = {}
        for item in bm25_ranked:
            rrf_scores[item["qa_id"]] = rrf_scores.get(item["qa_id"], 0) + 1.0 / (self.rrf_k + item["bm25_rank"])
        for item in dense_ranked:
            rrf_scores[item["qa_id"]] = rrf_scores.get(item["qa_id"], 0) + 1.0 / (self.rrf_k + item["dense_rank"])

        # 组装结果
        output = []
        for qa_id, rrf_score in rrf_scores.items():
            item = self.qa_data[qa_id].copy()
            item["score"] = rrf_score
            output.append(item)

        output.sort(key=lambda x: x["score"], reverse=True)
        return output[:self.rrf_top_k]
