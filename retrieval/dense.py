import gc
import time
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, docs_loc: List[str], docs_content: List[str],
                 metadatas: List[Dict],
                 model_path: str = "/home/moga/models/BGE-M3",
                 device: str = "cuda",
                 batch_size: int = 4,
                 index_dir: Optional[str] = None):
        self.metadatas = metadatas
        self.model_path = model_path
        self.batch_size = batch_size
        self._device = device
        self._loaded_from_disk = False

        self._load_model()
        self.loc_index = self._build_index(docs_loc, "定位")
        self.cont_index = self._build_index(docs_content, "内容")

        if index_dir:
            self.save(index_dir)

    @classmethod
    def load(cls, index_dir: str, metadatas: List[Dict],
             model_path: str = "", device: str = "cuda") -> "DenseRetriever":
        """从磁盘加载已保存的 FAISS 索引，模型在首次 search 时按需加载到 CPU。"""
        obj = cls.__new__(cls)
        obj.metadatas = metadatas
        obj.model_path = model_path
        obj.batch_size = 4
        obj._device = "cpu"
        obj.model = None
        obj._loaded_from_disk = True

        d = Path(index_dir) / "dense"
        obj.loc_index = faiss.read_index(str(d / "faiss_loc.index"))
        obj.cont_index = faiss.read_index(str(d / "faiss_content.index"))
        print(f"Dense FAISS 索引已从 {d} 加载")
        return obj

    def _load_model(self):
        print(f"Dense: 正在加载模型 {self.model_path} ...")
        t0 = time.time()
        self.model = SentenceTransformer(self.model_path, device=self._device,
                                          model_kwargs={"attn_implementation": "sdpa"})
        self.dim = self.model.get_embedding_dimension()
        print(f"Dense: 模型加载完成，维度={self.dim}，耗时 {time.time()-t0:.1f}s")

    def _ensure_model(self):
        """按需加载模型（用于从磁盘加载索引后的首次 search）。"""
        if self.model is None:
            self._load_model()

    def save(self, index_dir: str):
        """将 FAISS 索引持久化到磁盘。"""
        d = Path(index_dir) / "dense"
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.loc_index, str(d / "faiss_loc.index"))
        faiss.write_index(self.cont_index, str(d / "faiss_content.index"))
        print(f"Dense FAISS 索引已保存到 {d}")

    def _build_index(self, texts: List[str], label: str):
        print(f"Dense: 正在编码 {label}文本（{len(texts)} 条）...")
        t0 = time.time()
        max_seq = self.model[0].max_seq_length if hasattr(self.model[0], 'max_seq_length') else 512
        truncated = [t[:max_seq * 4] for t in texts]
        embs = self.model.encode(truncated, batch_size=self.batch_size,
                                 show_progress_bar=True, normalize_embeddings=True)
        embs = np.array(embs, dtype="float32")
        index = faiss.IndexFlatIP(self.dim)
        index.add(embs)
        del embs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Dense: {label}索引构建完成，耗时 {time.time()-t0:.1f}s")
        return index

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        self._ensure_model()
        k = min(top_k, len(self.metadatas))
        # 如果模型是从磁盘加载（_loaded_from_disk），在 CPU 上编码 query
        # 避免占用 GPU 显存影响 Reranker
        encode_device = "cpu" if getattr(self, "_loaded_from_disk", False) else None
        query_emb = np.array(
            self.model.encode([query], normalize_embeddings=True,
                              device=encode_device),
            dtype="float32",
        )

        # 收集双路分数，同一 chunk 可能被两个索引同时命中
        doc_scores = {}  # idx -> {source: score}
        for index, source in [(self.loc_index, "location"), (self.cont_index, "content")]:
            scores, indices = index.search(query_emb, k=k)
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                score_val = float(score)
                if score_val <= 0:
                    continue
                doc_scores.setdefault(idx, {})[source] = score_val

        output = []
        for idx, score_dict in doc_scores.items():
            sources = list(score_dict.keys())
            output.append({
                "scores": score_dict,
                "source": sources[0] if len(sources) == 1 else "both",
                "matched_tokens": {},
                **self.metadatas[idx],
            })

        return output
