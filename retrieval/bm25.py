import json
from pathlib import Path
from typing import List, Dict, Optional

import bm25s
from utils.tokenizer import tokenize_for_doc, tokenize_for_query


class BM25Retriever:
    def __init__(self, docs_loc: List[str], docs_content: List[str],
                 metadatas: List[Dict],
                 k1: float = 1.5, b: float = 0.75, backend: str = "numma",
                 index_dir: Optional[str] = None):
        self.metadatas = metadatas
        self.tokenized_loc = [tokenize_for_doc(doc) for doc in docs_loc]
        self.tokenized_content = [tokenize_for_doc(doc) for doc in docs_content]
        self.retriever_loc = self._build_index(self.tokenized_loc, k1, b, backend, "定位")
        self.retriever_content = self._build_index(self.tokenized_content, k1, b, backend, "正文")
        if index_dir:
            self.save(index_dir)

    @classmethod
    def load(cls, index_dir: str, metadatas: List[Dict]) -> "BM25Retriever":
        """从磁盘加载已保存的 BM25 索引，跳过分词和构建。"""
        obj = cls.__new__(cls) # 绕过 __init__，直接创建实例并赋属性
        obj.metadatas = metadatas
        d = Path(index_dir) / "bm25"
        obj.tokenized_loc = json.loads((d / "tokenized_loc.json").read_text(encoding="utf-8"))
        obj.tokenized_content = json.loads((d / "tokenized_content.json").read_text(encoding="utf-8"))
        obj.retriever_loc = bm25s.BM25.load(str(d / "loc"), load_corpus=False)
        obj.retriever_content = bm25s.BM25.load(str(d / "content"), load_corpus=False)
        print(f"BM25 索引已从 {d} 加载")
        return obj

    def save(self, index_dir: str):
        """将 BM25 索引和分词结果持久化到磁盘。"""
        d = Path(index_dir) / "bm25"
        d.mkdir(parents=True, exist_ok=True)
        self.retriever_loc.save(str(d / "loc"), corpus=None)
        self.retriever_content.save(str(d / "content"), corpus=None)
        (d / "tokenized_loc.json").write_text(
            json.dumps(self.tokenized_loc, ensure_ascii=False), encoding="utf-8")
        (d / "tokenized_content.json").write_text(
            json.dumps(self.tokenized_content, ensure_ascii=False), encoding="utf-8")
        print(f"BM25 索引已保存到 {d}")

    def _build_index(self, tokenized_corpus: List[List[str]],
                     k1: float, b: float, backend: str, label: str) -> bm25s.BM25:
        try:
            retriever = bm25s.BM25(method="lucene", k1=k1, b=b, backend=backend)
            retriever.index(tokenized_corpus)
            print(f"BM25 {label}索引构建完成，后端：{backend}，文档数：{len(tokenized_corpus)}")
        except (ImportError, RuntimeError, ValueError) as e:
            print(f"无法使用 {backend}：{e}，切换到 numpy")
            retriever = bm25s.BM25(method="lucene", k1=k1, b=b, backend="numpy")
            retriever.index(tokenized_corpus)
            print(f"BM25 {label}索引构建完成，后端：numpy，文档数：{len(tokenized_corpus)}")
        return retriever

    def search(self, query: str, top_k: int = 10) -> List[Dict]: # top_k 是每路最多返回的条数；返回值是字典列表，每个字典是一条命中文档
        query_tokens = tokenize_for_query(query)
        query_tokens_2d = [query_tokens]
        k = min(top_k, len(self.metadatas))

        # 初始化收集字典：收集双路分数，同一 chunk 可能被两个索引同时命中
        doc_scores = {}  # key 是文档索引（0, 1, 2...），value 是各路的分数。比如 {0: {"location": 0.188, "content": 0.053}}
        doc_matched = {}  # key 是文档索引，value 是各路匹配词。比如 {0: {"location": ["第一章"], "content": ["内容"]}}
        for retriever, tokenized_corpus, source in [
            (self.retriever_loc, self.tokenized_loc, "location"),
            (self.retriever_content, self.tokenized_content, "content"),
        ]:
            results, scores = retriever.retrieve(query_tokens_2d, k=k)
            for i in range(len(results[0])):
                score_val = float(scores[0, i])
                if score_val <= 0:
                    continue
                doc_idx = int(results[0, i])
                doc_scores.setdefault(doc_idx, {})[source] = score_val
                doc_token_set = set(tokenized_corpus[doc_idx])
                matched = sorted(t for t in query_tokens if t in doc_token_set)
                doc_matched.setdefault(doc_idx, {})[source] = matched

        output = []
        for idx, score_dict in doc_scores.items():
            sources = list(score_dict.keys())
            output.append({
                "scores": score_dict,
                "source": sources[0] if len(sources) == 1 else "both",
                "matched_tokens": doc_matched.get(idx, {}),
                **self.metadatas[idx],
            })

        return output


if __name__ == "__main__":
    # 简单测试
    docs_loc = ["doc1 第一章 第一条", "doc2 第一章 第二条", "doc3 第二章 第一条"]
    docs_content = ["这是第一条的内容", "这是第二条的内容", "这是第三条的内容"]
    metadatas = [{"file_name": "doc1", "chapter": "第一章", "article_no": "第一条"},
                 {"file_name": "doc2", "chapter": "第一章", "article_no": "第二条"},
                 {"file_name": "doc3", "chapter": "第二章", "article_no": "第一条"}]
    retriever = BM25Retriever(docs_loc, docs_content, metadatas)
    query = "第一章的内容"
    results = retriever.search(query)
    for r in results:
        print(r)