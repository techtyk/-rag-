import json
from pathlib import Path
from typing import List, Dict, Optional

from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever

# 索引文件完整性检查清单
INDEX_FILES = [
    "bm25/loc", "bm25/content",
    "bm25/tokenized_loc.json", "bm25/tokenized_content.json",
    "dense/faiss_loc.index", "dense/faiss_content.index",
    "metadatas.json",
]


def _check_index_complete(index_dir: str) -> bool:
    """检查索引目录中是否包含所有必需文件（含 metadatas）。"""
    d = Path(index_dir)
    return all((d / f).exists() for f in INDEX_FILES)


def _save_metadatas(index_dir: str, metadatas: List[Dict]):
    """将 metadatas 持久化到磁盘，供后续加载索引时使用。"""
    # 确保索引目录存在 -> 序列化 -> 写文件
    d = Path(index_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadatas.json").write_text(
        json.dumps(metadatas, ensure_ascii=False), encoding="utf-8")


def _load_metadatas(index_dir: str) -> List[Dict]:
    """从磁盘加载 metadatas。"""
    #   1. Path(index_dir) / "metadatas.json" → 拼出文件路径
    #   2. .read_text(encoding="utf-8") → 读取整个文件为字符串
    #   3. json.loads(...) → 把 JSON 字符串反序列化回 Python 列表
    return json.loads((Path(index_dir) / "metadatas.json").read_text(encoding="utf-8"))


class Retriever:
    """统一检索入口，管理 BM25 + Dense 的多路召回与 RRF 融合。"""

    def __init__(self, docs_loc: List[str], docs_content: List[str],
                 metadatas: List[Dict], config: dict):
        self.metadatas = metadatas
        self.bm25_recall_k = config.get("bm25_recall_k", 10)
        self.dense_recall_k = config.get("dense_recall_k", 10)
        self.rrf_method = config.get("rrf_method", "4way")
        self.rrf_k = config.get("rrf_k", 60)
        self.rrf_top_k = config.get("rrf_top_k", 15)
        self.rrf_2way_axis = config.get("rrf_2way_axis", "by_index")

        index_dir = config.get("index_dir")

        # BM25 检索器
        if index_dir and _check_index_complete(index_dir):
            self.bm25 = BM25Retriever.load(index_dir, metadatas)
        else:
            self.bm25 = BM25Retriever(
                docs_loc, docs_content, metadatas,
                k1=config.get("bm25_k1", 1.5),
                b=config.get("bm25_b", 0.75),
                backend=config.get("bm25_backend", "numpy"),
                index_dir=index_dir,
            )

        # Dense 检索器（可选，需提供 dense_model_path）
        self.dense = None
        model_path = config.get("dense_model_path")
        if model_path:
            if index_dir and _check_index_complete(index_dir):
                self.dense = DenseRetriever.load(
                    index_dir, metadatas,
                    model_path=model_path,
                    device=config.get("dense_device", "cuda"),
                )
            else:
                self.dense = DenseRetriever(
                    docs_loc, docs_content, metadatas,
                    model_path=model_path,
                    device=config.get("dense_device", "cuda"),
                    batch_size=config.get("dense_batch_size", 32),
                    index_dir=index_dir,
                )

        # 持久化 metadatas（新建索引时）
        if index_dir and not (Path(index_dir) / "metadatas.json").exists():
            _save_metadatas(index_dir, metadatas)

    @classmethod
    def load(cls, index_dir: str, config: dict) -> "Retriever":
        """从磁盘加载完整索引（含 metadatas），不需要传入 docs。"""
        metadatas = _load_metadatas(index_dir)
        obj = cls.__new__(cls)
        obj.metadatas = metadatas
        obj.bm25_recall_k = config.get("bm25_recall_k", 10)
        obj.dense_recall_k = config.get("dense_recall_k", 10)
        obj.rrf_method = config.get("rrf_method", "4way")
        obj.rrf_k = config.get("rrf_k", 60)
        obj.rrf_top_k = config.get("rrf_top_k", 15)
        obj.rrf_2way_axis = config.get("rrf_2way_axis", "by_index")

        obj.bm25 = BM25Retriever.load(index_dir, metadatas)

        obj.dense = None
        model_path = config.get("dense_model_path")
        if model_path:
            obj.dense = DenseRetriever.load(
                index_dir, metadatas,
                model_path=model_path,
                device=config.get("dense_device", "cuda"),
            )
        print(f"索引已从 {index_dir} 完整加载（{len(metadatas)} 条条款）")
        return obj

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """多路召回并 RRF 融合，返回候选列表。"""
        _k: int = top_k if top_k is not None else max(self.bm25_recall_k, self.dense_recall_k)

        # BM25 双路召回
        bm25_results = self.bm25.search(query, top_k=_k)

        if not self.dense:
            return bm25_results

        # Dense 双路召回
        dense_results = self.dense.search(query, top_k=_k)

        if self.rrf_method == "2way":
            return self._fuse_2way(bm25_results, dense_results, self.rrf_2way_axis)
        else:
            return self._fuse_4way(bm25_results, dense_results)

    def _rrf_score(self, ranked_lists: List[List[Dict]]) -> Dict[tuple, float]:
        """对多路排序结果计算 RRF 分数。返回 {doc_key: rrf_score}。"""
        scores = {}
        for lst in ranked_lists:
            for rank, r in enumerate(lst, start=1):
                key = (r["file_name"], r["chapter"], r["article_no"])
                scores[key] = scores.get(key, 0) + 1.0 / (self.rrf_k + rank)
        return scores

    def _fuse_4way(self, bm25_results: List[Dict],
                   dense_results: List[Dict]) -> List[Dict]:
        """四路 RRF：BM25定位、BM25内容、Dense定位、Dense内容 直接融合。

        每个结果内部的 scores dict 已包含双路分数，这里拆成四路独立排序列表。
        """
        # 拆分 BM25 结果为定位/内容两路
        bm25_loc = [r for r in bm25_results if "location" in r.get("scores", {})]
        bm25_loc.sort(key=lambda x: x["scores"].get("location", 0), reverse=True)
        bm25_cont = [r for r in bm25_results if "content" in r.get("scores", {})]
        bm25_cont.sort(key=lambda x: x["scores"].get("content", 0), reverse=True)

        # 拆分 Dense 结果为定位/内容两路
        dense_loc = [r for r in dense_results if "location" in r.get("scores", {})]
        dense_loc.sort(key=lambda x: x["scores"].get("location", 0), reverse=True)
        dense_cont = [r for r in dense_results if "content" in r.get("scores", {})]
        dense_cont.sort(key=lambda x: x["scores"].get("content", 0), reverse=True)

        # 四路 RRF
        rrf_scores = self._rrf_score([bm25_loc, bm25_cont, dense_loc, dense_cont])

        # 合并去重，保留原始元数据
        seen = {}
        for r in bm25_results + dense_results:
            key = (r["file_name"], r["chapter"], r["article_no"])
            if key not in seen:
                seen[key] = r

        # 赋 RRF 分数
        output = []
        for key, rrf_s in rrf_scores.items():
            r = seen[key].copy()
            r["score"] = rrf_s
            output.append(r)

        output.sort(key=lambda x: x["score"], reverse=True)
        return output[:self.rrf_top_k]

    def _fuse_2way(self, bm25_results: List[Dict],
                   dense_results: List[Dict], axis: str = "by_index") -> List[Dict]:
        """两路分组 RRF，分组方向由 axis 决定。

        axis="by_index":     定位组(BM25+Dense) RRF + 内容组(BM25+Dense) RRF
        axis="by_retriever": BM25组(定位+正文) RRF + Dense组(定位+正文) RRF
        """
        # 拆分各路结果
        bm25_loc = [r for r in bm25_results if "location" in r.get("scores", {})]
        bm25_loc.sort(key=lambda x: x["scores"].get("location", 0), reverse=True)
        bm25_cont = [r for r in bm25_results if "content" in r.get("scores", {})]
        bm25_cont.sort(key=lambda x: x["scores"].get("content", 0), reverse=True)

        dense_loc = [r for r in dense_results if "location" in r.get("scores", {})]
        dense_loc.sort(key=lambda x: x["scores"].get("location", 0), reverse=True)
        dense_cont = [r for r in dense_results if "content" in r.get("scores", {})]
        dense_cont.sort(key=lambda x: x["scores"].get("content", 0), reverse=True)

        if axis == "by_retriever":
            group_a = self._rrf_score([bm25_loc, bm25_cont])   # BM25组
            group_b = self._rrf_score([dense_loc, dense_cont])  # Dense组
        else:
            group_a = self._rrf_score([bm25_loc, dense_loc])    # 定位组
            group_b = self._rrf_score([bm25_cont, dense_cont])  # 内容组

        # 合并两组 RRF 结果
        combined = {}
        for key, score in group_a.items():
            combined[key] = score
        for key, score in group_b.items():
            combined[key] = combined.get(key, 0) + score

        # 去重，保留原始元数据
        seen = {}
        for r in bm25_results + dense_results:
            key = (r["file_name"], r["chapter"], r["article_no"])
            if key not in seen:
                seen[key] = r

        output = []
        for key, rrf_s in combined.items():
            r = seen[key].copy()
            r["score"] = rrf_s
            output.append(r)

        output.sort(key=lambda x: x["score"], reverse=True)
        return output[:self.rrf_top_k]
