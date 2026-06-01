"""
端到端测试：验证 RAG Pipeline 完整流程。

由于 question.json 无标准答案，本测试仅验证：
1. Pipeline 端到端可运行（解析 → 建索引 → 检索 → 精排）
2. 返回结果的结构正确性
3. 双路去重、分数降序等逻辑

算法效果需人工审核（通过 report 工具生成 HTML 辅助）。
"""

import unittest
import json
import random
from config import (KB_PATH, QA_PATH, INDEX_DIR, BM25_K1, BM25_B, BM25_BACKEND, BM25_RECALL_K,
                    RERANK_TOP_K, RERANKER_MODEL, RERANKER_MODEL_PATH,
                    DENSE_MODEL_PATH, DENSE_DEVICE, DENSE_RECALL_K,
                    RRF_METHOD, RRF_K, RRF_2WAY_AXIS)
from utils.doc_parser import parse_regulation
from retrieval.retrieve import Retriever
from rerank.reranker import RERANKER_REGISTRY

SEED = 42
SAMPLE_SIZE = 5
REQUIRED_KEYS = ("score", "matched_tokens", "source", "file_name",
                 "chapter", "article_no", "content", "rerank_score")


def _load_sample_questions(path: str, n: int = SAMPLE_SIZE) -> list[dict]:
    """从 question.json 抽样 n 条问题（统一提取 question 字段）。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for rows in data.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            q = row.get("问题") or row.get("question", "")
            if not q:
                continue
            questions.append({
                "question": q,
                "category_1": row.get("一级类目", ""),
                "category_2": row.get("二级类目", ""),
            })
    random.seed(SEED)
    return random.sample(questions, min(n, len(questions)))


class TestE2EPipeline(unittest.TestCase):
    """端到端冒烟测试（含 Reranker 精排）。"""

    @classmethod
    def setUpClass(cls):
        cls.docs_loc, cls.docs_content, cls.metadatas = parse_regulation(str(KB_PATH))
        # 索引持久化：首次构建并保存，后续直接加载
        cls.retriever = Retriever(
            cls.docs_loc, cls.docs_content, cls.metadatas,
            config={
                "index_dir": str(INDEX_DIR),
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B,
                "bm25_backend": BM25_BACKEND,
                "bm25_recall_k": BM25_RECALL_K,
                "dense_model_path": DENSE_MODEL_PATH,
                "dense_device": DENSE_DEVICE,
                "dense_batch_size": 4,
                "dense_recall_k": DENSE_RECALL_K,
                "rrf_method": RRF_METHOD,
                "rrf_k": RRF_K,
                "rrf_2way_axis": RRF_2WAY_AXIS,
            },
        )
        # Reranker 放 GPU（索引已持久化，Dense 模型仅按需编码单条 query，显存充足）
        reranker_cls = RERANKER_REGISTRY[RERANKER_MODEL]
        cls.reranker = reranker_cls(RERANKER_MODEL_PATH, device="cuda")
        cls.reranker_batch_size = 2
        cls.sample_questions = _load_sample_questions(str(QA_PATH))

    def _retrieve_and_rerank(self, query: str, top_k: int = None):
        """完整的 retrieve + rerank 流程。"""
        candidates = self.retriever.retrieve(query, top_k=top_k)
        if not candidates:
            return []
        documents = [r["content"] for r in candidates]
        scores = self.reranker.rerank(query, documents, batch_size=self.reranker_batch_size)
        for r, s in zip(candidates, scores):
            r["rerank_score"] = s
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:RERANK_TOP_K]

    # ---- 结构正确性 ----

    def test_parse_produces_results(self):
        self.assertGreater(len(self.metadatas), 0, "知识库应解析出条款")

    def test_loc_content_metadata_same_length(self):
        self.assertEqual(len(self.docs_loc), len(self.docs_content))
        self.assertEqual(len(self.docs_loc), len(self.metadatas))

    def test_pipeline_returns_results_for_sample_questions(self):
        for item in self.sample_questions:
            with self.subTest(question=item["question"][:30]):
                results = self._retrieve_and_rerank(item["question"])
                self.assertIsInstance(results, list)
                self.assertGreater(len(results), 0, f"应返回结果：{item['question'][:30]}")

    def test_result_structure(self):
        results = self._retrieve_and_rerank(self.sample_questions[0]["question"])
        for r in results:
            for key in REQUIRED_KEYS:
                self.assertIn(key, r, f"结果缺少字段: {key}")

    def test_rerank_scores_descending(self):
        for item in self.sample_questions:
            with self.subTest(question=item["question"][:30]):
                results = self._retrieve_and_rerank(item["question"])
                scores = [r["rerank_score"] for r in results]
                self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rerank_scores_in_valid_range(self):
        results = self._retrieve_and_rerank(self.sample_questions[0]["question"])
        for r in results:
            self.assertGreaterEqual(r["rerank_score"], 0.0)
            self.assertLessEqual(r["rerank_score"], 1.0)

    def test_rerank_top_k_respected(self):
        results = self._retrieve_and_rerank(self.sample_questions[0]["question"])
        self.assertLessEqual(len(results), RERANK_TOP_K)

    def test_no_duplicate_docs(self):
        for item in self.sample_questions:
            with self.subTest(question=item["question"][:30]):
                results = self._retrieve_and_rerank(item["question"])
                keys = [(r["file_name"], r["chapter"], r["article_no"], r["content"])
                        for r in results]
                self.assertEqual(len(keys), len(set(keys)), "结果中不应有完全重复的条款")

    def test_source_field_valid(self):
        for item in self.sample_questions:
            with self.subTest(question=item["question"][:30]):
                results = self._retrieve_and_rerank(item["question"])
                for r in results:
                    self.assertIn(r["source"], {"location", "content", "both"})

    def test_top_k_respected(self):
        """双路召回各用 top_k，四路总结果上限为 4*top_k。"""
        results = self.retriever.retrieve(self.sample_questions[0]["question"], top_k=3)
        self.assertLessEqual(len(results), 12)

    # ---- 辅助人工审核：打印结果摘要 ----

    def test_print_results_for_human_review(self):
        """打印抽样结果供人工审核（不是断言，仅展示）。"""
        print("\n" + "=" * 80)
        print("E2E 端到端检索 + Reranker 精排结果（人工审核用）")
        print("=" * 80)
        for i, item in enumerate(self.sample_questions, 1):
            query = item["question"]
            results = self._retrieve_and_rerank(query, top_k=10)
            print(f"\n【Q{i}】{query}")
            print(f"  类目：{item['category_1']} / {item['category_2']}")
            for j, r in enumerate(results[:3], 1):
                print(f"  Top{j} | rerank={r['rerank_score']:.4f} | "
                      f"rrf={r['score']:.6f} | source={r['source']}")
                print(f"        出处：{r['file_name']} · {r['chapter']} · {r['article_no']}")
                content = r["content"]
                print(f"        内容：{content[:120]}{'...' if len(content) > 120 else ''}")
            print(f"  --- 精排展示 {len(results)} 条 ---")
        print("=" * 80)


class TestRetrieverOnly(unittest.TestCase):
    """验证不带 Reranker 的 Retriever 仍正常工作（回归测试）。"""

    @classmethod
    def setUpClass(cls):
        cls.docs_loc, cls.docs_content, cls.metadatas = parse_regulation(str(KB_PATH))
        cls.retriever = Retriever(
            cls.docs_loc, cls.docs_content, cls.metadatas,
            config={
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B,
                "bm25_backend": BM25_BACKEND,
                "bm25_recall_k": BM25_RECALL_K,
            },
        )
        cls.sample_questions = _load_sample_questions(str(QA_PATH))

    def test_retriever_returns_results(self):
        results = self.retriever.retrieve(self.sample_questions[0]["question"])
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_scores_descending(self):
        results = self.retriever.retrieve(self.sample_questions[0]["question"])
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
