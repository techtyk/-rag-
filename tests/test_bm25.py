import unittest
from retrieval.bm25 import BM25Retriever


# 小型测试语料
DOCS_CONTENT = [
    "电动自行车整车质量应小于或等于55kg",
    "机动车来历证明和整车出厂合格证明",
    "脚踏骑行装置是电动自行车的必备配置",
    "整车质量标准限制为55公斤和63公斤两种",
    "登记规定要求机动车必须进行注册登记",
]
DOCS_LOC = [
    "test.json 第一章 第一条",
    "test.json 第一章 第二条",
    "test.json 第二章 第三条",
    "test.json 第二章 第四条",
    "test.json 第三章 第五条",
]
METADATAS = [
    {"file_name": "test.json", "chapter": "第一章", "article_no": "第一条", "content": DOCS_CONTENT[0]},
    {"file_name": "test.json", "chapter": "第一章", "article_no": "第二条", "content": DOCS_CONTENT[1]},
    {"file_name": "test.json", "chapter": "第二章", "article_no": "第三条", "content": DOCS_CONTENT[2]},
    {"file_name": "test.json", "chapter": "第二章", "article_no": "第四条", "content": DOCS_CONTENT[3]},
    {"file_name": "test.json", "chapter": "第三章", "article_no": "第五条", "content": DOCS_CONTENT[4]},
]


class TestBM25RetrieverBuild(unittest.TestCase):
    def test_build_with_numpy(self):
        retriever = BM25Retriever.build(DOCS_LOC, DOCS_CONTENT, METADATAS, backend="numpy")
        self.assertIsNotNone(retriever.retriever_loc)
        self.assertIsNotNone(retriever.retriever_content)
        self.assertEqual(len(retriever.tokenized_loc), 5)
        self.assertEqual(len(retriever.tokenized_content), 5)

    def test_fallback_on_bad_backend(self):
        retriever = BM25Retriever.build(DOCS_LOC, DOCS_CONTENT, METADATAS, backend="nonexistent")
        self.assertIsNotNone(retriever.retriever_loc)
        self.assertIsNotNone(retriever.retriever_content)


class TestBM25RetrieverSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = BM25Retriever.build(DOCS_LOC, DOCS_CONTENT, METADATAS, backend="numpy")

    def test_dual_recall_dedup(self):
        results = self.retriever.search("第一章", top_k=5)
        doc_indices = set()
        for r in results:
            key = (r["file_name"], r["chapter"], r["article_no"])
            doc_indices.add(key)
        self.assertEqual(len(doc_indices), len(results), "结果应无重复")

    def test_result_has_source(self):
        results = self.retriever.search("整车质量", top_k=3)
        for r in results:
            self.assertIn("source", r)
            self.assertIn(r["source"], {"location", "content", "both"})

    def test_result_structure(self):
        results = self.retriever.search("整车质量", top_k=1)
        r = results[0]
        for key in ("scores", "matched_tokens", "source", "file_name", "chapter", "article_no", "content"):
            self.assertIn(key, r)

    def test_scores_per_source(self):
        results = self.retriever.search("整车质量", top_k=3)
        for r in results:
            self.assertIsInstance(r["scores"], dict)
            for source, val in r["scores"].items():
                self.assertIn(source, {"location", "content"})
                self.assertGreater(val, 0)

    def test_top_k_larger_than_corpus(self):
        results = self.retriever.search("整车质量", top_k=100)
        # 零分过滤：只有匹配 query 的文档才返回（"整车质量" 命中 3 条）
        self.assertLessEqual(len(results), 5)
        self.assertGreater(len(results), 0)

    def test_empty_query(self):
        results = self.retriever.search("", top_k=3)
        self.assertIsInstance(results, list)

    def test_no_match_query(self):
        results = self.retriever.search("量子计算机", top_k=3)
        self.assertIsInstance(results, list)
        for r in results:
            self.assertEqual(r["matched_tokens"], {})

    def test_location_recall(self):
        results = self.retriever.search("第一章", top_k=3)
        sources = [r["source"] for r in results]
        self.assertIn("location", sources)


if __name__ == "__main__":
    unittest.main()
