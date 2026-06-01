"""Reranker 单元测试：验证 Qwen3-Reranker 的基本功能。"""

import unittest
from rerank.reranker import RERANKER_REGISTRY, Qwen3Reranker


class TestRerankerRegistry(unittest.TestCase):
    """测试 RERANKER_REGISTRY 映射。"""

    def test_registry_contains_all_models(self):
        self.assertIn("bge", RERANKER_REGISTRY)
        self.assertIn("gte", RERANKER_REGISTRY)
        self.assertIn("qwen3", RERANKER_REGISTRY)

    def test_qwen3_key_maps_to_correct_class(self):
        self.assertIs(RERANKER_REGISTRY["qwen3"], Qwen3Reranker)


class TestQwen3Reranker(unittest.TestCase):
    """测试 Qwen3-Reranker 实际推理。"""

    @classmethod
    def setUpClass(cls):
        cls.model_path = "/home/moga/models/reranker/Qwen3-Reranker-0.6B"
        cls.reranker = Qwen3Reranker(cls.model_path, device="cuda")

    def test_rerank_returns_correct_length(self):
        docs = ["文档一", "文档二", "文档三"]
        scores = self.reranker.rerank("测试查询", docs)
        self.assertEqual(len(scores), 3)

    def test_rerank_scores_in_valid_range(self):
        docs = ["电动自行车整车质量应小于55kg"]
        scores = self.reranker.rerank("电动自行车质量要求", docs)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_relevant_score_higher_than_irrelevant(self):
        query = "60岁能考驾照吗"
        relevant = "放宽小型汽车驾驶证申请年龄，70周岁以下均可报考"
        irrelevant = "电动自行车整车质量应小于或等于55kg"
        scores = self.reranker.rerank(query, [relevant, irrelevant])
        self.assertGreater(scores[0], scores[1],
                           f"相关文档得分({scores[0]:.4f})应高于不相关文档({scores[1]:.4f})")

    def test_single_document(self):
        docs = ["机动车来历证明和整车出厂合格证明"]
        scores = self.reranker.rerank("机动车登记", docs)
        self.assertEqual(len(scores), 1)
        self.assertIsInstance(scores[0], float)

    def test_empty_documents(self):
        scores = self.reranker.rerank("测试查询", [])
        self.assertEqual(len(scores), 0)


if __name__ == "__main__":
    unittest.main()
