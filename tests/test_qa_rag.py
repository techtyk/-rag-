"""QA RAG 单元测试。"""

import tempfile
from pathlib import Path

import pytest

from retrieval.qa_retrieve import QARetriever, _check_qa_index_complete
from utils.qa_data_loader import load_qa_pairs


# ─── 测试数据 ───

SAMPLE_QA = [
    {"qa_id": 0, "question": "驾驶证怎么换证？", "answer": "到车管所办理换证。", "category": "traffic"},
    {"qa_id": 1, "question": "闯红灯扣几分？", "answer": "闯红灯扣6分。", "category": "traffic"},
    {"qa_id": 2, "question": "地球绕太阳一圈需要多久？", "answer": "约365.25天。", "category": "unrelated"},
]

QA_CONFIG = {
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "bm25_backend": "numpy",
    "bm25_recall_k": 10,
    "dense_model_path": "/home/moga/models/embedding/Qwen3-Embedding-0.6B",
    "dense_device": "cuda",
    "dense_batch_size": 4,
    "dense_recall_k": 10,
    "rrf_k": 60,
    "rrf_top_k": 5,
}


class TestDataLoader:
    def test_load_qa_pairs_from_excel(self):
        path = str(Path(__file__).parent.parent.parent / "data" / "qa_knowledgebase_sample.xlsx")
        if not Path(path).exists():
            pytest.skip("QA Excel 文件不存在")
        qa_pairs = load_qa_pairs(path)
        assert len(qa_pairs) == 10
        for item in qa_pairs:
            assert "qa_id" in item
            assert "question" in item
            assert "answer" in item
            assert "category" in item

    def test_load_qa_pairs_has_traffic_and_unrelated(self):
        path = str(Path(__file__).parent.parent.parent / "data" / "qa_knowledgebase_sample.xlsx")
        if not Path(path).exists():
            pytest.skip("QA Excel 文件不存在")
        qa_pairs = load_qa_pairs(path)
        categories = {item["category"] for item in qa_pairs}
        assert "traffic" in categories
        assert "unrelated" in categories


class TestQARetriever:
    @pytest.fixture(scope="class")
    def qa_retriever(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {**QA_CONFIG, "index_dir": tmpdir}
            retriever = QARetriever.build(SAMPLE_QA, config)
            yield retriever

    def test_build_creates_index_files(self, qa_retriever):
        index_dir = qa_retriever.save.__func__  # 只检查 build 已执行
        assert qa_retriever.faiss_index is not None
        assert qa_retriever.bm25_index is not None
        assert len(qa_retriever.qa_data) == 3

    def test_retrieve_returns_results(self, qa_retriever):
        results = qa_retriever.retrieve("换证流程")
        assert len(results) > 0

    def test_retrieve_scores_descending(self, qa_retriever):
        results = qa_retriever.retrieve("闯红灯")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_exact_match_top1(self, qa_retriever):
        results = qa_retriever.retrieve("闯红灯扣几分？")
        assert results[0]["qa_id"] == 1

    def test_retrieve_no_duplicate_qa_ids(self, qa_retriever):
        results = qa_retriever.retrieve("驾驶证")
        qa_ids = [r["qa_id"] for r in results]
        assert len(qa_ids) == len(set(qa_ids))

    def test_retrieve_respects_top_k(self, qa_retriever):
        results = qa_retriever.retrieve("怎么办理")
        assert len(results) <= qa_retriever.rrf_top_k


class TestQARetrieverPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {**QA_CONFIG, "index_dir": tmpdir}
            original = QARetriever.build(SAMPLE_QA, config)

            # 从磁盘重新加载
            loaded = QARetriever.load(tmpdir, config)

            assert len(loaded.qa_data) == len(original.qa_data)
            for o, l in zip(original.qa_data, loaded.qa_data):
                assert o["question"] == l["question"]
                assert o["answer"] == l["answer"]

    def test_check_index_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert not _check_qa_index_complete(tmpdir)
            config = {**QA_CONFIG, "index_dir": tmpdir}
            QARetriever.build(SAMPLE_QA, config)
            assert _check_qa_index_complete(tmpdir)

    def test_load_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {**QA_CONFIG, "index_dir": tmpdir}
            QARetriever.build(SAMPLE_QA, config)
            loaded = QARetriever.load(tmpdir, config)
            results = loaded.retrieve("地球公转")
            assert len(results) > 0
            assert results[0]["qa_id"] == 2
