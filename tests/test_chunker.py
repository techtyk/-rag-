"""切分模块单元测试。"""
import sys
import unittest

sys.path.insert(0, '.')

from utils.chunker import (
    chunk_documents,
    _detect_chunk_type,
    _split_tab_table,
    _split_space_table,
    _split_prose,
)


class TestDetectChunkType(unittest.TestCase):

    def test_tab_table(self):
        content = "序号\t项目\t合格要求\n1\t车辆识别代号\t汽车..."
        self.assertEqual(_detect_chunk_type(content), "tab_table")

    def test_space_table(self):
        content = "序号业务网点名称详细地址\n1芜湖市辉泰..."
        self.assertEqual(_detect_chunk_type(content), "space_table")

    def test_prose(self):
        content = "本标准的全部技术内容为强制性，按照 GB/T 1.1—2009 给出的规则起草。"
        self.assertEqual(_detect_chunk_type(content), "prose")

    def test_prose_with_newlines(self):
        content = "办理变更备案的业务流程和具体事项为：\n（一）...\n（二）..."
        self.assertEqual(_detect_chunk_type(content), "prose")


class TestSplitTabTable(unittest.TestCase):

    def test_basic_split(self):
        header = "序号\t项目\t合格要求"
        rows = [f"{i}\t项目{i}\t{'内容' * 30}" for i in range(1, 20)]
        content = header + "\n" + "\n".join(rows)
        chunks = _split_tab_table(content, chunk_size=200)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)
            self.assertTrue(chunk.startswith(header))

    def test_skip_continuation_header(self):
        header = "序号\t项目\t合格要求"
        data = "1\t项目1\t内容\n表A.1（续）\n序号\t项目\t合格要求\n2\t项目2\t内容"
        content = header + "\n" + data
        chunks = _split_tab_table(content, chunk_size=500)
        for chunk in chunks:
            self.assertNotIn("表A.1（续）", chunk)

    def test_small_table_no_split(self):
        content = "序号\t项目\n1\t项目1\n2\t项目2"
        chunks = _split_tab_table(content, chunk_size=500)
        self.assertEqual(len(chunks), 1)


class TestSplitSpaceTable(unittest.TestCase):

    def test_basic_split(self):
        header = "序号业务网点名称详细地址"
        rows = [f"{i}芜湖市某某公司安徽省芜湖市{'某区' * 10}" for i in range(1, 20)]
        content = header + "\n" + "\n".join(rows)
        chunks = _split_space_table(content, chunk_size=200)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)
            self.assertTrue(chunk.startswith(header))

    def test_filter_separator_lines(self):
        header = "序号名称"
        data = "1项目1\n---\n2项目2\n---\n3项目3"
        content = header + "\n" + data
        chunks = _split_space_table(content, chunk_size=500)
        for chunk in chunks:
            self.assertNotIn("---", chunk)

    def test_filter_empty_lines(self):
        header = "序号名称"
        data = "1项目1\n\n\n2项目2"
        content = header + "\n" + data
        chunks = _split_space_table(content, chunk_size=500)
        self.assertEqual(len(chunks), 1)
        self.assertNotIn("\n\n", chunks[0])


class TestSplitProse(unittest.TestCase):

    def test_split_by_period(self):
        sentences = ["这是第{}句话，内容比较长。".format(i) * 5 for i in range(10)]
        content = "".join(sentences)
        chunks = _split_prose(content, chunk_size=200, chunk_overlap=0)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)

    def test_split_by_newline_when_no_period(self):
        lines = ["第{}行内容".format(i) * 10 for i in range(20)]
        content = "\n".join(lines)
        chunks = _split_prose(content, chunk_size=200, chunk_overlap=0)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)

    def test_no_overlap_on_sentence_split(self):
        content = "第一句话。" + "内容" * 80 + "。" + "第二部分。" + "内容" * 80 + "。"
        chunks = _split_prose(content, chunk_size=200, chunk_overlap=50)
        # 句号分隔不应产生 overlap
        for i in range(1, len(chunks)):
            overlap_text = chunks[i][:50]
            self.assertNotEqual(overlap_text, chunks[i - 1][-50:])

    def test_char_truncation_fallback(self):
        # 无任何分隔符的长文本
        content = "内容" * 500
        chunks = _split_prose(content, chunk_size=200, chunk_overlap=0)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)

    def test_short_content_unchanged(self):
        content = "这是一段短文本。"
        chunks = _split_prose(content, chunk_size=512, chunk_overlap=50)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], content)


class TestChunkDocuments(unittest.TestCase):

    def test_pass_through_short_docs(self):
        docs_loc = ["file1 ch1 art1", "file1 ch1 art2"]
        docs_content = ["短文本1", "短文本2"]
        metadatas = [
            {"file_name": "file1", "chapter": "ch1", "article_no": "art1", "content": "短文本1"},
            {"file_name": "file1", "chapter": "ch1", "article_no": "art2", "content": "短文本2"},
        ]
        out_loc, out_content, out_metas = chunk_documents(
            docs_loc, docs_content, metadatas, chunk_size=512)
        self.assertEqual(len(out_metas), 2)
        for m in out_metas:
            self.assertNotIn("chunk_seq", m)

    def test_split_long_doc(self):
        docs_loc = ["file1 ch1 art1"]
        long_content = "这是一段很长的文本。" * 100
        docs_content = [long_content]
        metadatas = [{"file_name": "file1", "chapter": "ch1",
                      "article_no": "art1", "content": long_content}]
        out_loc, out_content, out_metas = chunk_documents(
            docs_loc, docs_content, metadatas, chunk_size=200)
        self.assertGreater(len(out_metas), 1)
        for m in out_metas:
            self.assertIn("chunk_seq", m)
            self.assertIn("chunk_total", m)
        # chunk_seq 从 0 递增
        seqs = [m["chunk_seq"] for m in out_metas]
        self.assertEqual(seqs, list(range(len(seqs))))
        # chunk_total 一致
        totals = {m["chunk_total"] for m in out_metas}
        self.assertEqual(len(totals), 1)

    def test_loc_suffix(self):
        docs_loc = ["file1 ch1 art1"]
        long_content = "这是一段很长的文本。" * 100
        docs_content = [long_content]
        metadatas = [{"file_name": "file1", "chapter": "ch1",
                      "article_no": "art1", "content": long_content}]
        out_loc, _, _ = chunk_documents(
            docs_loc, docs_content, metadatas, chunk_size=200)
        self.assertGreater(len(out_loc), 1)
        for loc in out_loc:
            self.assertRegex(loc, r"\[chunk \d+/\d+\]$")

    def test_output_count(self):
        short_content = "短文本"
        long_content = "这是一段很长的文本。" * 100
        docs_loc = ["file1 ch1 art1", "file1 ch1 art2"]
        docs_content = [short_content, long_content]
        metadatas = [
            {"file_name": "file1", "chapter": "ch1", "article_no": "art1", "content": short_content},
            {"file_name": "file1", "chapter": "ch1", "article_no": "art2", "content": long_content},
        ]
        out_loc, out_content, out_metas = chunk_documents(
            docs_loc, docs_content, metadatas, chunk_size=200)
        # 1 透传 + N 个子 chunk
        passed = sum(1 for m in out_metas if "chunk_seq" not in m)
        split = sum(1 for m in out_metas if "chunk_seq" in m)
        self.assertEqual(passed, 1)
        self.assertGreater(split, 1)
        self.assertEqual(len(out_metas), passed + split)


if __name__ == "__main__":
    unittest.main()
