import unittest
from utils.tokenizer import tokenize_for_doc, tokenize_for_query, _filter_stop_words, STOP_WORDS


class TestFilterStopWords(unittest.TestCase):
    def test_removes_stop_words(self):
        tokens = ["整车", "的", "质量", "是", "呢", "限制"]
        result = _filter_stop_words(tokens)
        self.assertEqual(result, ["整车", "质量", "限制"])

    def test_removes_punctuation(self):
        tokens = ["整车", "，", "质量", "。"]
        result = _filter_stop_words(tokens)
        self.assertEqual(result, ["整车", "质量"])

    def test_removes_whitespace_only(self):
        tokens = ["整车", " ", "质量", "\n"]
        result = _filter_stop_words(tokens)
        self.assertEqual(result, ["整车", "质量"])

    def test_empty_input(self):
        self.assertEqual(_filter_stop_words([]), [])

    def test_all_stop_words(self):
        tokens = ["的", "了", "是", "呢"]
        self.assertEqual(_filter_stop_words(tokens), [])


class TestTokenizeForDoc(unittest.TestCase):
    def test_produces_tokens(self):
        result = tokenize_for_doc("整车质量限制")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertTrue(all(isinstance(t, str) for t in result))

    def test_stop_words_filtered(self):
        result = tokenize_for_doc("整车质量限制是多少呢")
        self.assertNotIn("是", result)
        self.assertNotIn("呢", result)

    def test_search_mode_subtokens(self):
        result = tokenize_for_doc("清华大学")
        self.assertIn("清华大学", result)
        self.assertIn("清华", result)
        self.assertIn("大学", result)

    def test_empty_string(self):
        result = tokenize_for_doc("")
        self.assertEqual(result, [])


class TestTokenizeForQuery(unittest.TestCase):
    def test_produces_tokens(self):
        result = tokenize_for_query("整车质量限制")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_stop_words_filtered(self):
        result = tokenize_for_query("整车质量限制是多少呢")
        self.assertNotIn("是", result)
        self.assertNotIn("呢", result)

    def test_no_subtokens_in_normal_mode(self):
        result = tokenize_for_query("清华大学")
        self.assertIn("清华大学", result)
        self.assertNotIn("清华", result)
        self.assertNotIn("大学", result)

    def test_empty_string(self):
        result = tokenize_for_query("")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
