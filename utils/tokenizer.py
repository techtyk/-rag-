import jieba
from typing import List

STOP_WORDS = set(list(
    "，。！？、；：""''「」【】（）《》—…·～\n\r\t "
    "的了在是我有和就不人都一上也很到说要去你会着没有看"
    "好自己这他她它们那被把从让比等更啊吧呢吗嘛呀么嗯"
    "其之与而或及以可该还能"
))


def _filter_stop_words(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t.strip() and t not in STOP_WORDS]


def tokenize_for_doc(text: str) -> List[str]:
    return _filter_stop_words(list(jieba.cut_for_search(text)))


def tokenize_for_query(text: str) -> List[str]:
    return _filter_stop_words(list(jieba.cut(text)))
