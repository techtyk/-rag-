"""Reranker 模块：支持 BGE、GTE(ONNX)、Qwen3 三种模型。"""

import time
from typing import List, Dict

import onnxruntime as ort
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


class BaseReranker:
    """Reranker 基类。"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._load()

    def _load(self):
        raise NotImplementedError

    def rerank(self, query: str, documents: List[str], batch_size: int = 16) -> List[float]:
        """对 (query, doc) 对打分，返回与 documents 等长的分数列表。"""
        raise NotImplementedError


class BGEReranker(BaseReranker):
    """BGE-reranker-v2-m3: XLMRobertaForSequenceClassification (safetensors)。"""

    def _load(self):
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device).eval()
        print(f"Reranker BGE 加载完成，耗时 {time.time()-t0:.1f}s")

    @torch.no_grad()
    def rerank(self, query: str, documents: List[str], batch_size: int = 16) -> List[float]:
        scores = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            pairs = [[query, doc] for doc in batch]
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True,
                return_tensors="pt", max_length=512,
            ).to(self.device)
            logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
            scores.extend(F.sigmoid(logits).tolist())
        return scores


class GTEReranker(BaseReranker):
    """gte-multilingual-reranker-base: ONNX 格式，使用 onnxruntime 推理。"""

    def _load(self):
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path.rstrip("/") + "/model.onnx"),
            sess_options=sess_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_names = {inp.name for inp in self.session.get_inputs()}
        print(f"Reranker GTE(ONNX) 加载完成，耗时 {time.time()-t0:.1f}s")

    def rerank(self, query: str, documents: List[str], batch_size: int = 16) -> List[float]:
        import numpy as np

        scores = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            inputs = self.tokenizer(
                [query] * len(batch), batch,
                padding=True, truncation=True,
                return_tensors="np", max_length=512,
            )
            ort_inputs = {k: v for k, v in inputs.items() if k in self._input_names}
            if "token_type_ids" in self._input_names and "token_type_ids" not in ort_inputs:
                ort_inputs["token_type_ids"] = np.zeros_like(ort_inputs["input_ids"])
            logits = self.session.run(None, ort_inputs)[0].flatten()
            probs = 1.0 / (1.0 + np.exp(-logits))
            scores.extend(probs.tolist())
        return scores


class Qwen3Reranker(BaseReranker):
    """Qwen3-Reranker-0.6B: LLM 架构，通过 yes/no logit 差值计算相关性。"""

    def _load(self):
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.float16,
        ).to(self.device).eval()
        self.token_yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no = self.tokenizer.convert_tokens_to_ids("no")

        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query "
            'and the Instruct provided. Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
        self._prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        self._max_length = 2048
        print(f"Reranker Qwen3 加载完成，耗时 {time.time()-t0:.1f}s")

    @staticmethod
    def _format_content(query: str, doc: str, instruction: str = "") -> str:
        inst = instruction or "Given a web search query, retrieve relevant passages that answer the query"
        return f"<Instruct>: {inst}\n<Query>: {query}\n<Document>: {doc}"

    @torch.no_grad()
    def rerank(self, query: str, documents: List[str], batch_size: int = 8) -> List[float]:
        scores = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            pairs = [self._format_content(query, doc) for doc in batch]
            max_content_len = self._max_length - len(self._prefix_ids) - len(self._suffix_ids)
            inputs = self.tokenizer(
                pairs, padding=False, truncation="longest_first",
                return_attention_mask=False, max_length=max_content_len,
            )
            for j in range(len(inputs["input_ids"])):
                inputs["input_ids"][j] = self._prefix_ids + inputs["input_ids"][j] + self._suffix_ids
            inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self._max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits[:, -1, :]
            yes_logits = logits[:, self.token_yes]
            no_logits = logits[:, self.token_no]
            batch_scores = torch.stack([no_logits, yes_logits], dim=1)
            probs = F.log_softmax(batch_scores, dim=1)[:, 1].exp()
            scores.extend(probs.tolist())
        return scores


# 模型名称 → 类的映射
RERANKER_REGISTRY = {
    "bge": BGEReranker,
    "gte": GTEReranker,
    "qwen3": Qwen3Reranker,
}
