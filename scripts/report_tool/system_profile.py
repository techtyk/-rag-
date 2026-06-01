"""
系统性能与架构报告。

测量各阶段运行效率与显存占用，并可视化代码宏观运行流程。
用法：
    cd app && python -m scripts.report_tool.system_profile [--num-questions 10] [--seed 42]
"""

import argparse
import gc
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import torch

from config import (KB_PATH, QA_PATH, INDEX_DIR, BM25_K1, BM25_B, BM25_BACKEND, BM25_RECALL_K,
                    DENSE_MODEL_PATH, DENSE_DEVICE, DENSE_BATCH_SIZE,
                    DENSE_RECALL_K, RRF_METHOD, RRF_K, RRF_2WAY_AXIS,
                    RERANKER_MODEL, RERANKER_MODEL_PATH, RERANKER_DEVICE, RERANK_TOP_K)
from utils.doc_parser import parse_regulation
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.retrieve import Retriever
from rerank.reranker import RERANKER_REGISTRY

REPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "doc" / "reports"


def _gpu_mem_mb() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 1),
    }


def _load_questions(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for rows in data.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            q = row.get("问题") or row.get("question", "")
            if q:
                questions.append({
                    "question": q,
                    "category_1": row.get("一级类目", ""),
                    "category_2": row.get("二级类目", ""),
                })
    return questions


def measure_stage(build_index: bool, num_questions: int, seed: int) -> dict:
    """逐阶段测量耗时和显存，返回完整的 profile 数据。"""
    profile = {
        "config": {
            "bm25_k1": BM25_K1, "bm25_b": BM25_B, "bm25_backend": BM25_BACKEND,
            "bm25_recall_k": BM25_RECALL_K,
            "dense_model": Path(DENSE_MODEL_PATH).name if DENSE_MODEL_PATH else "",
            "dense_recall_k": DENSE_RECALL_K, "dense_batch_size": DENSE_BATCH_SIZE,
            "rrf_method": RRF_METHOD, "rrf_k": RRF_K, "rrf_2way_axis": RRF_2WAY_AXIS,
            "reranker_model": RERANKER_MODEL,
            "reranker_model_name": Path(RERANKER_MODEL_PATH).name if RERANKER_MODEL_PATH else "",
            "rerank_top_k": RERANK_TOP_K,
        },
        "stages": {},
        "queries": [],
    }

    # ---- 阶段 1：文档解析 ----
    t0 = time.time()
    docs_loc, docs_content, metadatas = parse_regulation(str(KB_PATH))
    profile["stages"]["doc_parse"] = {
        "time_s": round(time.time() - t0, 2),
        "num_docs": len(metadatas),
        "gpu_mem": _gpu_mem_mb(),
    }

    # ---- 阶段 2：索引构建 or 加载 ----
    if build_index:
        t0 = time.time()
        bm25 = BM25Retriever(docs_loc, docs_content, metadatas,
                             k1=BM25_K1, b=BM25_B, backend=BM25_BACKEND)
        bm25_time = time.time() - t0
        profile["stages"]["bm25_build"] = {
            "time_s": round(bm25_time, 2),
            "gpu_mem": _gpu_mem_mb(),
        }

        t0 = time.time()
        mem_before = _gpu_mem_mb()
        dense = DenseRetriever(docs_loc, docs_content, metadatas,
                               model_path=DENSE_MODEL_PATH, device=DENSE_DEVICE,
                               batch_size=DENSE_BATCH_SIZE)
        dense_time = time.time() - t0
        mem_after = _gpu_mem_mb()
        profile["stages"]["dense_build"] = {
            "time_s": round(dense_time, 2),
            "gpu_allocated_delta_mb": round(
                mem_after["allocated_mb"] - mem_before["allocated_mb"], 1),
            "gpu_mem_after": mem_after,
        }
        del bm25, dense
        gc.collect()
        torch.cuda.empty_cache()

    # ---- 阶段 3：从持久化索引加载（模拟正常启动） ----
    t0 = time.time()
    config = {
        "index_dir": str(INDEX_DIR),
        "bm25_k1": BM25_K1, "bm25_b": BM25_B, "bm25_backend": BM25_BACKEND,
        "bm25_recall_k": BM25_RECALL_K,
        "dense_model_path": DENSE_MODEL_PATH,
        "dense_device": DENSE_DEVICE,
        "dense_batch_size": DENSE_BATCH_SIZE,
        "dense_recall_k": DENSE_RECALL_K,
        "rrf_method": RRF_METHOD, "rrf_k": RRF_K, "rrf_2way_axis": RRF_2WAY_AXIS,
    }
    retriever = Retriever.load(config["index_dir"], config)
    load_time = time.time() - t0
    profile["stages"]["index_load"] = {
        "time_s": round(load_time, 2),
        "gpu_mem": _gpu_mem_mb(),
    }

    # ---- 阶段 4：单条查询的各子阶段耗时 ----
    all_questions = _load_questions(str(QA_PATH))
    random.seed(seed)
    sampled = random.sample(all_questions, min(num_questions, len(all_questions)))

    # Reranker 加载一次，循环复用
    reranker_cls = RERANKER_REGISTRY[RERANKER_MODEL]
    reranker = reranker_cls(RERANKER_MODEL_PATH, device=RERANKER_DEVICE)

    per_query_times = {"bm25": [], "dense": [], "rrf": [], "rerank": [], "total": []}
    for item in sampled:
        query = item["question"]

        # BM25
        t0 = time.time()
        bm25_results = retriever.bm25.search(query, top_k=BM25_RECALL_K)
        bm25_t = time.time() - t0

        # Dense
        t0 = time.time()
        dense_results = retriever.dense.search(query, top_k=DENSE_RECALL_K)
        dense_t = time.time() - t0

        # RRF（手动计时 fuse）
        t0 = time.time()
        if RRF_METHOD == "2way":
            fused = retriever._fuse_2way(bm25_results, dense_results, RRF_2WAY_AXIS)
        else:
            fused = retriever._fuse_4way(bm25_results, dense_results)
        rrf_t = time.time() - t0

        query_profile = {
            "question": query[:50],
            "bm25_ms": round(bm25_t * 1000, 1),
            "dense_ms": round(dense_t * 1000, 1),
            "rrf_ms": round(rrf_t * 1000, 1),
            "candidates": len(fused),
        }

        # Reranker
        documents = [r["content"] for r in fused]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        rerank_scores = reranker.rerank(query, documents, batch_size=2)
        rerank_t = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

        query_profile["rerank_ms"] = round(rerank_t * 1000, 1)
        query_profile["rerank_peak_gpu_mb"] = round(peak_mem, 1)
        total_ms = query_profile["bm25_ms"] + query_profile["dense_ms"] + query_profile["rrf_ms"] + query_profile["rerank_ms"]
        query_profile["total_ms"] = round(total_ms, 1)

        profile["queries"].append(query_profile)
        per_query_times["bm25"].append(query_profile["bm25_ms"])
        per_query_times["dense"].append(query_profile["dense_ms"])
        per_query_times["rrf"].append(query_profile["rrf_ms"])
        per_query_times["rerank"].append(query_profile["rerank_ms"])
        per_query_times["total"].append(query_profile["total_ms"])

    # 释放 Reranker
    del reranker
    gc.collect()
    torch.cuda.empty_cache()

    # 汇总统计
    def _stats(lst):
        if not lst:
            return {"avg": 0, "min": 0, "max": 0}
        return {"avg": round(sum(lst) / len(lst), 1),
                "min": round(min(lst), 1), "max": round(max(lst), 1)}

    profile["summary"] = {k: _stats(v) for k, v in per_query_times.items()}
    profile["summary"]["num_queries"] = len(sampled)

    # 总显存快照
    profile["final_gpu_mem"] = _gpu_mem_mb()
    if torch.cuda.is_available():
        profile["gpu_info"] = {
            "device": torch.cuda.get_device_name(0),
            "total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024),
        }

    del retriever
    gc.collect()
    torch.cuda.empty_cache()

    return profile


def generate_report(profile: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = profile["config"]
    stages = profile["stages"]
    queries = profile["queries"]
    summary = profile["summary"]

    # 预提取 HTML 中需要的值，避免 f-string 中 {{}} 与 .get() 冲突
    idx_load_time = stages.get("index_load", {}).get("time_s", "?")
    bm25_avg = summary.get("bm25", {}).get("avg", "?")
    dense_avg = summary.get("dense", {}).get("avg", "?")
    rrf_avg = summary.get("rrf", {}).get("avg", "?")
    rerank_avg = summary.get("rerank", {}).get("avg", "?")
    bm25_minmax = f'{summary.get("bm25", {}).get("min", 0)} / {summary.get("bm25", {}).get("max", 0)}'
    dense_minmax = f'{summary.get("dense", {}).get("min", 0)} / {summary.get("dense", {}).get("max", 0)}'
    rrf_minmax = f'{summary.get("rrf", {}).get("min", 0)} / {summary.get("rrf", {}).get("max", 0)}'
    rerank_minmax = f'{summary.get("rerank", {}).get("min", 0)} / {summary.get("rerank", {}).get("max", 0)}'
    total_avg = summary.get("total", {}).get("avg", 0)
    total_minmax = f'{summary.get("total", {}).get("min", 0)} / {summary.get("total", {}).get("max", 0)}'
    num_queries = summary.get("num_queries", 0)
    doc_parse = stages.get("doc_parse", {})
    num_docs = doc_parse.get("num_docs", "?")

    # 查询详情表格行
    query_rows = ""
    for i, q in enumerate(queries):
        query_rows += f"""<tr>
            <td>{i+1}</td>
            <td class="q-cell" title="{q['question']}">{q['question'][:40]}...</td>
            <td>{q['bm25_ms']}</td>
            <td>{q['dense_ms']}</td>
            <td>{q['rrf_ms']}</td>
            <td>{q['rerank_ms']}</td>
            <td><strong>{q['total_ms']}</strong></td>
            <td>{q['candidates']}</td>
            <td>{q['rerank_peak_gpu_mb']}</td>
        </tr>"""

    # 汇总行
    def _sum_cell(key):
        s = summary.get(key, {})
        return f'{s.get("avg", 0)} / {s.get("min", 0)} / {s.get("max", 0)}'

    # 阶段耗时卡片
    stage_cards = ""
    stage_labels = {
        "doc_parse": "文档解析",
        "bm25_build": "BM25 索引构建",
        "dense_build": "Dense 索引构建",
        "index_load": "索引加载（启动）",
    }
    for key, label in stage_labels.items():
        if key not in stages:
            continue
        s = stages[key]
        gpu_info = ""
        if "gpu_mem" in s:
            gm = s["gpu_mem"]
            gpu_info = f'GPU: {gm["allocated_mb"]}MB alloc / {gm["reserved_mb"]}MB reserved'
        elif "gpu_mem_after" in s:
            gm = s["gpu_mem_after"]
            delta = s.get("gpu_allocated_delta_mb", 0)
            gpu_info = f'GPU: {gm["allocated_mb"]}MB alloc (构建增量 {delta}MB)'
        extra = ""
        if "num_docs" in s:
            extra = f'<span class="stage-extra">文档数: {s["num_docs"]}</span>'
        stage_cards += f"""<div class="stage-card">
            <div class="stage-label">{label}</div>
            <div class="stage-time">{s["time_s"]}s</div>
            {extra}
            <div class="stage-gpu">{gpu_info}</div>
        </div>"""

    # GPU 信息
    gpu_html = ""
    if "gpu_info" in profile:
        gi = profile["gpu_info"]
        gpu_html = f"""<div class="info-card">
            <h3>GPU 信息</h3>
            <table class="info-table"><tbody>
                <tr><td>设备</td><td>{gi["device"]}</td></tr>
                <tr><td>总显存</td><td>{gi["total_vram_mb"]} MB</td></tr>
            </tbody></table>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>系统性能与架构报告 - {ts}</title>
<style>
:root {{ --bg:#0f1117; --card:#1a1d28; --border:#2a2d3a; --text:#e0e0e0;
  --muted:#8b8fa3; --accent:#4fc3f7; --accent2:#66bb6a; --warn:#ffa726; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--text);
  line-height:1.6; padding:20px; max-width:1200px; margin:0 auto; }}
h1 {{ text-align:center; padding:24px 0 8px; font-size:1.6em; color:var(--accent); }}
.subtitle {{ text-align:center; color:var(--muted); font-size:0.85em; margin-bottom:32px; }}
h2 {{ color:var(--accent); border-bottom:1px solid var(--border); padding:16px 0 8px; margin:24px 0 16px;
  font-size:1.2em; }}

/* 架构流程图 */
.arch-flow {{ display:flex; align-items:center; justify-content:center; gap:0;
  padding:24px 16px; overflow-x:auto; flex-wrap:wrap; }}
.arch-step {{ background:var(--card); border:1px solid var(--border); border-radius:8px;
  padding:12px 16px; text-align:center; min-width:130px; position:relative; }}
.arch-step .step-title {{ font-weight:600; font-size:0.9em; }}
.arch-step .step-detail {{ font-size:0.75em; color:var(--muted); margin-top:4px; }}
.arch-arrow {{ color:var(--muted); font-size:1.5em; padding:0 4px; flex-shrink:0; }}
.arch-step.highlight {{ border-color:var(--accent); background:#1a2535; }}

/* 阶段卡片 */
.stages-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
  gap:12px; margin:16px 0; }}
.stage-card {{ background:var(--card); border:1px solid var(--border); border-radius:8px;
  padding:14px; }}
.stage-label {{ font-weight:600; color:var(--accent); font-size:0.9em; }}
.stage-time {{ font-size:1.4em; font-weight:700; margin:6px 0 4px; }}
.stage-extra {{ font-size:0.8em; color:var(--accent2); }}
.stage-gpu {{ font-size:0.75em; color:var(--muted); margin-top:4px; }}

/* 信息表格 */
.info-card {{ background:var(--card); border:1px solid var(--border); border-radius:8px;
  padding:16px; margin:12px 0; }}
.info-card h3 {{ color:var(--accent); margin-bottom:10px; font-size:0.95em; }}
.info-table {{ width:100%; border-collapse:collapse; }}
.info-table td {{ padding:4px 12px; font-size:0.85em; border-bottom:1px solid var(--border); }}
.info-table td:first-child {{ color:var(--muted); width:140px; }}

/* 查询详情表 */
.query-table {{ width:100%; border-collapse:collapse; font-size:0.82em; margin:12px 0; }}
.query-table th {{ background:var(--card); padding:8px 6px; text-align:center;
  border-bottom:2px solid var(--accent); color:var(--accent); white-space:nowrap; }}
.query-table td {{ padding:6px; text-align:center; border-bottom:1px solid var(--border); }}
.query-table .q-cell {{ text-align:left; max-width:200px; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; }}

/* 汇总条 */
.summary-bar {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:12px; margin:16px 0; }}
.sum-item {{ background:var(--card); border:1px solid var(--border); border-radius:8px;
  padding:12px; text-align:center; }}
.sum-label {{ font-size:0.75em; color:var(--muted); }}
.sum-value {{ font-size:1.3em; font-weight:700; margin-top:4px; }}
.sum-detail {{ font-size:0.7em; color:var(--muted); }}

/* 代码模块 */
.module-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
  gap:12px; margin:12px 0; }}
.module-card {{ background:var(--card); border:1px solid var(--border); border-radius:8px;
  padding:14px; }}
.module-card .m-name {{ font-family:'Fira Code',monospace; color:var(--accent); font-size:0.9em;
  font-weight:600; }}
.module-card .m-file {{ font-family:'Fira Code',monospace; color:var(--muted); font-size:0.75em; }}
.module-card .m-desc {{ font-size:0.8em; color:var(--text); margin-top:6px; }}
.module-card .m-deps {{ font-size:0.7em; color:var(--warn); margin-top:4px; }}
</style>
</head>
<body>
<h1>系统性能与架构报告</h1>
<div class="subtitle">生成时间：{ts} | 查询数：{num_queries} | RRF: {cfg["rrf_method"]} | Reranker: {cfg["reranker_model"]}</div>

<!-- 架构流程 -->
<h2>1. Pipeline 运行流程</h2>
<div class="arch-flow">
  <div class="arch-step">
    <div class="step-title">文档解析</div>
    <div class="step-detail">doc_parser.py<br/>JSON → 条款列表</div>
  </div>
  <div class="arch-arrow">→</div>
  <div class="arch-step">
    <div class="step-title">索引构建</div>
    <div class="step-detail">BM25 + Dense FAISS<br/>定位 + 内容 双路</div>
  </div>
  <div class="arch-arrow">→</div>
  <div class="arch-step highlight">
    <div class="step-title">索引加载</div>
    <div class="step-detail">从磁盘加载持久化索引<br/>启动 ~{idx_load_time}s</div>
  </div>
  <div class="arch-arrow">→</div>
  <div class="arch-step highlight">
    <div class="step-title">BM25 召回</div>
    <div class="step-detail">定位 + 内容 各 top-{cfg["bm25_recall_k"]}<br/>~{bm25_avg}ms</div>
  </div>
  <div class="arch-arrow">+</div>
  <div class="arch-step highlight">
    <div class="step-title">Dense 召回</div>
    <div class="step-detail">定位 + 内容 各 top-{cfg["dense_recall_k"]}<br/>~{dense_avg}ms</div>
  </div>
  <div class="arch-arrow">→</div>
  <div class="arch-step">
    <div class="step-title">RRF 融合</div>
    <div class="step-detail">{cfg["rrf_method"]} 模式, k={cfg["rrf_k"]}<br/>~{rrf_avg}ms</div>
  </div>
  <div class="arch-arrow">→</div>
  <div class="arch-step">
    <div class="step-title">Reranker 精排</div>
    <div class="step-detail">{cfg["reranker_model"]}<br/>~{rerank_avg}ms</div>
  </div>
  <div class="arch-arrow">→</div>
  <div class="arch-step">
    <div class="step-title">输出 Top-{cfg["rerank_top_k"]}</div>
    <div class="step-detail">最终结果</div>
  </div>
</div>

<!-- 代码模块说明 -->
<h2>2. 代码模块职责</h2>
<div class="module-grid">
  <div class="module-card">
    <div class="m-name">Retriever（统一入口）</div>
    <div class="m-file">retrieval/retrieve.py</div>
    <div class="m-desc">管理 BM25 + Dense 的多路召回与 RRF 融合。支持 4way（四路独立 RRF）和 2way（分组 RRF）两种模式。</div>
    <div class="m-deps">→ BM25Retriever, DenseRetriever</div>
  </div>
  <div class="module-card">
    <div class="m-name">BM25Retriever</div>
    <div class="m-file">retrieval/bm25.py</div>
    <div class="m-desc">基于 bm25s 的稀疏检索。对定位文本（文件名+章节+条款号）和内容文本分别建索引，双路召回后合并。</div>
    <div class="m-deps">→ tokenizer (分词)</div>
  </div>
  <div class="module-card">
    <div class="m-name">DenseRetriever</div>
    <div class="m-file">retrieval/dense.py</div>
    <div class="m-desc">基于 SentenceTransformer + FAISS 的稠密检索。编码定位和内容文本为向量，按内积相似度召回。支持懒加载：索引从磁盘加载后模型仅按需载入 CPU。</div>
    <div class="m-deps">→ SentenceTransformer, FAISS</div>
  </div>
  <div class="module-card">
    <div class="m-name">Reranker</div>
    <div class="m-file">rerank/reranker.py</div>
    <div class="m-desc">精排模块，独立于检索阶段。支持 BGE（Cross-Encoder）、GTE（ONNX）、Qwen3（LLM yes/no logit）三种模型。通过注册表模式实例化。</div>
    <div class="m-deps">独立模块，不依赖 Retriever</div>
  </div>
  <div class="module-card">
    <div class="m-name">Pipeline 编排</div>
    <div class="m-file">main.py</div>
    <div class="m-desc">两阶段编排：先加载索引（或构建新索引），再进入交互循环。每次查询：Retriever.retrieve() → Reranker.rerank() → 按 rerank_score 排序 → 展示。</div>
    <div class="m-deps">→ Retriever, Reranker</div>
  </div>
  <div class="module-card">
    <div class="m-name">文档解析器</div>
    <div class="m-file">utils/doc_parser.py</div>
    <div class="m-desc">递归解析法规 JSON 树结构，提取每个条款的定位文本、内容文本和元数据（文件名、章节、条款号）。</div>
    <div class="m-deps">无外部依赖</div>
  </div>
</div>

<!-- 启动阶段耗时 -->
<h2>3. 启动阶段耗时</h2>
<div class="stages-grid">{stage_cards}</div>

{gpu_html}

<!-- 查询性能 -->
<h2>4. 查询阶段性能（ms）</h2>
<div class="summary-bar">
  <div class="sum-item">
    <div class="sum-label">BM25 召回</div>
    <div class="sum-value">{bm25_avg}ms</div>
    <div class="sum-detail">min/max: {bm25_minmax}</div>
  </div>
  <div class="sum-item">
    <div class="sum-label">Dense 召回</div>
    <div class="sum-value">{dense_avg}ms</div>
    <div class="sum-detail">min/max: {dense_minmax}</div>
  </div>
  <div class="sum-item">
    <div class="sum-label">RRF 融合</div>
    <div class="sum-value">{rrf_avg}ms</div>
    <div class="sum-detail">min/max: {rrf_minmax}</div>
  </div>
  <div class="sum-item">
    <div class="sum-label">Reranker 精排</div>
    <div class="sum-value">{rerank_avg}ms</div>
    <div class="sum-detail">min/max: {rerank_minmax}</div>
  </div>
  <div class="sum-item">
    <div class="sum-label">端到端总耗时</div>
    <div class="sum-value" style="color:var(--accent)">{total_avg}ms</div>
    <div class="sum-detail">min/max: {total_minmax}</div>
  </div>
</div>

<table class="query-table">
  <thead><tr>
    <th>#</th><th>查询</th><th>BM25 (ms)</th><th>Dense (ms)</th>
    <th>RRF (ms)</th><th>Rerank (ms)</th><th>总计 (ms)</th>
    <th>候选数</th><th>GPU峰值 (MB)</th>
  </tr></thead>
  <tbody>{query_rows}</tbody>
</table>

<!-- 配置详情 -->
<h2>5. 当前配置</h2>
<div class="info-card">
  <table class="info-table"><tbody>
    <tr><td>知识库文档数</td><td>{num_docs}</td></tr>
    <tr><td>BM25 参数</td><td>k1={cfg["bm25_k1"]}, b={cfg["bm25_b"]}, backend={cfg["bm25_backend"]}</td></tr>
    <tr><td>BM25 召回数</td><td>{cfg["bm25_recall_k"]} / 路（定位 + 内容）</td></tr>
    <tr><td>Dense 模型</td><td>{cfg["dense_model"]}</td></tr>
    <tr><td>Dense 召回数</td><td>{cfg["dense_recall_k"]} / 路（定位 + 内容）</td></tr>
    <tr><td>Dense batch_size</td><td>{cfg["dense_batch_size"]}</td></tr>
    <tr><td>RRF 模式</td><td>{cfg["rrf_method"]}, k={cfg["rrf_k"]}</td></tr>
    <tr><td>Reranker</td><td>{cfg["reranker_model_name"]}</td></tr>
    <tr><td>Rerank Top-K</td><td>{cfg["rerank_top_k"]}</td></tr>
    <tr><td>索引目录</td><td>{INDEX_DIR}</td></tr>
  </tbody></table>
</div>

</body></html>"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"system_profile_{ts}.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n报告已生成：{out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="系统性能与架构报告")
    parser.add_argument("--num-questions", type=int, default=10, help="抽样问题数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--skip-build", action="store_true",
                        help="跳过索引构建测量（仅测量加载和查询）")
    args = parser.parse_args()

    profile = measure_stage(
        build_index=not args.skip_build,
        num_questions=args.num_questions,
        seed=args.seed,
    )
    generate_report(profile)


if __name__ == "__main__":
    main()
