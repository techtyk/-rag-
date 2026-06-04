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

import numpy as np
import torch

from config import (KB_PATH, QA_PATH, INDEX_DIR, BM25_K1, BM25_B, BM25_BACKEND, BM25_RECALL_K,
                    DENSE_MODEL_PATH, DENSE_DEVICE, INDEX_DEVICE, INDEX_BATCH_SIZE,
                    DENSE_RECALL_K, RRF_METHOD, RRF_K, RRF_TOP_K, RRF_2WAY_AXIS,
                    RERANKER_MODEL, RERANKER_MODEL_PATH, RERANKER_DEVICE,
                    RERANKER_BATCH_SIZE, RERANK_TOP_K,
                    QA_DATA_PATH, QA_INDEX_DIR, QA_BM25_K1, QA_BM25_B, QA_BM25_BACKEND,
                    QA_BM25_RECALL_K, QA_INDEX_DEVICE, QA_INDEX_BATCH_SIZE,
                    QA_DENSE_RECALL_K, QA_RRF_K, QA_RRF_TOP_K, QA_RERANK_TOP_K,
                    QA_SIMILARITY_THRESHOLD, QA_RERANKER_BATCH_SIZE)
from utils.doc_parser import parse_regulation
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.retrieve import Retriever
from retrieval.qa_retrieve import QARetriever, _check_qa_index_complete
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
            "dense_recall_k": DENSE_RECALL_K, "dense_batch_size": INDEX_BATCH_SIZE,
            "rrf_method": RRF_METHOD, "rrf_k": RRF_K, "rrf_2way_axis": RRF_2WAY_AXIS,
            "rrf_top_k": RRF_TOP_K,
            "reranker_model": RERANKER_MODEL,
            "reranker_model_name": Path(RERANKER_MODEL_PATH).name if RERANKER_MODEL_PATH else "",
            "reranker_batch_size": RERANKER_BATCH_SIZE,
            "rerank_top_k": RERANK_TOP_K,
            "qa_bm25_k1": QA_BM25_K1, "qa_bm25_b": QA_BM25_B, "qa_bm25_backend": QA_BM25_BACKEND,
            "qa_bm25_recall_k": QA_BM25_RECALL_K, "qa_dense_recall_k": QA_DENSE_RECALL_K,
            "qa_rrf_k": QA_RRF_K, "qa_rrf_top_k": QA_RRF_TOP_K,
            "qa_rerank_top_k": QA_RERANK_TOP_K, "qa_similarity_threshold": QA_SIMILARITY_THRESHOLD,
            "qa_reranker_batch_size": QA_RERANKER_BATCH_SIZE,
            "qa_data_path": str(QA_DATA_PATH), "qa_index_dir": str(QA_INDEX_DIR),
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
        bm25 = BM25Retriever.build(docs_loc, docs_content, metadatas,
                             k1=BM25_K1, b=BM25_B, backend=BM25_BACKEND)
        bm25_time = time.time() - t0
        profile["stages"]["bm25_build"] = {
            "time_s": round(bm25_time, 2),
            "gpu_mem": _gpu_mem_mb(),
        }

        t0 = time.time()
        mem_before = _gpu_mem_mb()
        dense = DenseRetriever.build(docs_loc, docs_content, metadatas,
                               model_path=DENSE_MODEL_PATH, device=INDEX_DEVICE,
                               batch_size=INDEX_BATCH_SIZE)
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
        "dense_device": INDEX_DEVICE,
        "dense_batch_size": INDEX_BATCH_SIZE,
        "dense_recall_k": DENSE_RECALL_K,
        "rrf_method": RRF_METHOD, "rrf_k": RRF_K, "rrf_2way_axis": RRF_2WAY_AXIS,
        "rrf_top_k": RRF_TOP_K,
    }
    retriever = Retriever.load(config["index_dir"], config)
    load_time = time.time() - t0
    profile["stages"]["index_load"] = {
        "time_s": round(load_time, 2),
        "gpu_mem": _gpu_mem_mb(),
    }

    # 加载 QA 索引，共享法规 Dense 模型
    qa_config = {
        "index_dir": str(QA_INDEX_DIR),
        "bm25_k1": QA_BM25_K1, "bm25_b": QA_BM25_B, "bm25_backend": QA_BM25_BACKEND,
        "bm25_recall_k": QA_BM25_RECALL_K,
        "dense_model_path": DENSE_MODEL_PATH,
        "dense_device": QA_INDEX_DEVICE,
        "dense_batch_size": QA_INDEX_BATCH_SIZE,
        "dense_recall_k": QA_DENSE_RECALL_K,
        "rrf_k": QA_RRF_K,
        "rrf_top_k": QA_RRF_TOP_K,
    }
    # 触发 Dense 模型加载（懒加载），确保 QA 可共享
    if retriever.dense:
        retriever.dense._ensure_model()
    shared_model = retriever.dense.model if retriever.dense else None
    if _check_qa_index_complete(qa_config["index_dir"]):
        qa_retriever = QARetriever.load(qa_config["index_dir"], qa_config,
                                        shared_model=shared_model)
    else:
        print("QA 索引未找到，QA 检索将跳过")
        qa_retriever = None

    # ---- 阶段 4：单条查询的各子阶段耗时 ----
    all_questions = _load_questions(str(QA_PATH))
    random.seed(seed)
    sampled = random.sample(all_questions, min(num_questions, len(all_questions)))

    # Reranker 加载一次，循环复用
    reranker_cls = RERANKER_REGISTRY[RERANKER_MODEL]
    reranker = reranker_cls(RERANKER_MODEL_PATH, device=RERANKER_DEVICE)

    per_query_times = {"bm25": [], "dense": [], "rrf": [], "qa_retrieve": [], "rerank": [], "total": []}
    for item in sampled:
        query = item["question"]

        # 法规 BM25
        t0 = time.time()
        bm25_results = retriever.bm25.search(query, top_k=BM25_RECALL_K)
        bm25_t = time.time() - t0

        # 法规 Dense
        t0 = time.time()
        dense_results = retriever.dense.search(query, top_k=DENSE_RECALL_K)
        dense_t = time.time() - t0

        # 法规 RRF
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

        # QA 检索（共享 query_emb）
        qa_candidates = []
        qa_t = 0
        if qa_retriever is not None:
            t0 = time.time()
            query_emb = np.array(
                retriever.dense.model.encode([query], normalize_embeddings=True,
                                             device=DENSE_DEVICE),
                dtype="float32")
            qa_candidates = qa_retriever.retrieve(query, query_emb=query_emb)
            qa_t = time.time() - t0

        query_profile["qa_retrieve_ms"] = round(qa_t * 1000, 1)
        query_profile["qa_candidates"] = len(qa_candidates)

        # Rerank（法规和 QA 分别独立精排）
        # QA 问题文本极短，与法规内容长度差异过大，合并 batch 会导致无效 padding
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        # 法规 Rerank
        reg_docs = [r["content"] for r in fused]
        reranker.rerank(query, reg_docs, batch_size=RERANKER_BATCH_SIZE) if reg_docs else None
        # QA Rerank
        qa_docs = [q["question"] for q in qa_candidates]
        reranker.rerank(query, qa_docs, batch_size=QA_RERANKER_BATCH_SIZE) if qa_docs else None
        rerank_t = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

        query_profile["rerank_ms"] = round(rerank_t * 1000, 1)
        query_profile["rerank_peak_gpu_mb"] = round(peak_mem, 1)
        query_profile["total_ms"] = round(
            query_profile["bm25_ms"] + query_profile["dense_ms"] + query_profile["rrf_ms"]
            + query_profile["qa_retrieve_ms"] + query_profile["rerank_ms"], 1)

        profile["queries"].append(query_profile)
        per_query_times["bm25"].append(query_profile["bm25_ms"])
        per_query_times["dense"].append(query_profile["dense_ms"])
        per_query_times["rrf"].append(query_profile["rrf_ms"])
        per_query_times["qa_retrieve"].append(query_profile["qa_retrieve_ms"])
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

    return profile


def generate_report(profile: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = profile["config"]
    stages = profile["stages"]
    queries = profile["queries"]
    summary = profile["summary"]

    # 预提取 HTML 中需要的值
    idx_load_time = stages.get("index_load", {}).get("time_s", "?")
    bm25_avg = summary.get("bm25", {}).get("avg", "?")
    dense_avg = summary.get("dense", {}).get("avg", "?")
    rrf_avg = summary.get("rrf", {}).get("avg", "?")
    qa_avg = summary.get("qa_retrieve", {}).get("avg", "?")
    rerank_avg = summary.get("rerank", {}).get("avg", "?")
    bm25_minmax = f'{summary.get("bm25", {}).get("min", 0)} / {summary.get("bm25", {}).get("max", 0)}'
    dense_minmax = f'{summary.get("dense", {}).get("min", 0)} / {summary.get("dense", {}).get("max", 0)}'
    rrf_minmax = f'{summary.get("rrf", {}).get("min", 0)} / {summary.get("rrf", {}).get("max", 0)}'
    qa_minmax = f'{summary.get("qa_retrieve", {}).get("min", 0)} / {summary.get("qa_retrieve", {}).get("max", 0)}'
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
            <td class="qa-col">{q['qa_retrieve_ms']}</td>
            <td>{q['rerank_ms']}</td>
            <td><strong>{q['total_ms']}</strong></td>
            <td>{q['candidates']}</td>
            <td class="qa-col">{q['qa_candidates']}</td>
            <td>{q['rerank_peak_gpu_mb']}</td>
        </tr>"""

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
  --muted:#8b8fa3; --accent:#4fc3f7; --accent2:#66bb6a; --warn:#ffa726;
  --qa-color:#ab47bc; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--text);
  line-height:1.6; padding:20px; max-width:1200px; margin:0 auto; }}
h1 {{ text-align:center; padding:24px 0 8px; font-size:1.6em; color:var(--accent); }}
.subtitle {{ text-align:center; color:var(--muted); font-size:0.85em; margin-bottom:32px; }}
h2 {{ color:var(--accent); border-bottom:1px solid var(--border); padding:16px 0 8px; margin:24px 0 16px;
  font-size:1.2em; }}

/* 架构流程图 */
.arch-container {{ padding:24px 16px; overflow-x:auto; }}
.arch-row {{ display:flex; align-items:center; justify-content:center; gap:8px;
  flex-wrap:wrap; margin:8px 0; }}
.arch-step {{ background:var(--card); border:1px solid var(--border); border-radius:8px;
  padding:12px 16px; text-align:center; min-width:120px; }}
.arch-step .step-title {{ font-weight:600; font-size:0.9em; }}
.arch-step .step-detail {{ font-size:0.75em; color:var(--muted); margin-top:4px; }}
.arch-arrow {{ color:var(--muted); font-size:1.5em; padding:0 4px; flex-shrink:0; }}
.arch-step.highlight {{ border-color:var(--accent); background:#1a2535; }}
.arch-step.qa-highlight {{ border-color:var(--qa-color); background:#251a2e; }}
.branch-label {{ font-weight:700; font-size:0.85em; padding:4px 12px; border-radius:4px;
  text-align:center; min-width:80px; }}
.branch-label.reg {{ background:#1a2535; color:var(--accent); border:1px solid var(--accent); }}
.branch-label.qa {{ background:#251a2e; color:var(--qa-color); border:1px solid var(--qa-color); }}
.arch-merge {{ text-align:center; color:var(--muted); font-size:0.8em; padding:4px 0; }}

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
.info-table td:first-child {{ color:var(--muted); width:160px; }}
.info-table tr.section-header td {{ padding-top:12px; font-weight:600; color:var(--accent);
  border-bottom:1px solid var(--accent); }}

/* 查询详情表 */
.query-table {{ width:100%; border-collapse:collapse; font-size:0.82em; margin:12px 0; }}
.query-table th {{ background:var(--card); padding:8px 6px; text-align:center;
  border-bottom:2px solid var(--accent); color:var(--accent); white-space:nowrap; }}
.query-table th.qa-col {{ color:var(--qa-color); border-bottom-color:var(--qa-color); }}
.query-table td {{ padding:6px; text-align:center; border-bottom:1px solid var(--border); }}
.query-table td.qa-col {{ color:var(--qa-color); }}
.query-table .q-cell {{ text-align:left; max-width:200px; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; }}

/* 汇总条 */
.summary-bar {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:12px; margin:16px 0; }}
.sum-item {{ background:var(--card); border:1px solid var(--border); border-radius:8px;
  padding:12px; text-align:center; }}
.sum-item.qa {{ border-color:var(--qa-color); }}
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
.module-card .m-name.qa {{ color:var(--qa-color); }}
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
<div class="arch-container">
  <!-- 顶层：启动阶段 -->
  <div class="arch-row">
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
  </div>

  <div class="arch-merge">↓ 查询阶段：双路 RAG 并行 ↓</div>

  <!-- 中层：双路 RAG 并行 -->
  <div class="arch-row" style="gap:40px;">
    <!-- 法规 RAG -->
    <div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
      <div class="branch-label reg">法规 RAG</div>
      <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
        <div class="arch-step highlight">
          <div class="step-title">BM25 双路</div>
          <div class="step-detail">定位 + 内容<br/>各 top-{cfg["bm25_recall_k"]}<br/>~{bm25_avg}ms</div>
        </div>
        <div class="arch-arrow">+</div>
        <div class="arch-step highlight">
          <div class="step-title">Dense 双路</div>
          <div class="step-detail">定位 + 内容<br/>各 top-{cfg["dense_recall_k"]}<br/>~{dense_avg}ms</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-step">
          <div class="step-title">4路 RRF</div>
          <div class="step-detail">k={cfg["rrf_k"]}<br/>~{rrf_avg}ms<br/>≤{cfg["rrf_top_k"]} 条</div>
        </div>
      </div>
    </div>

    <!-- QA RAG -->
    <div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
      <div class="branch-label qa">QA RAG</div>
      <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
        <div class="arch-step qa-highlight">
          <div class="step-title">BM25</div>
          <div class="step-detail">问题索引<br/>top-{cfg["qa_bm25_recall_k"]}</div>
        </div>
        <div class="arch-arrow">+</div>
        <div class="arch-step qa-highlight">
          <div class="step-title">Dense</div>
          <div class="step-detail">问题向量<br/>top-{cfg["qa_dense_recall_k"]}<br/>共享 query_emb</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-step qa-highlight">
          <div class="step-title">2路 RRF</div>
          <div class="step-detail">k={cfg["qa_rrf_k"]}<br/>≤{cfg["qa_rrf_top_k"]} 条</div>
        </div>
      </div>
    </div>
  </div>

  <div class="arch-merge">↓ 分别精排（QA 问题极短，合并 batch 会导致无效 padding） ↓</div>

  <!-- 底层：分开 Rerank -->
  <div class="arch-row" style="gap:40px;">
    <div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
      <div class="branch-label reg">法规 Rerank</div>
      <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
        <div class="arch-step highlight">
          <div class="step-title">法规 Rerank</div>
          <div class="step-detail">{cfg["reranker_model"]}<br/>batch_size={cfg["reranker_batch_size"]}<br/>~{rerank_avg}ms</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-step">
          <div class="step-title">法规输出</div>
          <div class="step-detail">top-{cfg["rerank_top_k"]}</div>
        </div>
      </div>
    </div>
    <div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
      <div class="branch-label qa">QA Rerank</div>
      <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
        <div class="arch-step qa-highlight">
          <div class="step-title">QA Rerank</div>
          <div class="step-detail">{cfg["reranker_model"]}<br/>batch_size={cfg["qa_reranker_batch_size"]}</div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-step qa-highlight">
          <div class="step-title">QA 输出</div>
          <div class="step-detail">top-{cfg["qa_rerank_top_k"]}<br/>阈值 ≥{cfg["qa_similarity_threshold"]}</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- 代码模块说明 -->
<h2>2. 代码模块职责</h2>
<div class="module-grid">
  <div class="module-card">
    <div class="m-name">Retriever（法规统一入口）</div>
    <div class="m-file">retrieval/retrieve.py</div>
    <div class="m-desc">管理法规 BM25 + Dense 的多路召回与 4路/2路 RRF 融合。定位索引和内容索引各一路，支持 4way 和 2way 两种融合模式。</div>
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
    <div class="m-desc">基于 SentenceTransformer + FAISS 的稠密检索。编码定位和内容文本为向量，按内积相似度召回。模型可共享给 QA 检索器复用。</div>
    <div class="m-deps">→ SentenceTransformer, FAISS</div>
  </div>
  <div class="module-card">
    <div class="m-name qa">QARetriever（QA 检索器）</div>
    <div class="m-file">retrieval/qa_retrieve.py</div>
    <div class="m-desc">QA 问答对检索。对问题文本建单索引（非双路），BM25 + Dense 2 路 RRF 融合。支持共享 Dense 模型实例和预编码 query embedding，避免重复编码。</div>
    <div class="m-deps">→ BM25 (bm25s), Dense (共享模型), tokenizer</div>
  </div>
  <div class="module-card">
    <div class="m-name">Reranker</div>
    <div class="m-file">rerank/reranker.py</div>
    <div class="m-desc">精排模块。支持 BGE、GTE、Qwen3 三种模型。法规和 QA 分别独立精排（QA 问题极短，合并 batch 会导致无效 padding）。</div>
    <div class="m-deps">独立模块，不依赖 Retriever</div>
  </div>
  <div class="module-card">
    <div class="m-name">Pipeline 编排</div>
    <div class="m-file">main.py / server.py</div>
    <div class="m-desc">双路 RAG 编排：法规 Retriever.retrieve() + QA QARetriever.retrieve() → 法规和 QA 分别独立 Rerank → 法规组排序截断 + QA 组排序阈值过滤。</div>
    <div class="m-deps">→ Retriever, QARetriever, Reranker</div>
  </div>
  <div class="module-card">
    <div class="m-name">文档解析器</div>
    <div class="m-file">utils/doc_parser.py</div>
    <div class="m-desc">递归解析法规 JSON 树结构，提取每个条款的定位文本、内容文本和元数据。</div>
    <div class="m-deps">无外部依赖</div>
  </div>
  <div class="module-card">
    <div class="m-name qa">QA 数据加载器</div>
    <div class="m-file">utils/qa_data_loader.py</div>
    <div class="m-desc">从 Excel 文件加载 QA 问答对。验证必需列（qa_id, question, answer, category），返回标准化字典列表。</div>
    <div class="m-deps">→ pandas</div>
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
    <div class="sum-label">法规 BM25</div>
    <div class="sum-value">{bm25_avg}ms</div>
    <div class="sum-detail">min/max: {bm25_minmax}</div>
  </div>
  <div class="sum-item">
    <div class="sum-label">法规 Dense</div>
    <div class="sum-value">{dense_avg}ms</div>
    <div class="sum-detail">min/max: {dense_minmax}</div>
  </div>
  <div class="sum-item">
    <div class="sum-label">法规 RRF</div>
    <div class="sum-value">{rrf_avg}ms</div>
    <div class="sum-detail">min/max: {rrf_minmax}</div>
  </div>
  <div class="sum-item qa">
    <div class="sum-label" style="color:var(--qa-color)">QA 检索</div>
    <div class="sum-value" style="color:var(--qa-color)">{qa_avg}ms</div>
    <div class="sum-detail">min/max: {qa_minmax}</div>
  </div>
  <div class="sum-item">
    <div class="sum-label">Rerank（法规+QA）</div>
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
    <th>#</th><th>查询</th><th>法规 BM25 (ms)</th><th>法规 Dense (ms)</th>
    <th>法规 RRF (ms)</th><th class="qa-col">QA 检索 (ms)</th><th>Rerank (ms)</th><th>总计 (ms)</th>
    <th>法规候选</th><th class="qa-col">QA 候选</th><th>GPU峰值 (MB)</th>
  </tr></thead>
  <tbody>{query_rows}</tbody>
</table>

<!-- 配置详情 -->
<h2>5. 当前配置</h2>
<div class="info-card">
  <table class="info-table"><tbody>
    <tr class="section-header"><td colspan="2">法规 RAG</td></tr>
    <tr><td>知识库文档数</td><td>{num_docs}</td></tr>
    <tr><td>BM25 参数</td><td>k1={cfg["bm25_k1"]}, b={cfg["bm25_b"]}, backend={cfg["bm25_backend"]}</td></tr>
    <tr><td>BM25 召回数</td><td>{cfg["bm25_recall_k"]} / 路（定位 + 内容）</td></tr>
    <tr><td>Dense 模型</td><td>{cfg["dense_model"]}</td></tr>
    <tr><td>Dense 召回数</td><td>{cfg["dense_recall_k"]} / 路（定位 + 内容）</td></tr>
    <tr><td>Dense batch_size</td><td>{cfg["dense_batch_size"]}</td></tr>
    <tr><td>RRF 模式</td><td>{cfg["rrf_method"]}, k={cfg["rrf_k"]}</td></tr>
    <tr><td>RRF Top-K（送入 Reranker）</td><td>{cfg["rrf_top_k"]}</td></tr>
    <tr><td>Rerank Top-K（最终输出）</td><td>{cfg["rerank_top_k"]}</td></tr>
    <tr class="section-header"><td colspan="2">QA RAG</td></tr>
    <tr><td>QA 数据源</td><td>{cfg["qa_data_path"]}</td></tr>
    <tr><td>QA BM25 参数</td><td>k1={cfg["qa_bm25_k1"]}, b={cfg["qa_bm25_b"]}, backend={cfg["qa_bm25_backend"]}</td></tr>
    <tr><td>QA BM25 召回数</td><td>{cfg["qa_bm25_recall_k"]}</td></tr>
    <tr><td>QA Dense 召回数</td><td>{cfg["qa_dense_recall_k"]}</td></tr>
    <tr><td>QA RRF 参数</td><td>k={cfg["qa_rrf_k"]}, top_k={cfg["qa_rrf_top_k"]}</td></tr>
    <tr><td>QA Rerank Top-K</td><td>{cfg["qa_rerank_top_k"]}</td></tr>
    <tr><td>QA 相似度阈值</td><td>{cfg["qa_similarity_threshold"]}</td></tr>
    <tr class="section-header"><td colspan="2">Reranker（法规和 QA 分别独立精排）</td></tr>
    <tr><td>模型</td><td>{cfg["reranker_model_name"]}</td></tr>
    <tr><td>法规 Batch Size</td><td>{cfg["reranker_batch_size"]}（≥ RRF Top-K {cfg["rrf_top_k"]}）</td></tr>
    <tr><td>QA Batch Size</td><td>{cfg["qa_reranker_batch_size"]}（≥ QA RRF Top-K {cfg["qa_rrf_top_k"]}）</td></tr>
    <tr><td>分开精排原因</td><td>QA 问题文本极短（~20-40字符），与法规内容（~数百-数千字符）长度差异过大，合并 batch 会导致 QA 被无效 pad 到法规长度</td></tr>
    <tr><td>索引目录（法规）</td><td>{INDEX_DIR}</td></tr>
    <tr><td>索引目录（QA）</td><td>{cfg["qa_index_dir"]}</td></tr>
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
