"""
Dense 模型对比报告生成器。

逐个加载候选模型，在同一批抽样问题上检索，输出自包含 HTML 对比报告。
用法：
    cd app && python -m scripts.report_tool.dense_compare [--num-questions 10] [--seed 42]
"""

import argparse
import gc
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch

from config import KB_PATH, QA_PATH, DENSE_RECALL_K, RERANK_TOP_K
from utils.doc_parser import parse_regulation
from retrieval.dense import DenseRetriever

REPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "doc" / "reports"

DENSE_MODELS = [
    {"name": "Qwen3-0.6B", "path": "/home/moga/models/Qwen3-Embedding-0.6B"},
    {"name": "BGE-M3", "path": "/home/moga/models/BGE-M3"},
    {"name": "Jina-v2-zh", "path": "/home/moga/models/jina-embeddings-v2-base-zh"},
    {"name": "BCE", "path": "/home/moga/models/bce-embedding-base_v1"},
]


def load_questions(path: str) -> list[dict]:
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
    return questions


def run_model(model_cfg: dict, docs_loc, docs_content, metadatas,
              questions: list[dict]) -> dict:
    """加载单个模型，建索引，检索，返回结果和统计。"""
    model_name = model_cfg["name"]
    model_path = model_cfg["path"]

    print(f"\n{'='*60}")
    print(f"开始评估：{model_name} ({model_path})")
    print(f"{'='*60}")

    # 建索引
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    t0 = time.time()
    try:
        retriever = DenseRetriever.build(
            docs_loc, docs_content, metadatas,
            model_path=model_path, device="cuda", batch_size=4,
        )
    except Exception as e:
        print(f"  模型加载/索引失败：{e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "stats": {"model_name": model_name, "model_path": model_path,
                       "dim": 0, "emb_size_mb": 0, "gpu_peak_mb": 0,
                       "index_time_s": 0, "avg_query_ms": 0,
                       "num_articles": len(metadatas), "error": str(e)},
            "queries": [],
        }
    index_time = time.time() - t0
    gpu_peak_mb = 0
    if torch.cuda.is_available():
        gpu_peak_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)

    dim = retriever.dim
    # 索引大小估算：2 个 FAISS 索引 × N 条 × dim 维 × 4 bytes
    emb_size_mb = len(metadatas) * dim * 4 * 2 / 1024 / 1024

    # 检索
    query_results = []
    total_query_time = 0
    for item in questions:
        t1 = time.time()
        results = retriever.search(item["question"], top_k=DENSE_RECALL_K)
        total_query_time += time.time() - t1
        query_results.append({
            "question": item["question"],
            "category_1": item["category_1"],
            "category_2": item["category_2"],
            "results": results,
        })

    avg_query_time = total_query_time / len(questions) if questions else 0

    stats = {
        "model_name": model_name,
        "model_path": model_path,
        "dim": dim,
        "emb_size_mb": round(emb_size_mb, 1),
        "gpu_peak_mb": gpu_peak_mb,
        "index_time_s": round(index_time, 1),
        "avg_query_ms": round(avg_query_time * 1000, 1),
        "num_articles": len(metadatas),
    }
    print(f"  维度={dim}, GPU峰值={gpu_peak_mb}MB, 索引耗时={index_time:.1f}s, 平均查询={avg_query_time*1000:.1f}ms")

    # 释放模型和 GPU 显存
    del retriever
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"stats": stats, "queries": query_results}


def generate_report(all_results: list[dict], seed: int) -> Path:
    # 过滤掉失败的模型
    valid_results = [r for r in all_results if r["queries"]]
    failed_models = [r["stats"]["model_name"] for r in all_results if not r["queries"]]
    if failed_models:
        print(f"跳过失败的模型：{', '.join(failed_models)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_names = [r["stats"]["model_name"] for r in valid_results]
    stats_rows = ""
    for r in all_results:
        s = r["stats"]
        err = s.get("error", "")
        stats_rows += f"""<tr>
            <td>{s['model_name']}</td>
            <td>{s['dim']}</td>
            <td>{s['emb_size_mb']}</td>
            <td>{s['gpu_peak_mb']}</td>
            <td>{s['index_time_s']}</td>
            <td>{s['avg_query_ms']}</td>
        </tr>"""
    # 失败模型单独列出
    for r in all_results:
        if r["stats"].get("error"):
            stats_rows += f"""<tr style="opacity:0.4"><td>{r['stats']['model_name']}</td>
                <td colspan="5" style="color:#f85149">加载失败：{r['stats']['error'][:80]}</td></tr>"""

    # 每个问题生成一行对比
    question_cards = ""
    num_q = len(valid_results[0]["queries"])
    for qi in range(num_q):
        q_text = valid_results[0]["queries"][qi]["question"]
        cat = valid_results[0]["queries"][qi]["category_1"]
        cat2 = valid_results[0]["queries"][qi]["category_2"]

        cols = ""
        for r in valid_results:
            mn = r["stats"]["model_name"]
            q_results = r["queries"][qi]["results"]
            total = len(q_results)
            items = ""
            for j, res in enumerate(q_results):
                scores = res.get("scores", {"location": res["score"]} if res.get("source") == "location" else {"content": res["score"]})
                full_content = res["content"]
                preview = full_content[:150]
                ref = f"{res['file_name']} · {res['chapter']} · {res['article_no']}"
                hidden_cls = " r-item-extra" if j >= RERANK_TOP_K else ""

                # 分数和来源标签
                score_parts = []
                source_tags = ""
                for src_key, src_label in [("location", "定位"), ("content", "正文")]:
                    if src_key in scores:
                        score_parts.append(f"{src_label}={scores[src_key]:.4f}")
                        source_tags += f'<span class="r-source {src_key}">{src_label}</span>'
                score_display = " | ".join(score_parts)

                content_html = f"""<div class="r-content"><span class="c-preview">{preview}{"..." if len(full_content) > 150 else ""}</span><span class="c-full" style="display:none">{full_content}</span></div>"""
                if len(full_content) > 150:
                    content_html += """<span class="r-content-toggle" onclick="
                        var box=this.previousElementSibling;
                        var pv=box.querySelector('.c-preview');
                        var fl=box.querySelector('.c-full');
                        if(fl.style.display==='none'){
                            pv.style.display='none'; fl.style.display='inline';
                            this.textContent='收起';
                        }else{
                            pv.style.display='inline'; fl.style.display='none';
                            this.textContent='展开';
                        }
                    ">展开</span>"""
                items += f"""<div class="r-item{hidden_cls}">
                    <span class="r-rank">#{j+1}</span>
                    <span class="r-score">{score_display}</span>
                    {source_tags}
                    <div class="r-ref">{ref}</div>
                    {content_html}
                </div>"""
            extra = total - RERANK_TOP_K
            toggle = ""
            if extra > 0:
                toggle = f"""<button class="r-toggle-btn" onclick="
                    var extras=this.parentElement.querySelectorAll('.r-item-extra');
                    var show=extras.length>0&&getComputedStyle(extras[0]).display==='none';
                    extras.forEach(e=>e.style.display=show?'block':'none');
                    this.textContent=show?'收起 (保留前{RERANK_TOP_K}条)':'展开全部 ({total}条)';
                ">展开全部 ({total}条)</button>"""
            cols += f"""<div class="model-col"><h3>{mn}</h3><div class="model-results">{items}{toggle}</div></div>"""

        question_cards += f"""<div class="q-card">
            <div class="q-header" onclick="this.parentElement.classList.toggle('collapsed')">
                <span class="q-id">Q{qi+1}</span>
                <span class="q-text">{q_text}</span>
                <span class="q-cat">{cat} / {cat2}</span>
                <span class="q-toggle">&#9660;</span>
            </div>
            <div class="q-body"><div class="models-grid">{cols}</div></div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dense 模型对比报告 - {ts}</title>
<style>
:root {{ --bg:#0f1117; --card:#1a1d28; --border:#2a2d3a; --text:#e0e0e0;
  --muted:#8b8fa3; --loc-accent:#4fc3f7; --cont-accent:#ab47bc; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
  background:var(--bg); color:var(--text); line-height:1.6; padding:20px;
  max-width:1600px; margin:0 auto; }}
h1 {{ text-align:center; font-size:1.5em; margin-bottom:6px; color:#fff; }}
.meta {{ text-align:center; color:var(--muted); margin-bottom:20px; font-size:0.85em; }}
table.stats {{ width:100%; border-collapse:collapse; margin-bottom:24px; font-size:0.85em; }}
table.stats th {{ background:rgba(79,195,247,0.1); color:var(--loc-accent);
  padding:8px 12px; text-align:left; border:1px solid var(--border); }}
table.stats td {{ padding:8px 12px; border:1px solid var(--border); }}
.q-card {{ background:var(--card); border:1px solid var(--border);
  border-radius:12px; margin-bottom:18px; overflow:hidden; }}
.q-header {{ display:flex; align-items:center; padding:12px 18px; cursor:pointer;
  gap:12px; transition:background 0.15s; }}
.q-header:hover {{ background:#252836; }}
.q-id {{ background:#2a2d3a; border-radius:6px; padding:2px 10px;
  font-size:0.78em; color:var(--muted); flex-shrink:0; }}
.q-text {{ flex:1; font-weight:600; color:#fff; }}
.q-cat {{ font-size:0.72em; color:var(--muted); flex-shrink:0; }}
.q-toggle {{ color:var(--muted); font-size:1.1em; flex-shrink:0; transition:transform 0.2s; }}
.q-card.collapsed .q-toggle {{ transform:rotate(-90deg); }}
.q-card.collapsed .q-body {{ display:none; }}
.models-grid {{ display:grid; grid-template-columns:repeat({len(model_names)},1fr);
  gap:12px; padding:12px; }}
@media (max-width:1200px) {{ .models-grid {{ grid-template-columns:1fr 1fr; }} }}
@media (max-width:700px) {{ .models-grid {{ grid-template-columns:1fr; }} }}
.model-col {{ background:var(--bg); border:1px solid var(--border);
  border-radius:8px; overflow:hidden; }}
.model-col h3 {{ font-size:0.85em; padding:8px 12px; color:var(--loc-accent);
  background:rgba(79,195,247,0.06); margin:0; }}
.r-item {{ padding:8px 12px; border-bottom:1px solid var(--border); font-size:0.82em; }}
.r-item:last-child {{ border-bottom:none; }}
.r-rank {{ font-weight:700; color:#fff; }}
.r-score {{ font-family:monospace; color:var(--muted); margin-left:8px; font-size:0.85em; }}
.r-source {{ display:inline-block; padding:1px 6px; border-radius:3px;
  font-size:0.72em; margin-left:6px; }}
.r-source.location {{ background:rgba(79,195,247,0.1); color:var(--loc-accent);
  border:1px solid rgba(79,195,247,0.2); }}
.r-source.content {{ background:rgba(171,71,188,0.1); color:var(--cont-accent);
  border:1px solid rgba(171,71,188,0.2); }}
.r-ref {{ font-size:0.75em; color:var(--muted); margin:2px 0; }}
.r-content {{ color:#c0c4d0; font-size:0.82em; }}
.r-item-extra {{ display:none; }}
.r-toggle-btn {{ width:100%; padding:6px; margin-top:4px; border:1px dashed var(--border);
  background:transparent; color:var(--muted); cursor:pointer; font-size:0.78em;
  border-radius:4px; transition:color 0.15s; }}
.r-toggle-btn:hover {{ color:var(--loc-accent); }}
.r-content-toggle {{ display:inline-block; color:var(--loc-accent); font-size:0.75em;
  cursor:pointer; margin-top:2px; opacity:0.7; }}
.r-content-toggle:hover {{ opacity:1; text-decoration:underline; }}
.footer {{ text-align:center; color:var(--muted); font-size:0.75em;
  margin-top:30px; padding:10px; }}
</style>
</head>
<body>
<h1>Dense 模型对比报告</h1>
<div class="meta">{ts} | 每路召回={DENSE_RECALL_K} | 默认展示={RERANK_TOP_K} | 种子={seed} | 模型数={len(model_names)}</div>
<table class="stats">
<tr><th>模型</th><th>向量维度</th><th>索引大小(MB)</th><th>GPU峰值显存(MB)</th><th>建索引耗时(s)</th><th>平均单次查询(ms)<br><small style="font-weight:normal;color:var(--muted)">单条 query 双路检索耗时的均值</small></th></tr>
{stats_rows}
</table>
{question_cards}
<div class="footer">Generated by tools/dense_compare.py</div>
</body>
</html>"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"dense_compare_{ts}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n报告已生成：{out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Dense 模型对比报告")
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 解析知识库
    docs_loc, docs_content, metadatas = parse_regulation(str(KB_PATH))
    print(f"知识库：{len(metadatas)} 个条款")

    # 抽样问题
    all_questions = load_questions(str(QA_PATH))
    random.seed(args.seed)
    sampled = random.sample(all_questions, min(args.num_questions, len(all_questions)))
    print(f"抽样 {len(sampled)} 条问题（种子={args.seed}）")

    # 逐个模型评估
    all_results = []
    for model_cfg in DENSE_MODELS:
        result = run_model(model_cfg, docs_loc, docs_content, metadatas,
                           sampled)
        all_results.append(result)

    # 生成报告
    generate_report(all_results, args.seed)


if __name__ == "__main__":
    main()
