"""
Reranker 模型对比实验。

固定使用 4-way RRF 召回，对比三种 Reranker 模型的精排效果。
用法：
    cd app && python -m scripts.report_tool.rerank_compare [--num-questions 10] [--seed 42]
"""

import argparse
import gc
import json
import random
from datetime import datetime
from pathlib import Path

import torch

from config import (KB_PATH, QA_PATH, BM25_K1, BM25_B, BM25_BACKEND, BM25_RECALL_K,
                    DENSE_MODEL_PATH, DENSE_DEVICE, DENSE_BATCH_SIZE,
                    DENSE_RECALL_K, RRF_K, RERANK_TOP_K)
from utils.doc_parser import parse_regulation
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from rerank.reranker import BGEReranker, GTEReranker, Qwen3Reranker

REPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "doc" / "reports"
FIXED_OUTPUT = 10
RERANK_OUTPUT = RERANK_TOP_K

RERANKER_CONFIGS = [
    ("bge", "BGE-reranker-v2-m3", "/home/moga/models/reranker/bge-reranker-v2-m3", BGEReranker),
    ("gte", "GTE-multilingual-reranker", "/home/moga/models/reranker/gte-multilingual-reranker-base", GTEReranker),
    ("qwen3", "Qwen3-Reranker-0.6B", "/home/moga/models/reranker/Qwen3-Reranker-0.6B", Qwen3Reranker),
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


def _rrf_score(ranked_lists: list[list[dict]], rrf_k: int) -> dict[tuple, float]:
    scores = {}
    for lst in ranked_lists:
        for rank, r in enumerate(lst, start=1):
            key = (r["file_name"], r["chapter"], r["article_no"])
            scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank)
    return scores


def _split_four_lists(bm25_results, dense_results):
    bm25_loc = [r for r in bm25_results if "location" in r.get("scores", {})]
    bm25_loc.sort(key=lambda x: x["scores"].get("location", 0), reverse=True)
    bm25_cont = [r for r in bm25_results if "content" in r.get("scores", {})]
    bm25_cont.sort(key=lambda x: x["scores"].get("content", 0), reverse=True)
    dense_loc = [r for r in dense_results if "location" in r.get("scores", {})]
    dense_loc.sort(key=lambda x: x["scores"].get("location", 0), reverse=True)
    dense_cont = [r for r in dense_results if "content" in r.get("scores", {})]
    dense_cont.sort(key=lambda x: x["scores"].get("content", 0), reverse=True)
    return bm25_loc, bm25_cont, dense_loc, dense_cont


def fuse_4way(bm25_results, dense_results, rrf_k, top_n):
    bm25_loc, bm25_cont, dense_loc, dense_cont = _split_four_lists(bm25_results, dense_results)
    rrf_scores = _rrf_score([bm25_loc, bm25_cont, dense_loc, dense_cont], rrf_k)

    seen = {}
    for r in bm25_results + dense_results:
        key = (r["file_name"], r["chapter"], r["article_no"])
        if key not in seen:
            seen[key] = r

    output = []
    for key, rrf_s in rrf_scores.items():
        r = seen[key]
        output.append({
            "key": key,
            "rrf_score": rrf_s,
            "file_name": r["file_name"],
            "chapter": r["chapter"],
            "article_no": r["article_no"],
            "content": r["content"],
            "scores": r.get("scores", {}),
            "matched_tokens": r.get("matched_tokens", {}),
            "source": r.get("source", ""),
        })
    output.sort(key=lambda x: x["rrf_score"], reverse=True)
    return output[:top_n]


def generate_report(comparison_data: list[dict], seed: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_labels = {cfg[0]: cfg[1] for cfg in RERANKER_CONFIGS}
    model_ids = [cfg[0] for cfg in RERANKER_CONFIGS]
    col_sides = ["col0", "col1", "col2"]

    cards = ""
    for qi, item in enumerate(comparison_data):
        q_text = item["question"]
        cat = item["category_1"]
        cat2 = item["category_2"]
        reranked = item["reranked"]

        # 统计重叠
        key_sets = {mid: {r["key"] for r in reranked[mid]} for mid in model_ids}
        total_unique = len(set.union(*key_sets.values()))
        overlaps = {}
        for a in range(len(model_ids)):
            for b in range(a + 1, len(model_ids)):
                ma, mb = model_ids[a], model_ids[b]
                overlaps[f"{ma}/{mb}"] = len(key_sets[ma] & key_sets[mb])

        # 独有 chunk
        only_sets = {}
        for mid in model_ids:
            others = set.union(*(key_sets[m] for m in model_ids if m != mid))
            only_sets[mid] = key_sets[mid] - others

        # 多列共现
        multi_col_keys = set()
        for k in set.union(*key_sets.values()):
            count = sum(1 for mid in model_ids if k in key_sets[mid])
            if count >= 2:
                multi_col_keys.add(k)

        def render_column(results, label, side, only_set):
            items = ""
            for j, r in enumerate(results):
                key = r["key"]
                is_only = key in only_set
                is_overlap = key in multi_col_keys
                only_class = " r-only" if is_only else ""
                overlap_attr = f' data-overlap-key="{key[0]}|{key[1]}|{key[2]}"' if is_overlap else ""

                rerank_score = f'<span class="r-score rerank">{r["rerank_score"]:.4f}</span>'
                rrf_score = f'<span class="r-score rrf">RRF={r["rrf_score"]:.6f}</span>'

                # 索引来源标签
                raw = r.get("scores", {})
                has_loc = "location" in raw
                has_cont = "content" in raw
                source_tags = ""
                if has_loc:
                    source_tags += '<span class="r-source location">定位</span>'
                if has_cont:
                    source_tags += '<span class="r-source content">正文</span>'

                only_badge = '<span class="r-only-badge">独有</span>' if is_only else ""

                ref = f"{r['file_name']} · {r['chapter']} · {r['article_no']}"
                full_content = r["content"]
                preview = full_content[:150]
                content_escaped = full_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                preview_escaped = preview.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

                content_html = f'<div class="r-content"><span class="c-preview">{preview_escaped}{"..." if len(full_content) > 150 else ""}</span><span class="c-full" style="display:none">{content_escaped}</span></div>'
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
                items += f"""<div class="r-item{only_class}"{overlap_attr}>
                    <div class="r-line1">
                        <span class="r-rank">#{j+1}</span>
                        {rerank_score}
                        {rrf_score}
                        {source_tags}
                        {only_badge}
                    </div>
                    <div class="r-ref">{ref}</div>
                    {content_html}
                </div>"""
            return f"""<div class="method-col" data-side="{side}"><h3>{label}<small>({len(results)} 条)</small></h3><div class="method-results">{items}</div></div>"""

        cols = []
        for idx, (mid, label, _, _) in enumerate(RERANKER_CONFIGS):
            cols.append(render_column(reranked[mid], label, col_sides[idx], only_sets[mid]))

        overlap_text = " ".join(f"{k}={v}" for k, v in overlaps.items())
        cards += f"""<div class="q-card" data-q="{qi}">
            <div class="q-header" onclick="this.parentElement.classList.toggle('collapsed')">
                <span class="q-id">Q{qi+1}</span>
                <span class="q-text">{q_text}</span>
                <span class="q-cat">{cat} / {cat2}</span>
                <span class="q-overlap">重叠 {overlap_text} | 唯一chunk={total_unique}</span>
                <span class="q-toggle">&#9660;</span>
            </div>
            <div class="q-body"><div class="methods-grid">{"".join(cols)}</div></div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reranker 模型对比 - {ts}</title>
<style>
:root {{ --bg:#0f1117; --card:#1a1d28; --border:#2a2d3a; --text:#e0e0e0;
  --muted:#8b8fa3; --loc-accent:#4fc3f7; --cont-accent:#ab47bc;
  --col0:#4fc3f7; --col1:#66bb6a; --col2:#ffa726; --only-color:#ff6b6b; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
  background:var(--bg); color:var(--text); line-height:1.6; padding:20px;
  max-width:1800px; margin:0 auto; }}
h1 {{ text-align:center; font-size:1.5em; margin-bottom:6px; color:#fff; }}
.meta {{ text-align:center; color:var(--muted); margin-bottom:20px; font-size:0.85em; }}
.desc {{ max-width:1100px; margin:0 auto 20px; padding:12px 18px; background:var(--card);
  border:1px solid var(--border); border-radius:8px; font-size:0.82em; color:var(--muted); }}
.desc b {{ color:var(--text); }}
.q-card {{ background:var(--card); border:1px solid var(--border);
  border-radius:12px; margin-bottom:18px; overflow:hidden; position:relative; }}
.q-header {{ display:flex; align-items:center; padding:12px 18px; cursor:pointer;
  gap:12px; transition:background 0.15s; flex-wrap:wrap; position:relative; z-index:2; }}
.q-header:hover {{ background:#252836; }}
.q-id {{ background:#2a2d3a; border-radius:6px; padding:2px 10px;
  font-size:0.78em; color:var(--muted); flex-shrink:0; }}
.q-text {{ flex:1; font-weight:600; color:#fff; min-width:200px; }}
.q-cat {{ font-size:0.72em; color:var(--muted); flex-shrink:0; }}
.q-overlap {{ font-size:0.7em; color:#ffa726; flex-shrink:0; }}
.q-toggle {{ color:var(--muted); font-size:1.1em; flex-shrink:0; transition:transform 0.2s; }}
.q-card.collapsed .q-toggle {{ transform:rotate(-90deg); }}
.q-card.collapsed .q-body {{ display:none; }}
.q-body {{ position:relative; }}
.methods-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; padding:12px;
  position:relative; z-index:2; }}
@media (max-width:1200px) {{ .methods-grid {{ grid-template-columns:1fr; }} }}
.method-col {{ background:var(--bg); border:1px solid var(--border);
  border-radius:8px; overflow:hidden; }}
.method-col h3 {{ font-size:0.82em; padding:8px 12px; margin:0; }}
.method-col[data-side="col0"] h3 {{ color:var(--col0); background:rgba(79,195,247,0.06); }}
.method-col[data-side="col1"] h3 {{ color:var(--col1); background:rgba(102,187,106,0.06); }}
.method-col[data-side="col2"] h3 {{ color:var(--col2); background:rgba(255,167,38,0.06); }}
.method-col h3 small {{ font-weight:normal; color:var(--muted); margin-left:6px; }}
.r-item {{ padding:8px 12px; border-bottom:1px solid var(--border); font-size:0.8em;
  transition:background 0.15s; }}
.r-item:last-child {{ border-bottom:none; }}
.r-item:hover {{ background:rgba(255,255,255,0.03); }}
.r-item.match-highlight {{ background:rgba(255,235,59,0.08); box-shadow:inset 0 0 0 1px rgba(255,235,59,0.25); }}
.r-item.r-only {{ border-left:3px solid var(--only-color); background:rgba(255,107,107,0.05); }}
.r-item.r-only:hover {{ background:rgba(255,107,107,0.1); }}
.r-line1 {{ display:flex; align-items:center; gap:4px; flex-wrap:wrap; }}
.r-rank {{ font-weight:700; color:#fff; }}
.r-score.rerank {{ font-family:monospace; color:#ffa726; margin-left:4px; font-size:0.95em; font-weight:600; }}
.r-score.rrf {{ font-family:monospace; color:var(--muted); font-size:0.72em; margin-left:2px; }}
.r-source {{ display:inline-block; padding:1px 6px; border-radius:3px;
  font-size:0.72em; margin-left:4px; }}
.r-source.location {{ background:rgba(79,195,247,0.1); color:var(--loc-accent);
  border:1px solid rgba(79,195,247,0.2); }}
.r-source.content {{ background:rgba(171,71,188,0.1); color:var(--cont-accent);
  border:1px solid rgba(171,71,188,0.2); }}
.r-only-badge {{ display:inline-block; padding:1px 6px; border-radius:3px;
  font-size:0.7em; margin-left:6px; background:rgba(255,107,107,0.15);
  color:var(--only-color); border:1px solid rgba(255,107,107,0.3);
  font-weight:600; }}
.r-ref {{ font-size:0.72em; color:var(--muted); margin:2px 0; }}
.r-content {{ color:#c0c4d0; font-size:0.8em; }}
.r-content-toggle {{ display:inline-block; color:var(--loc-accent); font-size:0.75em;
  cursor:pointer; margin-top:2px; opacity:0.7; }}
.r-content-toggle:hover {{ opacity:1; text-decoration:underline; }}
.footer {{ text-align:center; color:var(--muted); font-size:0.75em;
  margin-top:30px; padding:10px; }}
</style>
</head>
<body>
<h1>Reranker 模型对比实验</h1>
<div class="meta">{ts} | 4-way RRF 召回={FIXED_OUTPUT} | Reranker 精排={RERANK_OUTPUT} | RRF_K={RRF_K} | 种子={seed}</div>
<div class="desc">
<b>实验设计：</b>固定使用 4-way RRF 召回 {FIXED_OUTPUT} 条，然后用三种 Reranker 模型精排取前 {RERANK_OUTPUT} 条。<br>
<b>BGE-reranker-v2-m3：</b>XLMRoberta Cross-Encoder，~568M 参数，CMTEB-R=72.16<br>
<b>GTE-multilingual-reranker-base：</b>ONNX Cross-Encoder，0.3B 参数，CMTEB-R=74.08<br>
<b>Qwen3-Reranker-0.6B：</b>LLM yes/no logit，0.6B 参数，MTEB-R=65.80<br>
<b>独有标记：</b><span style="color:var(--only-color)">红色左边框 + 独有标签</span>的条目仅出现在当前模型中。<br>
<b>悬停高亮：</b>鼠标悬停一个 chunk 时，所有列中相同的 chunk 同时高亮显示。
</div>
{cards}
<div class="footer">Generated by scripts/report_tool/rerank_compare.py</div>
<script>
(function() {{
    function setupHighlight(qCard) {{
        const items = qCard.querySelectorAll('.r-item[data-overlap-key]');
        items.forEach(el => {{
            el.addEventListener('mouseenter', () => {{
                const key = el.getAttribute('data-overlap-key');
                qCard.querySelectorAll('.r-item[data-overlap-key="' + key + '"]').forEach(e => {{
                    e.classList.add('match-highlight');
                }});
            }});
            el.addEventListener('mouseleave', () => {{
                const key = el.getAttribute('data-overlap-key');
                qCard.querySelectorAll('.r-item[data-overlap-key="' + key + '"]').forEach(e => {{
                    e.classList.remove('match-highlight');
                }});
            }});
        }});
    }}
    document.querySelectorAll('.q-card').forEach(setupHighlight);
}})();
</script>
</body>
</html>"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"rerank_compare_{ts}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n报告已生成：{out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Reranker 模型对比实验")
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 解析知识库
    docs_loc, docs_content, metadatas = parse_regulation(str(KB_PATH))
    print(f"知识库：{len(metadatas)} 个条款")

    # 构建检索器
    print("构建 BM25 索引...")
    bm25 = BM25Retriever.build(
        docs_loc, docs_content, metadatas,
        k1=BM25_K1, b=BM25_B, backend=BM25_BACKEND,
    )

    print("构建 Dense 索引...")
    dense = DenseRetriever.build(
        docs_loc, docs_content, metadatas,
        model_path=DENSE_MODEL_PATH, device=DENSE_DEVICE, batch_size=4,
    )

    # 抽样问题
    all_questions = load_questions(str(QA_PATH))
    random.seed(args.seed)
    sampled = random.sample(all_questions, min(args.num_questions, len(all_questions)))
    print(f"抽样 {len(sampled)} 条问题（种子={args.seed}）")

    # 先做 4-way RRF 召回，收集所有结果
    print("\n--- 4-way RRF 召回 ---")
    rrf_results_per_query = []
    for qi, item in enumerate(sampled):
        q = item["question"]
        print(f"  Q{qi+1}: {q}")
        bm25_results = bm25.search(q, top_k=BM25_RECALL_K)
        dense_results = dense.search(q, top_k=DENSE_RECALL_K)
        rrf_results = fuse_4way(bm25_results, dense_results, RRF_K, FIXED_OUTPUT)
        rrf_results_per_query.append(rrf_results)

    # 释放检索器
    del dense; del bm25; gc.collect(); torch.cuda.empty_cache()

    # 逐个加载 Reranker，逐题精排
    comparison_data = [{"question": item["question"],
                        "category_1": item["category_1"],
                        "category_2": item["category_2"],
                        "reranked": {}} for item in sampled]

    for mid, label, model_path, cls in RERANKER_CONFIGS:
        print(f"\n--- 加载 Reranker: {label} ---")
        reranker = cls(model_path, device=DENSE_DEVICE)

        for qi, rrf_results in enumerate(rrf_results_per_query):
            documents = [r["content"] for r in rrf_results]
            scores = reranker.rerank(sampled[qi]["question"], documents)

            reranked = []
            for r, s in zip(rrf_results, scores):
                entry = dict(r)
                entry["rerank_score"] = s
                reranked.append(entry)
            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            comparison_data[qi]["reranked"][mid] = reranked[:RERANK_OUTPUT]

        del reranker; gc.collect(); torch.cuda.empty_cache()

    # 生成报告
    generate_report(comparison_data, args.seed)


if __name__ == "__main__":
    main()
