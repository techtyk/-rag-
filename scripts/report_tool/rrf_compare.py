"""
RRF 融合策略对比实验。

对比三种 RRF 融合策略在固定输出 10 条时的检索质量差异：
  4-way：四路直接融合
  2-way 按索引：定位组(BM25+Dense) + 内容组(BM25+Dense)
  2-way 按检索器：BM25组(定位+正文) + Dense组(定位+正文)
用法：
    cd app && python -m tools.report2.rrf_compare [--num-questions 10] [--seed 42]
"""

import argparse
import gc
import json
import random
from datetime import datetime
from pathlib import Path

import torch

from config import (KB_PATH, QA_PATH, BM25_K1, BM25_B, BM25_BACKEND, BM25_RECALL_K,
                    DENSE_MODEL_PATH, DENSE_DEVICE, DENSE_RECALL_K, RRF_K)
from utils.doc_parser import parse_regulation
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever

REPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "doc" / "reports"
FIXED_OUTPUT = 10  # 固定输出条数
HALF_OUTPUT = 5    # 2-way 每路各取的条数


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


def _split_four_lists(bm25_results, dense_results):
    """将 BM25/Dense 结果拆分为四路排序列表。"""
    bm25_loc = [r for r in bm25_results if "location" in r.get("scores", {})]
    bm25_loc.sort(key=lambda x: x["scores"].get("location", 0), reverse=True)
    bm25_cont = [r for r in bm25_results if "content" in r.get("scores", {})]
    bm25_cont.sort(key=lambda x: x["scores"].get("content", 0), reverse=True)
    dense_loc = [r for r in dense_results if "location" in r.get("scores", {})]
    dense_loc.sort(key=lambda x: x["scores"].get("location", 0), reverse=True)
    dense_cont = [r for r in dense_results if "content" in r.get("scores", {})]
    dense_cont.sort(key=lambda x: x["scores"].get("content", 0), reverse=True)
    return bm25_loc, bm25_cont, dense_loc, dense_cont


def rrf_ranked_lists(lists_of_results: list[list[dict]], rrf_k: int) -> dict[tuple, float]:
    """对多路排序结果计算 RRF 分数。返回 {doc_key: rrf_score}。"""
    scores = {}
    for lst in lists_of_results:
        for rank, r in enumerate(lst, start=1):
            key = (r["file_name"], r["chapter"], r["article_no"])
            scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank)
    return scores


def _collect_raw_scores(bm25_results, dense_results) -> dict[tuple, dict]:
    """收集每个 chunk 的四路原始分数。返回 {doc_key: {bm25_loc, bm25_cont, dense_loc, dense_cont}}。"""
    raw = {}
    for r in bm25_results:
        key = (r["file_name"], r["chapter"], r["article_no"])
        raw.setdefault(key, {})
        if "location" in r.get("scores", {}):
            raw[key]["bm25_loc"] = r["scores"]["location"]
        if "content" in r.get("scores", {}):
            raw[key]["bm25_cont"] = r["scores"]["content"]
        raw[key].setdefault("content", r.get("content", ""))
        raw[key].setdefault("file_name", r["file_name"])
        raw[key].setdefault("chapter", r["chapter"])
        raw[key].setdefault("article_no", r["article_no"])
    for r in dense_results:
        key = (r["file_name"], r["chapter"], r["article_no"])
        raw.setdefault(key, {})
        if "location" in r.get("scores", {}):
            raw[key]["dense_loc"] = r["scores"]["location"]
        if "content" in r.get("scores", {}):
            raw[key]["dense_cont"] = r["scores"]["content"]
        raw[key].setdefault("content", r.get("content", ""))
        raw[key].setdefault("file_name", r["file_name"])
        raw[key].setdefault("chapter", r["chapter"])
        raw[key].setdefault("article_no", r["article_no"])
    return raw


def fuse_4way(bm25_results, dense_results, rrf_k, top_n):
    """四路 RRF：BM25定位、BM25内容、Dense定位、Dense内容 直接融合后截断。"""
    bm25_loc, bm25_cont, dense_loc, dense_cont = _split_four_lists(bm25_results, dense_results)
    rrf_scores = rrf_ranked_lists([bm25_loc, bm25_cont, dense_loc, dense_cont], rrf_k)
    raw_scores = _collect_raw_scores(bm25_results, dense_results)

    output = []
    for key, rrf_s in rrf_scores.items():
        info = raw_scores[key]
        output.append({
            "key": key,
            "score": rrf_s,
            "raw_scores": {k: v for k, v in info.items()
                           if k in ("bm25_loc", "bm25_cont", "dense_loc", "dense_cont")},
            "file_name": info["file_name"],
            "chapter": info["chapter"],
            "article_no": info["article_no"],
            "content": info["content"],
        })
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_n]


def fuse_2way(bm25_results, dense_results, rrf_k, half_n, axis="by_index"):
    """两路分组 RRF，axis 决定分组方向。
    axis="by_index":     定位组(BM25+Dense) + 内容组(BM25+Dense)
    axis="by_retriever": BM25组(定位+正文) + Dense组(定位+正文)
    """
    bm25_loc, bm25_cont, dense_loc, dense_cont = _split_four_lists(bm25_results, dense_results)
    raw_scores = _collect_raw_scores(bm25_results, dense_results)

    if axis == "by_retriever":
        group_a_rrf = rrf_ranked_lists([bm25_loc, bm25_cont], rrf_k)   # BM25组
        group_b_rrf = rrf_ranked_lists([dense_loc, dense_cont], rrf_k)  # Dense组
        label_a, label_b = "BM25组", "Dense组"
    else:
        group_a_rrf = rrf_ranked_lists([bm25_loc, dense_loc], rrf_k)    # 定位组
        group_b_rrf = rrf_ranked_lists([bm25_cont, dense_cont], rrf_k)  # 内容组
        label_a, label_b = "定位组", "正文组"

    group_a_keys = [k for k, _ in sorted(group_a_rrf.items(), key=lambda x: x[1], reverse=True)[:half_n]]
    group_b_keys = [k for k, _ in sorted(group_b_rrf.items(), key=lambda x: x[1], reverse=True)[:half_n]]

    seen = {}
    for key in group_a_keys:
        info = raw_scores[key]
        seen[key] = {
            "key": key,
            "score": group_a_rrf[key],
            "rrf_group_a": group_a_rrf[key],
            "rrf_group_b": None,
            "label_a": label_a,
            "label_b": label_b,
            "raw_scores": {k: v for k, v in info.items()
                           if k in ("bm25_loc", "bm25_cont", "dense_loc", "dense_cont")},
            "file_name": info["file_name"],
            "chapter": info["chapter"],
            "article_no": info["article_no"],
            "content": info["content"],
        }
    for key in group_b_keys:
        info = raw_scores[key]
        if key in seen:
            seen[key]["rrf_group_b"] = group_b_rrf[key]
        else:
            seen[key] = {
                "key": key,
                "score": group_b_rrf[key],
                "rrf_group_a": None,
                "rrf_group_b": group_b_rrf[key],
                "label_a": label_a,
                "label_b": label_b,
                "raw_scores": {k: v for k, v in info.items()
                               if k in ("bm25_loc", "bm25_cont", "dense_loc", "dense_cont")},
                "file_name": info["file_name"],
                "chapter": info["chapter"],
                "article_no": info["article_no"],
                "content": info["content"],
            }

    output = list(seen.values())
    return output


def generate_report(comparison_data: list[dict], seed: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    cards = ""
    for qi, item in enumerate(comparison_data):
        q_text = item["question"]
        cat = item["category_1"]
        cat2 = item["category_2"]
        results_4way = item["results_4way"]
        results_2way_idx = item["results_2way_idx"]
        results_2way_ret = item["results_2way_ret"]
        all_keys = item["all_keys"]

        # 计算每种方法的独有 chunk
        def get_only_set(result_list, all_others):
            my_keys = {r["key"] for r in result_list}
            return my_keys - all_others

        keys_4 = {r["key"] for r in results_4way}
        keys_idx = {r["key"] for r in results_2way_idx}
        keys_ret = {r["key"] for r in results_2way_ret}

        only_4way = get_only_set(results_4way, keys_idx | keys_ret)
        only_2idx = get_only_set(results_2way_idx, keys_4 | keys_ret)
        only_2ret = get_only_set(results_2way_ret, keys_4 | keys_idx)

        # 一个 chunk 出现在多列中就算 overlap（用于 data-overlap-key 和连线）
        multi_col_keys = set()
        for k in all_keys:
            count = (1 if k in keys_4 else 0) + (1 if k in keys_idx else 0) + (1 if k in keys_ret else 0)
            if count >= 2:
                multi_col_keys.add(k)

        def render_column(results, label, side, only_set, overlap_set):
            items = ""
            for j, r in enumerate(results):
                key = r["key"]
                is_only = key in only_set
                is_overlap = key in overlap_set
                only_class = " r-only" if is_only else ""
                overlap_attr = f' data-overlap-key="{key[0]}|{key[1]}|{key[2]}"' if is_overlap else ""

                # RRF 分数
                rrf_html = f'<span class="r-score rrf">{r["score"]:.6f}</span>'
                # 2-way 双 RRF 分数
                if r.get("rrf_group_a") is not None or r.get("rrf_group_b") is not None:
                    rrf_parts = []
                    la = r.get("label_a", "A组")
                    lb = r.get("label_b", "B组")
                    if r.get("rrf_group_a") is not None:
                        rrf_parts.append(f'{la}={r["rrf_group_a"]:.6f}')
                    if r.get("rrf_group_b") is not None:
                        rrf_parts.append(f'{lb}={r["rrf_group_b"]:.6f}')
                    rrf_html += f'<span class="r-rrf-detail">[{", ".join(rrf_parts)}]</span>'

                # 四路原始分数
                raw = r.get("raw_scores", {})
                raw_parts = []
                for src_key, src_label, src_cls in [
                    ("bm25_loc", "BM25定位", "bm25 loc"),
                    ("bm25_cont", "BM25正文", "bm25 cont"),
                    ("dense_loc", "Dense定位", "dense loc"),
                    ("dense_cont", "Dense正文", "dense cont"),
                ]:
                    if src_key in raw:
                        raw_parts.append(f'<span class="raw-{src_cls}">{src_label}={raw[src_key]:.4f}</span>')
                raw_display = " ".join(raw_parts)

                # 索引来源标签
                has_loc = "bm25_loc" in raw or "dense_loc" in raw
                has_cont = "bm25_cont" in raw or "dense_cont" in raw
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

                content_html = f"""<div class="r-content"><span class="c-preview">{preview_escaped}{"..." if len(full_content) > 150 else ""}</span><span class="c-full" style="display:none">{content_escaped}</span></div>"""
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
                        {rrf_html}
                        {source_tags}
                        {only_badge}
                    </div>
                    <div class="r-raw-scores">{raw_display}</div>
                    <div class="r-ref">{ref}</div>
                    {content_html}
                </div>"""
            return f"""<div class="method-col" data-side="{side}"><h3>{label}<small>({len(results)} 条)</small></h3><div class="method-results">{items}</div></div>"""

        col_4way = render_column(results_4way, "4-way RRF", "col0", only_4way, multi_col_keys)
        col_2idx = render_column(results_2way_idx, "2-way 按索引 (5+5)", "col1", only_2idx, multi_col_keys)
        col_2ret = render_column(results_2way_ret, "2-way 按检索器 (5+5)", "col2", only_2ret, multi_col_keys)

        # 计算三对重叠
        total_unique = len(keys_4 | keys_idx | keys_ret)
        overlap_4_idx = len(keys_4 & keys_idx)
        overlap_4_ret = len(keys_4 & keys_ret)
        overlap_idx_ret = len(keys_idx & keys_ret)
        overlap_all = len(keys_4 & keys_idx & keys_ret)

        cards += f"""<div class="q-card" data-q="{qi}">
            <div class="q-header" onclick="this.parentElement.classList.toggle('collapsed')">
                <span class="q-id">Q{qi+1}</span>
                <span class="q-text">{q_text}</span>
                <span class="q-cat">{cat} / {cat2}</span>
                <span class="q-overlap">重叠 4/idx={overlap_4_idx} 4/ret={overlap_4_ret} idx/ret={overlap_idx_ret} 三共={overlap_all}/{total_unique}</span>
                <span class="q-toggle">&#9660;</span>
            </div>
            <div class="q-body"><div class="methods-grid">{col_4way}{col_2idx}{col_2ret}</div></div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RRF 融合策略对比 - {ts}</title>
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
.r-score.rrf {{ font-family:monospace; color:#ffa726; margin-left:4px; font-size:0.85em; }}
.r-rrf-detail {{ font-family:monospace; color:#ce93d8; font-size:0.75em; margin-left:4px; }}
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
.r-raw-scores {{ font-family:monospace; color:var(--muted); font-size:0.72em;
  margin:2px 0 0 24px; display:flex; flex-wrap:wrap; gap:2px 8px; }}
.raw-bm25.loc {{ color:#64b5f6; }}
.raw-bm25.cont {{ color:#ba68c8; }}
.raw-dense.loc {{ color:#4dd0e1; }}
.raw-dense.cont {{ color:#f06292; }}
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
<h1>RRF 融合策略对比实验</h1>
<div class="meta">{ts} | 固定输出={FIXED_OUTPUT} | 2-way 分配={HALF_OUTPUT}+{HALF_OUTPUT} | RRF_K={RRF_K} | 模型=BM25+{DENSE_MODEL_PATH.split("/")[-1]} | 种子={seed}</div>
<div class="desc">
<b>实验设计：</b>固定输出 {FIXED_OUTPUT} 条 chunk，对比三种 RRF 策略的检索质量。<br>
<b>4-way：</b>BM25定位 + BM25正文 + Dense定位 + Dense正文 → 四路 RRF → 取前 {FIXED_OUTPUT}<br>
<b>2-way 按索引：</b>定位组(BM25+Dense) RRF → 前 {HALF_OUTPUT}；正文组(BM25+Dense) RRF → 前 {HALF_OUTPUT}<br>
<b>2-way 按检索器：</b>BM25组(定位+正文) RRF → 前 {HALF_OUTPUT}；Dense组(定位+正文) RRF → 前 {HALF_OUTPUT}<br>
<b>独有标记：</b><span style="color:var(--only-color)">红色左边框 + 独有标签</span>的条目仅出现在当前策略中。<br>
<b>悬停高亮：</b>鼠标悬停一个 chunk 时，所有列中相同的 chunk 同时高亮显示。
</div>
{cards}
<div class="footer">Generated by tools/report2/rrf_compare.py</div>
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
    out_path = REPORT_DIR / f"rrf_compare_{ts}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n报告已生成：{out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="RRF 融合策略对比实验")
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

    # 逐题对比
    comparison_data = []
    for qi, item in enumerate(sampled):
        q = item["question"]
        print(f"  Q{qi+1}: {q}")

        bm25_results = bm25.search(q, top_k=BM25_RECALL_K)
        dense_results = dense.search(q, top_k=DENSE_RECALL_K)

        results_4way = fuse_4way(bm25_results, dense_results, RRF_K, FIXED_OUTPUT)
        results_2way_idx = fuse_2way(bm25_results, dense_results, RRF_K, HALF_OUTPUT, axis="by_index")
        results_2way_ret = fuse_2way(bm25_results, dense_results, RRF_K, HALF_OUTPUT, axis="by_retriever")

        keys_4 = {r["key"] for r in results_4way}
        keys_idx = {r["key"] for r in results_2way_idx}
        keys_ret = {r["key"] for r in results_2way_ret}

        comparison_data.append({
            "question": q,
            "category_1": item["category_1"],
            "category_2": item["category_2"],
            "results_4way": results_4way,
            "results_2way_idx": results_2way_idx,
            "results_2way_ret": results_2way_ret,
            "all_keys": keys_4 | keys_idx | keys_ret,
        })

    # 释放资源
    del dense; gc.collect(); torch.cuda.empty_cache()

    # 生成报告
    generate_report(comparison_data, args.seed)


if __name__ == "__main__":
    main()
