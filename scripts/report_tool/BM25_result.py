"""
RAG Pipeline 检索报告生成器。

生成自包含 HTML 报告，展示检索结果、匹配词元高亮、可折叠卡片。
用法：
    cd app && python -m scripts.report_tool.BM25_result [--num-questions 10] [--top-k 5]
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from config import KB_PATH, QA_PATH, BM25_K1, BM25_B, BM25_BACKEND, BM25_RECALL_K
from utils.doc_parser import parse_regulation
from utils.tokenizer import tokenize_for_query
from retrieval.retrieve import Retriever

REPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "doc" / "reports"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG 检索报告 - {timestamp}</title>
<style>
:root {{
  --bg: #0f1117; --card-bg: #1a1d28; --border: #2a2d3a;
  --text: #e0e0e0; --text-muted: #8b8fa3;
  --loc-accent: #4fc3f7; --content-accent: #ab47bc;
  --loc-bg: rgba(79,195,247,0.08); --content-bg: rgba(171,71,188,0.08);
  --loc-hl: rgba(79,195,247,0.25); --content-hl: rgba(171,71,188,0.25);
  --hover: #252836;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
  background:var(--bg); color:var(--text); line-height:1.7; padding:20px;
  max-width:1400px; margin:0 auto; }}
h1 {{ text-align:center; font-size:1.5em; margin-bottom:6px; color:#fff; }}
.meta {{ text-align:center; color:var(--text-muted); margin-bottom:20px; font-size:0.85em; }}
.stats-bar {{ display:flex; justify-content:center; gap:24px; margin-bottom:28px; flex-wrap:wrap; }}
.stat {{ text-align:center; padding:8px 18px; background:var(--card-bg);
  border-radius:8px; border:1px solid var(--border); min-width:110px; }}
.stat-value {{ font-size:1.3em; font-weight:700; color:#fff; }}
.stat-label {{ font-size:0.72em; color:var(--text-muted); margin-top:2px; }}
.q-card {{ background:var(--card-bg); border:1px solid var(--border);
  border-radius:12px; margin-bottom:18px; overflow:hidden; }}
.q-header {{ display:flex; align-items:center; padding:12px 18px; cursor:pointer;
  gap:12px; transition:background 0.15s; }}
.q-header:hover {{ background:var(--hover); }}
.q-id {{ background:#2a2d3a; border-radius:6px; padding:2px 10px;
  font-size:0.78em; color:var(--text-muted); flex-shrink:0; }}
.q-text {{ flex:1; font-weight:600; font-size:0.95em; color:#fff; }}
.q-cat {{ font-size:0.72em; color:var(--text-muted); flex-shrink:0; }}
.q-toggle {{ font-size:1.1em; color:var(--text-muted); transition:transform 0.2s; flex-shrink:0; }}
.q-card.collapsed .q-toggle {{ transform:rotate(-90deg); }}
.q-card.collapsed .q-body {{ display:none; }}
.q-body {{ padding:0 18px 14px; }}
.token-cloud {{ padding:8px 12px; display:flex; flex-wrap:wrap; gap:5px;
  border-bottom:1px solid var(--border); margin-bottom:8px; }}
.token-tag {{ display:inline-block; padding:1px 8px; border-radius:4px;
  font-size:0.78em; font-family:"SF Mono","Fira Code",monospace;
  transition:transform 0.1s; }}
.token-tag:hover {{ transform:scale(1.1); }}
.token-tag.query {{ background:rgba(255,255,255,0.08); color:#ccc;
  border:1px solid rgba(255,255,255,0.2); }}
.token-tag.matched {{ background:rgba(76,175,80,0.15); color:#81c784;
  border:1px solid rgba(76,175,80,0.3); }}
.token-label {{ font-size:0.7em; color:var(--text-muted); margin-right:4px; }}
.result-item {{ padding:10px 12px; border-bottom:1px solid var(--border);
  font-size:0.85em; transition:background 0.1s; }}
.result-item:last-child {{ border-bottom:none; }}
.result-item:hover {{ background:rgba(255,255,255,0.02); }}
.result-meta {{ display:flex; justify-content:space-between; align-items:center;
  margin-bottom:4px; }}
.result-rank {{ font-weight:700; font-size:1.05em; color:#fff; }}
.result-score {{ font-family:"SF Mono","Fira Code",monospace; font-size:0.85em;
  color:var(--text-muted); }}
.result-source {{ display:inline-block; padding:1px 8px; border-radius:4px;
  font-size:0.72em; margin-left:8px; }}
.result-source.loc {{ background:var(--loc-bg); color:var(--loc-accent);
  border:1px solid rgba(79,195,247,0.2); }}
.result-source.cont {{ background:var(--content-bg); color:var(--content-accent);
  border:1px solid rgba(171,71,188,0.2); }}
.result-ref {{ font-size:0.78em; color:var(--text-muted); margin-bottom:4px; }}
.result-content {{ color:#c0c4d0; line-height:1.8; }}
.result-content.truncated {{ max-height:5em; overflow:hidden; position:relative; }}
.result-content.truncated::after {{ content:""; position:absolute; bottom:0;
  left:0; right:0; height:24px;
  background:linear-gradient(transparent,var(--card-bg)); pointer-events:none; }}
.result-content.expanded {{ max-height:none !important; }}
.result-content.expanded::after {{ display:none; }}
.expand-btn {{ background:none; border:1px solid var(--border); color:var(--text-muted);
  padding:2px 10px; border-radius:4px; cursor:pointer; font-size:0.72em;
  margin-top:4px; }}
.expand-btn:hover {{ border-color:#555; color:#fff; }}
mark.hl-loc {{ background:var(--loc-hl); color:#fff; border-radius:2px; padding:0 1px; }}
mark.hl-cont {{ background:var(--content-hl); color:#fff; border-radius:2px; padding:0 1px; }}
.footer {{ text-align:center; color:var(--text-muted); font-size:0.75em;
  margin-top:30px; padding:10px; }}
@media (max-width:900px) {{ .stats-bar {{ gap:12px; }} }}
</style>
</head>
<body>
<h1>RAG Pipeline 检索报告</h1>
<div class="meta">{timestamp} | BM25 k1={k1} b={b} | backend={backend}</div>
<div class="stats-bar">
  <div class="stat"><div class="stat-value">{total_articles}</div><div class="stat-label">知识库条款</div></div>
  <div class="stat"><div class="stat-value">{num_questions}</div><div class="stat-label">测试问题</div></div>
  <div class="stat"><div class="stat-value">{top_k}</div><div class="stat-label">召回 Top-K</div></div>
  <div class="stat"><div class="stat-value">{seed}</div><div class="stat-label">随机种子</div></div>
</div>
<div id="container"></div>
<div class="footer">Generated by tools/report.py</div>
<script>
const DATA = {data_json};

function hlContent(text, matched, source) {{
  if (!matched || matched.length === 0) return escHtml(text);
  const cls = source === 'location' ? 'hl-loc' : 'hl-cont';
  let out = escHtml(text);
  // Replace longest matches first to avoid partial overlap
  const sorted = [...matched].sort((a, b) => b.length - a.length);
  for (const tok of sorted) {{
    const esc = escHtml(tok);
    const re = new RegExp(escRegex(esc), 'gi');
    out = out.replace(re, `<mark class="${{cls}}">${{esc}}</mark>`);
  }}
  return out;
}}
function escHtml(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}
function escRegex(s) {{ return s.replace(/[.*+?^${{}}()|[\]\\]/g,'\\$&'); }}

function renderResults(results) {{
  return results.map((r, i) => {{
    const srcCls = r.source === 'location' ? 'loc' : 'cont';
    const srcLabel = r.source === 'location' ? '定位' : '正文';
    let bodyHtml;
    if (r.source === 'location') {{
      // 定位索引：高亮定位文本，正文仅展示
      const locHtml = hlContent(r.loc_text, r.matched_tokens, r.source);
      bodyHtml = `
        <div class="result-loc-highlight" style="margin-bottom:6px;padding:4px 8px;background:var(--loc-bg);border-radius:4px;">
          <span style="font-size:0.72em;color:var(--loc-accent);">命中定位：</span>${{locHtml}}
        </div>
        <div class="result-content truncated" id="c-${{r._uid}}">${{escHtml(r.content)}}</div>
        <button class="expand-btn" onclick="toggleContent('c-${{r._uid}}', this)">展开正文</button>`;
    }} else {{
      // 正文索引：高亮正文内容
      const contentHtml = hlContent(r.content, r.matched_tokens, r.source);
      bodyHtml = `
        <div class="result-content truncated" id="c-${{r._uid}}">${{contentHtml}}</div>
        <button class="expand-btn" onclick="toggleContent('c-${{r._uid}}', this)">展开</button>`;
    }}
    return `
      <div class="result-item">
        <div class="result-meta">
          <span><span class="result-rank">#${{i+1}}</span>
          <span class="result-source ${{srcCls}}">${{srcLabel}}</span></span>
          <span class="result-score">score: ${{r.score.toFixed(4)}}</span>
        </div>
        ${{bodyHtml}}
      </div>`;
  }}).join('');
}}

function renderTokenCloud(queryTokens, matched) {{
  const matchedSet = new Set(matched);
  const tags = queryTokens.map(t => {{
    const cls = matchedSet.has(t) ? 'matched' : 'query';
    return `<span class="token-tag ${{cls}}">${{escHtml(t)}}</span>`;
  }}).join('');
  return `<div class="token-cloud"><span class="token-label">Query Tokens:</span>${{tags}}</div>`;
}}

let uid = 0;
const container = document.getElementById('container');
for (const item of DATA) {{
  uid++;
  const allMatched = new Set();
  item.results.forEach(r => r.matched_tokens.forEach(t => allMatched.add(t)));
  const resultsWithUid = item.results.map(r => {{ r._uid = uid + '_' + r._idx; return r; }});

  const card = document.createElement('div');
  card.className = 'q-card';
  card.innerHTML = `
    <div class="q-header" onclick="this.parentElement.classList.toggle('collapsed')">
      <span class="q-id">Q${{item.id}}</span>
      <span class="q-text">${{escHtml(item.question)}}</span>
      <span class="q-cat">${{escHtml(item.category_1)}} / ${{escHtml(item.category_2)}}</span>
      <span class="q-toggle">▼</span>
    </div>
    <div class="q-body">
      ${{renderTokenCloud(item.query_tokens, [...allMatched])}}
      ${{renderResults(resultsWithUid)}}
    </div>`;
  container.appendChild(card);
}}

function toggleContent(id, btn) {{
  const el = document.getElementById(id);
  el.classList.toggle('truncated');
  el.classList.toggle('expanded');
  btn.textContent = el.classList.contains('expanded') ? '收起' : '展开';
}}
</script>
</body>
</html>"""


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


def generate_report(num_questions: int = 10, top_k: int = 5, seed: int = 42) -> Path:
    docs_loc, docs_content, metadatas = parse_regulation(str(KB_PATH))
    retriever = Retriever(
        docs_loc, docs_content, metadatas,
        config={
            "bm25_k1": BM25_K1, "bm25_b": BM25_B,
            "bm25_backend": BM25_BACKEND, "bm25_recall_k": BM25_RECALL_K,
        },
    )

    all_questions = load_questions(str(QA_PATH))
    random.seed(seed)
    sampled = random.sample(all_questions, min(num_questions, len(all_questions)))

    report_data = []
    for i, item in enumerate(sampled, 1):
        query = item["question"]
        results = retriever.retrieve(query, top_k=top_k)
        query_tokens = tokenize_for_query(query)
        report_data.append({
            "id": i,
            "question": query,
            "category_1": item["category_1"],
            "category_2": item["category_2"],
            "query_tokens": query_tokens,
            "results": [
                {**r, "_idx": j,
                 "loc_text": f"{r['file_name']} {r['chapter']} {r['article_no']}"}
                for j, r in enumerate(results)
            ],
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html = HTML_TEMPLATE.format(
        timestamp=ts,
        k1=BM25_K1, b=BM25_B, backend=BM25_BACKEND,
        total_articles=len(metadatas),
        num_questions=num_questions,
        top_k=top_k,
        seed=seed,
        data_json=json.dumps(report_data, ensure_ascii=False),
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"bm25_result_{ts}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"报告已生成：{out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 检索报告生成器")
    parser.add_argument("--num-questions", type=int, default=10, help="抽样问题数")
    parser.add_argument("--top-k", type=int, default=5, help="每条问题召回数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    generate_report(args.num_questions, args.top_k, args.seed)
