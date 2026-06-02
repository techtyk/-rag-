"""Sheet5 检索效果评估脚本

遍历 question.json 的 Sheet5（720条），对每条问题跑完整 pipeline（BM25 + Dense + RRF + Reranker），
提取 Top 5 结果生成 HTML 报告。

为避免 GPU 显存不足，分两个子进程执行：
  Phase 1 (--phase1): 多路召回 + RRF 融合，结果保存到临时 JSON
  Phase 2 (--phase2): 加载 JSON → Reranker 精排 → 生成 HTML

直接运行（不加参数）自动依次执行两个阶段。

用法:
    cd /home/moga/project/dense_training/app
    conda run -n retrieval python -m scripts.retrieval_eval
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

REPORTS_DIR = APP_DIR.parent / "doc" / "reports"
TEMP_RESULTS = APP_DIR / "artifacts" / "retrieval_eval_temp.json"

_FIELDS_TO_KEEP = ["score", "scores", "source", "matched_tokens",
                   "file_name", "chapter", "article_no", "content"]


# ======================================================================
#  Phase 1: 检索
# ======================================================================

def run_phase1():
    from config import (INDEX_DIR, BM25_K1, BM25_B, BM25_BACKEND, BM25_RECALL_K,
                        DENSE_MODEL_PATH, DENSE_DEVICE, DENSE_BATCH_SIZE,
                        DENSE_RECALL_K, RRF_METHOD, RRF_K, RRF_TOP_K, RRF_2WAY_AXIS)
    from retrieval.retrieve import Retriever, _check_index_complete

    config = {
        "index_dir": str(INDEX_DIR),
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "bm25_backend": BM25_BACKEND,
        "bm25_recall_k": BM25_RECALL_K,
        "dense_model_path": DENSE_MODEL_PATH,
        "dense_device": DENSE_DEVICE,
        "dense_batch_size": DENSE_BATCH_SIZE,
        "dense_recall_k": DENSE_RECALL_K,
        "rrf_method": RRF_METHOD,
        "rrf_k": RRF_K,
        "rrf_top_k": RRF_TOP_K,
        "rrf_2way_axis": RRF_2WAY_AXIS,
    }

    if not _check_index_complete(config["index_dir"]):
        print("索引未找到，请先运行 main.py 构建索引。")
        sys.exit(1)

    retriever = Retriever.load(config["index_dir"], config)

    from config import QA_PATH
    with open(QA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("Sheet5", [])
    print(f"加载 {len(questions)} 条问题")

    results = []
    total = len(questions)
    phase1_start = time.time()

    for i, item in enumerate(questions):
        t0 = time.time()
        candidates = retriever.retrieve(item["question"])
        retrieval_ms = (time.time() - t0) * 1000

        trimmed = [{k: c[k] for k in _FIELDS_TO_KEEP if k in c} for c in candidates]
        results.append({
            "question": item["question"],
            "cat1": item.get("一级类目", ""),
            "cat2": item.get("二级类目", ""),
            "cat3": item.get("三级类目", ""),
            "candidates": trimmed,
            "retrieval_ms": round(retrieval_ms, 1),
        })
        if (i + 1) % 50 == 0 or i == total - 1:
            elapsed = time.time() - phase1_start
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  检索进度 {i + 1}/{total}  耗时 {elapsed:.1f}s  剩余 ~{eta:.0f}s")

    phase1_time = time.time() - phase1_start

    TEMP_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(TEMP_RESULTS, "w", encoding="utf-8") as f:
        json.dump({"phase1_time": round(phase1_time, 1), "results": results},
                  f, ensure_ascii=False)
    print(f"检索结果已保存：{TEMP_RESULTS}（总耗时 {phase1_time:.1f}s）")


# ======================================================================
#  Phase 2: 精排 + 报告生成
# ======================================================================

def run_phase2():
    from config import (RERANKER_MODEL, RERANKER_MODEL_PATH, RERANKER_DEVICE,
                        RERANK_TOP_K)

    with open(TEMP_RESULTS, "r", encoding="utf-8") as f:
        payload = json.load(f)
    phase1_time = payload["phase1_time"]
    results = payload["results"]

    total = len(results)
    print(f"加载 {total} 条检索结果")

    from rerank.reranker import RERANKER_REGISTRY
    reranker_cls = RERANKER_REGISTRY[RERANKER_MODEL]
    reranker = reranker_cls(RERANKER_MODEL_PATH, device=RERANKER_DEVICE)

    phase2_start = time.time()
    for i, r in enumerate(results):
        candidates = r["candidates"]
        if not candidates:
            r["top5"] = []
            r["rerank_ms"] = 0
            continue

        documents = [c["content"] for c in candidates]
        t0 = time.time()
        rerank_scores = reranker.rerank(r["question"], documents, batch_size=2)
        r["rerank_ms"] = round((time.time() - t0) * 1000, 1)

        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = s
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        r["top5"] = candidates[:RERANK_TOP_K]
        del r["candidates"]

        if (i + 1) % 50 == 0 or i == total - 1:
            elapsed = time.time() - phase2_start
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  精排进度 {i + 1}/{total}  耗时 {elapsed:.1f}s  剩余 ~{eta:.0f}s")

    phase2_time = time.time() - phase2_start
    total_time = phase1_time + phase2_time

    # 统计
    all_scores = [r["top5"][0]["rerank_score"] for r in results if r["top5"]]
    print(f"\nTop1 Rerank 分值：min={min(all_scores):.4f}  max={max(all_scores):.4f}  "
          f"avg={sum(all_scores)/len(all_scores):.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"retrieval_eval_sheet5_{timestamp}.html"

    html = generate_html(results, phase1_time, phase2_time, total_time)
    report_path.write_text(html, encoding="utf-8")
    print(f"报告已生成：{report_path}")

    TEMP_RESULTS.unlink(missing_ok=True)


# ======================================================================
#  HTML 生成
# ======================================================================

def generate_html(results, phase1_time, phase2_time, total_time):
    from config import BM25_RECALL_K, DENSE_RECALL_K, RRF_TOP_K, RERANK_TOP_K

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)

    all_scores = [r["top5"][0]["rerank_score"] for r in results if r["top5"]]
    score_min = min(all_scores) if all_scores else 0
    score_max = max(all_scores) if all_scores else 0
    score_avg = sum(all_scores) / len(all_scores) if all_scores else 0

    # 逐 query 延时统计
    retrieval_times = [r["retrieval_ms"] for r in results]
    rerank_times = [r["rerank_ms"] for r in results]
    avg_retrieval = sum(retrieval_times) / len(retrieval_times)
    avg_rerank = sum(rerank_times) / len(rerank_times)
    avg_total = avg_retrieval + avg_rerank

    rows_html = ""
    for idx, item in enumerate(results, 1):
        question = item["question"]
        category = f"{item['cat1']} / {item['cat2']} / {item['cat3']}"
        top5 = item["top5"]
        q_total = item["retrieval_ms"] + item["rerank_ms"]
        timing = (f"检索 {item['retrieval_ms']:.0f}ms + "
                  f"精排 {item['rerank_ms']:.0f}ms = "
                  f"{q_total:.0f}ms")

        rows_html += (
            f'\n        <tr class="question-row" id="q{idx}">'
            f'\n            <td rowspan="{max(len(top5), 1)}">{idx}</td>'
            f'\n            <td rowspan="{max(len(top5), 1)}" class="question-cell">'
            f'\n                <span class="category">{category}</span><br>'
            f'\n                <span class="question-text">{question}</span><br>'
            f'\n                <span class="timing">{timing}</span>'
            f'\n            </td>'
        )

        if not top5:
            rows_html += ('\n            <td colspan="7" class="no-result">'
                          '未召回任何结果</td>\n        </tr>')
            continue

        for rank, r in enumerate(top5):
            rerank_s = r["rerank_score"]
            rrf_s = r.get("score", 0)
            score_class = ("score-high" if rerank_s >= 0.8
                           else "score-mid" if rerank_s >= 0.5
                           else "score-low")
            content = r.get("content", "")
            content_short = content[:150] + ("..." if len(content) > 150 else "")

            cols = (
                f'\n            <td>{rank + 1}</td>'
                f'\n            <td class="{score_class}">{rerank_s:.4f}</td>'
                f'\n            <td>{rrf_s:.6f}</td>'
                f'\n            <td>{r.get("source", "")}</td>'
                f'\n            <td>{r.get("file_name", "")} · {r.get("chapter", "")} · {r.get("article_no", "")}</td>'
                f'\n            <td class="content-cell" title="{content}">{content_short}</td>'
                f'\n            <td>{r.get("matched_tokens", {})}</td>'
            )

            if rank == 0:
                rows_html += cols + "\n        </tr>"
            else:
                rows_html += f'\n        <tr class="result-sub-row">{cols}\n        </tr>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Sheet5 检索效果评估报告</title>
<style>
    body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; margin: 20px; background: #f5f5f5; }}
    h1 {{ color: #333; }}
    .summary {{ background: #fff; padding: 15px 20px; border-radius: 8px; margin-bottom: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .summary span {{ margin-right: 30px; font-size: 14px; color: #555; }}
    .summary strong {{ color: #222; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 13px; }}
    th {{ background: #4a5568; color: #fff; padding: 10px 8px; text-align: left;
         position: sticky; top: 0; z-index: 10; }}
    td {{ padding: 8px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }}
    .question-row td {{ border-bottom: 2px solid #cbd5e0; background: #fafafa; }}
    .result-sub-row td {{ background: #fff; }}
    .question-cell {{ min-width: 200px; }}
    .category {{ font-size: 11px; color: #888; }}
    .question-text {{ font-weight: bold; color: #2d3748; }}
    .timing {{ font-size: 11px; color: #718096; font-family: monospace; }}
    .score-high {{ color: #38a169; font-weight: bold; }}
    .score-mid {{ color: #d69e2e; font-weight: bold; }}
    .score-low {{ color: #e53e3e; font-weight: bold; }}
    .content-cell {{ max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .no-result {{ color: #e53e3e; text-align: center; padding: 15px; }}
    tr:hover {{ background: #ebf8ff !important; }}
</style>
</head>
<body>
<h1>Sheet5 检索效果评估报告</h1>
<div class="summary">
    <span>生成时间：<strong>{now}</strong></span><br>
    <span>总问题数：<strong>{total}</strong></span>
    <span>检索总耗时：<strong>{phase1_time:.1f}s</strong></span>
    <span>精排总耗时：<strong>{phase2_time:.1f}s</strong></span>
    <span>总耗时：<strong>{total_time:.1f}s</strong></span><br>
    <span>平均检索：<strong>{avg_retrieval:.0f}ms/query</strong></span>
    <span>平均精排：<strong>{avg_rerank:.0f}ms/query</strong></span>
    <span>平均端到端：<strong>{avg_total:.0f}ms/query</strong></span><br>
    <span>Top1 Rerank 分值范围：<strong>{score_min:.4f} ~ {score_max:.4f}</strong></span>
    <span>Top1 Rerank 均值：<strong>{score_avg:.4f}</strong></span><br>
    <span>召回参数：<strong>BM25×{BM25_RECALL_K} + Dense×{DENSE_RECALL_K} → RRF截断{RRF_TOP_K} → Reranker精排{RERANK_TOP_K}</strong></span>
</div>
<table>
<thead>
<tr>
    <th>#</th>
    <th>问题</th>
    <th>排名</th>
    <th>Rerank</th>
    <th>RRF</th>
    <th>来源</th>
    <th>出处</th>
    <th>内容</th>
    <th>命中词元</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</body>
</html>"""


# ======================================================================
#  入口：自动编排两阶段子进程
# ======================================================================

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--phase1":
        run_phase1()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--phase2":
        run_phase2()
        return

    print("=" * 60)
    print("Sheet5 检索效果评估（两阶段子进程执行）")
    print("=" * 60)

    env = os.environ.copy()
    env["TQDM_DISABLE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    python = sys.executable

    # Phase 1
    print("\n--- Phase 1: 多路召回 + RRF 融合 ---")
    t0 = time.time()
    r1 = subprocess.run(
        [python, "-m", "scripts.retrieval_eval", "--phase1"],
        env=env, cwd=str(APP_DIR),
    )
    if r1.returncode != 0:
        print("Phase 1 失败")
        sys.exit(1)
    phase1_time = time.time() - t0
    print(f"Phase 1 完成（{phase1_time:.1f}s）")

    # Phase 2
    print("\n--- Phase 2: Reranker 精排 + 报告生成 ---")
    t1 = time.time()
    r2 = subprocess.run(
        [python, "-m", "scripts.retrieval_eval", "--phase2"],
        env=env, cwd=str(APP_DIR),
    )
    if r2.returncode != 0:
        print("Phase 2 失败")
        sys.exit(1)
    phase2_time = time.time() - t1
    total_time = time.time() - t0
    print(f"\nPhase 2 完成（{phase2_time:.1f}s）")
    print(f"总耗时：{total_time:.1f}s")


if __name__ == "__main__":
    main()
