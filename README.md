# 法规 RAG Pipeline

基于 BM25 + Dense Retrieval + Reranker 的多阶段法规检索系统，支持法规文档检索和 QA 问答对匹配双路 RAG。

## 架构概览

```
Query
  ├── Query 编码（一次）：query_emb（共享给法规和 QA 的 Dense 检索）
  │
  ├── 法规 RAG：
  │     BM25 双路召回（定位索引 + 内容索引）
  │       + Dense 双路召回（定位向量 + 内容向量）
  │       → 4 路 RRF 融合 → reg_candidates (≤15)
  │
  ├── QA RAG：
  │     BM25 召回（问题索引）+ Dense 召回（问题向量）
  │       → 2 路 RRF 融合 → qa_candidates (≤5)
  │
  └── 合并 Rerank（单次推理）：
        法规内容 + QA 问题 拼成单个 batch → Reranker 精排
        → 法规组截断(top-5) + QA组截断阈值过滤(top-3, ≥0.3)
```

## 目录结构

```
app/                         # 项目根目录
├── config.py                # 全局配置（模型路径、超参数）
├── index.py                 # 索引构建入口（支持 --qa / --reg 参数）
├── main.py                  # 交互式 Pipeline 入口
├── server.py                # FastAPI 服务入口
├── data/
│   ├── knowledge_base/      # 法规 JSON 文件（知识库，源头输入）
│   └── question.json        # 测试问题集
├── artifacts/               # 衍生产物（可重建，建议 gitignore）
│   ├── index/               # 法规 RAG 索引
│   │   ├── bm25/
│   │   ├── dense/
│   │   └── metadatas.json
│   └── qa_index/            # QA RAG 索引
│       ├── bm25/
│       ├── dense/
│       └── qa_data.json
├── retrieval/               # 召回模块
│   ├── bm25.py              # BM25 双路检索器（法规）
│   ├── dense.py             # Dense 向量检索器（法规）
│   ├── retrieve.py          # 法规统一检索入口（RRF 融合）
│   └── qa_retrieve.py       # QA 检索器（单问题索引 + 2 路 RRF）
├── rerank/
│   └── reranker.py          # Reranker 实现（BGE / GTE / Qwen3）
├── utils/
│   ├── doc_parser.py        # 法规 JSON 解析器
│   ├── qa_data_loader.py    # QA 数据加载器（从 Excel 读取）
│   ├── chunker.py           # 智能文档切分
│   └── tokenizer.py         # 分词工具（jieba + 停用词）
├── scripts/
│   └── report_tool/         # 独立运行的报告生成脚本
└── tests/                   # 测试套件
```

## 环境配置

### 1. 创建 Conda 环境

```bash
conda env create -f conda_envs/environment.yml
```

（`conda_envs/` 位于本项目上级目录。）

### 2. 模型准备

需要以下模型文件（路径可在 `config.py` 中修改）：

| 用途 | 默认路径 | 模型 |
|------|---------|------|
| Embedding | `/home/moga/models/embedding/Qwen3-Embedding-0.6B` | Qwen3-Embedding-0.6B |
| Reranker | `/home/moga/models/reranker/Qwen3-Reranker-0.6B` | Qwen3-Reranker-0.6B |

## 运行

首先确保自己已经在项目根目录下

### 构建索引

首次使用前需要先构建索引（法规索引约 1 分钟，QA 索引约 3 秒）：

```bash
conda activate retrieval
python index.py            # 构建所有索引
python index.py --reg      # 仅构建法规索引
python index.py --qa       # 仅构建 QA 索引
```

索引分别保存在 `artifacts/index/` 和 `artifacts/qa_index/` 下。如需重建，删除对应目录后重新运行即可。

### 交互式查询

```bash
conda run -n retrieval python main.py
```

启动时从磁盘加载已有索引和模型（约 30 秒）。如索引不存在，会提示先运行 `python index.py`。

### API 服务

启动 HTTP 服务供外部调用：

```bash
conda activate retrieval
python server.py
```

服务启动时从磁盘加载索引和模型（约 30 秒），就绪后监听 `0.0.0.0:18400`（端口可在 `config.py` 的 `SERVER_PORT` 中修改）。如索引不存在，`/query` 返回 503，需要先建立索引。

**端点：**

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查，返回 `{"status": "ready", "reranker_model": "qwen3"}` |
| `POST` | `/query` | 查询接口（含法规 RAG + QA RAG），请求体为 JSON |

**请求参数：**

```json
{
  "query": "查询文本（必填）",
  "top_k": 5,
  "debug": false
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 查询文本 |
| `top_k` | int | 否 | 法规精排返回条数，默认取 `config.py` 中的 `RERANK_TOP_K` |
| `debug` | bool | 否 | `true` 时返回召回数量、RRF 配置等中间信息 |

**响应示例：**

```json
{
  "query": "安全生产责任",
  "count": 5,
  "results": [
    {
      "rerank_score": 0.993,
      "rrf_score": 0.031,
      "source": "content",
      "matched_tokens": {"location": ["安全", "责任"], "content": ["安全", "生产"]},
      "file_name": "安徽省实施《中华人民共和国道路交通安全法》办法(修正)",
      "chapter": "第二章　交通安全管理责任",
      "article_no": "第六条",
      "content": "县级以上人民政府应当建立并落实..."
    }
  ],
  "qa_results": [
    {
      "qa_id": 0,
      "matched_question": "盲道被占用停车，盲人怎么走？能举报吗？",
      "answer": "盲道被占，盲人可向交警或拨打12345举报...",
      "rerank_score": 0.85
    }
  ],
  "elapsed_ms": 1460.3,
  "debug": null
}
```

`qa_results` 字段说明：
- 当 QA 匹配的 rerank 分数低于 `QA_SIMILARITY_THRESHOLD`（默认 0.3）时，该字段为 `null`
- 最多返回 `QA_RERANK_TOP_K`（默认 3）条匹配结果

`debug=true` 时额外返回：

```json
{
  "debug": {
    "recall_count": 15,
    "qa_recall_count": 5,
    "rrf_top_k": 15,
    "qa_rrf_top_k": 5,
    "rerank_model": "qwen3",
    "rrf_method": "4way",
    "qa_similarity_threshold": 0.3
  }
}
```

**调用示例：**

```bash
# 简洁模式
curl -X POST http://<IP>:18400/query \
  -H "Content-Type: application/json" \
  -d '{"query": "安全生产责任"}'

# 自定义 top_k + debug 模式
curl -X POST http://<IP>:18400/query \
  -H "Content-Type: application/json" \
  -d '{"query": "安全生产责任", "top_k": 3, "debug": true}'
```

### 运行测试

```bash
conda run -n retrieval python -m pytest tests/ -v
```

### 生成报告

```bash
conda run -n retrieval python -m scripts.report_tool.rerank_compare
conda run -n retrieval python -m scripts.report_tool.dense_compare
conda run -n retrieval python -m scripts.report_tool.rrf_compare
conda run -n retrieval python -m scripts.report_tool.system_profile
```

报告输出到 `doc/reports/`。

## 配置说明

所有配置项在 `config.py` 中，按模块分组。

### BM25 召回（法规）

| 配置项 | 默认值 | 说明 |
|--------|-------|------|
| `BM25_K1` / `BM25_B` | 1.5 / 0.75 | BM25 词频/文档长度超参数 |
| `BM25_BACKEND` | `"numpy"` | BM25 计算后端 |
| `BM25_RECALL_K` | 20 | BM25 每路（定位/内容）最大召回条数 |

### 索引构建

| 配置项 | 默认值 | 说明 |
|--------|-------|------|
| `INDEX_DEVICE` | `"cuda"` | 构建索引时 Embedding 模型使用的设备 |
| `INDEX_BATCH_SIZE` | `4` | 构建索引时编码批大小；内容文本序列较长，过大会 OOM |

### Dense 召回（法规）

| 配置项 | 默认值 | 说明 |
|--------|-------|------|
| `DENSE_MODEL_PATH` | `/home/moga/models/embedding/Qwen3-Embedding-0.6B` | Embedding 模型路径 |
| `DENSE_DEVICE` | `"cuda"` | 检索时 query 编码设备，可选 `"cpu"` 但速度会显著下降 |
| `DENSE_BATCH_SIZE` | 4 | 检索时编码批大小 |
| `DENSE_RECALL_K` | 20 | Dense 每路（定位/内容）最大召回条数 |

### RRF 融合（法规）

| 配置项 | 默认值 | 说明 |
|--------|-------|------|
| `RRF_METHOD` | `"4way"` | 融合模式：`"4way"` 四路直接融合；`"2way"` 两路分组融合 |
| `RRF_K` | 60 | RRF 平滑常数（越大排名差异越平滑） |
| `RRF_TOP_K` | 15 | RRF 融合后截断条数，即送入 Reranker 的法规候选数量 |
| `RRF_2WAY_AXIS` | `"by_index"` | 仅 `RRF_METHOD="2way"` 时生效 |

### Reranker 精排

| 配置项 | 默认值 | 说明 |
|--------|-------|------|
| `RERANKER_MODEL` | `"qwen3"` | 模型选择：`"qwen3"`（区分度最强）、`"bge"`（备选） |
| `RERANKER_MODEL_PATH` | `/home/moga/models/reranker/Qwen3-Reranker-0.6B` | 模型路径 |
| `RERANKER_DEVICE` | `"cuda"` | 推理设备 |
| `RERANKER_BATCH_SIZE` | 20 | 精排批大小；需覆盖 RRF_TOP_K + QA_RRF_TOP_K |
| `RERANK_TOP_K` | 5 | 法规精排后输出条数 |

### QA RAG

| 配置项 | 默认值 | 说明 |
|--------|-------|------|
| `QA_DATA_PATH` | `data/qa_knowledgebase_sample.xlsx` | QA 知识库 Excel 路径 |
| `QA_INDEX_DIR` | `artifacts/qa_index/` | QA 索引目录 |
| `QA_BM25_K1` / `QA_BM25_B` | 1.5 / 0.75 | QA BM25 超参数 |
| `QA_BM25_RECALL_K` | 10 | QA BM25 最大召回条数 |
| `QA_DENSE_RECALL_K` | 10 | QA Dense 最大召回条数 |
| `QA_RRF_K` | 60 | QA RRF 平滑常数 |
| `QA_RRF_TOP_K` | 5 | QA RRF 截断条数（送 Reranker 的候选数） |
| `QA_RERANK_TOP_K` | 3 | QA Rerank 后返回给下游的最多条数 |
| `QA_SIMILARITY_THRESHOLD` | 0.3 | rerank 低于此阈值的 QA 结果被丢弃；设为 `float('-inf')` 则始终返回 |

约束：`RRF_TOP_K(15) + QA_RRF_TOP_K(5) ≤ RERANKER_BATCH_SIZE(20)`

### API 服务

| 配置项 | 默认值 | 说明 |
|--------|-------|------|
| `SERVER_HOST` | `"0.0.0.0"` | 监听地址 |
| `SERVER_PORT` | `18400` | 监听端口 |
