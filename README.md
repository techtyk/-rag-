# 法规 RAG Pipeline

基于 BM25 + Dense Retrieval + Reranker 的多阶段法规检索系统。

## 架构概览

```
Query
  ↓
BM25 双路召回（定位索引 + 内容索引）
  ↓
Dense 双路召回（定位向量 + 内容向量）
  ↓
RRF 融合（四路排序结果合并）
  ↓
Reranker 精排（Qwen3-Reranker）
  ↓
最终 Top-K 结果
```

## 目录结构

```
app/
├── config.py              # 全局配置（模型路径、超参数）
├── main.py                # 交互式 Pipeline 入口
├── data/
│   ├── knowledge_base/    # 法规 JSON 文件（知识库，源头输入）
│   └── question.json      # 测试问题集
├── artifacts/             # 衍生产物（可重建，建议 gitignore）
│   └── index/
│       ├── bm25/          # BM25 索引 + 分词结果
│       ├── dense/         # FAISS 向量索引
│       └── metadatas.json # 文档元数据
├── retrieval/             # 召回模块
│   ├── bm25.py            # BM25 双路检索器
│   ├── dense.py           # Dense 向量检索器
│   └── retrieve.py        # 统一检索入口（RRF 融合）
├── rerank/                # 重排模块
│   └── reranker.py        # Reranker 实现（BGE / GTE / Qwen3）
├── utils/
│   ├── doc_parser.py      # 法规 JSON 解析器
│   └── tokenizer.py       # 分词工具（jieba + 停用词）
├── scripts/
│   └── report_tool/       # 独立运行的报告生成脚本
└── tests/                 # 测试套件
```

## 环境配置

### 1. 创建 Conda 环境

```bash
cd /home/moga/project/dense_training
conda env create -f conda_envs/environment.yml
```

### 2. 模型准备

需要以下模型文件（路径可在 `config.py` 中修改）：

| 用途 | 默认路径 | 模型 |
|------|---------|------|
| Embedding | `/home/moga/models/embedding/Qwen3-Embedding-0.6B` | Qwen3-Embedding-0.6B |
| Reranker | `/home/moga/models/reranker/Qwen3-Reranker-0.6B` | Qwen3-Reranker-0.6B |

## 运行

### 交互式查询

```bash
cd /home/moga/project/dense_training/app
conda run -n retrieval python main.py
```

首次运行会解析文档并构建索引（约 5 分钟），后续启动直接加载已有索引（约 30 秒）。

### 运行测试

```bash
cd /home/moga/project/dense_training/app
conda run -n retrieval python -m pytest tests/ -v
```

### 生成报告

```bash
# Reranker 模型对比
conda run -n retrieval python -m scripts.report_tool.rerank_compare

# Dense 模型对比
conda run -n retrieval python -m scripts.report_tool.dense_compare

# RRF 融合策略对比
conda run -n retrieval python -m scripts.report_tool.rrf_compare

# 系统性能与架构报告
conda run -n retrieval python -m scripts.report_tool.system_profile
```

报告输出到 `/home/moga/project/dense_training/doc/reports/`。

## 配置说明

所有配置项在 `config.py` 中，主要参数：

| 配置项 | 默认值 | 说明 |
|--------|-------|------|
| `BM25_K1` / `BM25_B` | 1.5 / 0.75 | BM25 超参数 |
| `BM25_RECALL_K` | 10 | BM25 每路召回数 |
| `DENSE_RECALL_K` | 10 | Dense 每路召回数 |
| `RRF_METHOD` | `"4way"` | RRF 融合模式（`"4way"` 或 `"2way"`） |
| `RRF_K` | 60 | RRF 平滑常数 |
| `RERANKER_MODEL` | `"qwen3"` | Reranker 模型（`"qwen3"` 或 `"bge"`） |
| `RERANK_TOP_K` | 5 | 最终输出条数 |
