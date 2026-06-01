# 检索模块未来规划
本文档是[ROADMAP](../ROADMAP.md)中的任务2的ROADMAP

## 整体架构

```
Query
  ↓
BM25 双路召回（定位索引 + 内容索引，各 top K）
  ↓
合并去重（≤ 2K 条候选）
  ↓
Dense Retrieval 召回（预留）
  ↓
RRF 融合（BM25 + Dense 多路结果合并排序）
  ↓
Reranker 精排
  ↓
最终 top K 结果
```

## 1. BM25 双路召回（已完成）

在 Retrieval 阶段，对定位索引和内容索引分别召回 top K 条结果，合并去重后送入下游。

- **定位索引**：`"文件名 章节 条款号"` 格式的文本，分词后可用于按法规结构定位
- **内容索引**：条款正文文本，用于按语义内容检索

两路各召回 K 条（而非 K/2），充分召回后合并去重，避免因截断丢失候选。最终由 Reranker 统一排序，无需在 Retrieval 阶段做加权融合或意图判断。

## 2. Dense Retrieval + RRF 融合（已完成）

引入向量检索，与 BM25 形成互补：
- BM25 擅长精确关键词匹配
- Dense 擅长语义相似度匹配

使用 RRF（Reciprocal Rank Fusion）将 BM25 和 Dense 的排序结果融合，取长补短。

## 3. 中期：Reranker 精排（已完成）

### step3.1（已完成）
索引持久化：BM25（bm25s save/load）+ Dense（faiss write/read）+ metadatas（JSON），启动时检测已有索引直接加载。索引存放在 `artifacts/index/` 下，按组件分为 `bm25/` 和 `dense/` 子目录。
### step3.2（已完成）
在融合结果之上加一层精排模型（Qwen3-Reranker-0.6B），对候选集做二次打分，输出最终排序。

Reranker 的排序能力替代了原本在 Retrieval 阶段做分数加权的需求——让专业的层做专业的事。

## 4. 长期：端侧部署与优化

- 模型量化（如 ONNX / TensorRT）以适应端侧算力
- 查询缓存与批量推理优化
