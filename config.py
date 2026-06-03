from pathlib import Path

# ==================== 路径配置 ====================
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
KB_PATH = DATA_DIR / "knowledge_base"
QA_PATH = DATA_DIR / "question.json"
INDEX_DIR = APP_DIR / "artifacts" / "index"

# ==================== 索引构建配置 ====================
INDEX_DEVICE = "cuda"
INDEX_BATCH_SIZE = 4     # 构建索引时无 Reranker 竞争显存；内容文本序列较长，过大会 OOM

# ==================== 文档切分配置 ====================
CHUNK_SIZE = 512        # 触发切分的长度阈值（字符数）；仅超限文档进入切分管线
CHUNK_OVERLAP = 50      # 低语义分隔符（" " 或字符截断）切分时的重叠字符数

# ==================== BM25 配置 ====================
BM25_K1 = 1.5
BM25_B = 0.75
BM25_BACKEND = "numpy"
BM25_TOP_K = 5

# 双路召回：定位索引和内容索引各自最多召回 top K 条
BM25_RECALL_K = 20

# ==================== Dense Retrieval ====================
DENSE_RECALL_K = 20
DENSE_MODEL_PATH = "/home/moga/models/embedding/Qwen3-Embedding-0.6B"
DENSE_DEVICE = "cuda"    # 检索时 query 编码设备，可选 "cpu" 但速度会显著下降（~15x）
DENSE_BATCH_SIZE = 4

# ==================== RRF 融合 ====================
# RRF_METHOD: 融合模式
#   "4way"  — 四路 RRF：BM25定位、BM25内容、Dense定位、Dense内容，直接做 RRF
#   "2way"  — 两路分组 RRF，分组方向由 RRF_2WAY_AXIS 决定
RRF_METHOD = "4way"
RRF_K = 60
RRF_TOP_K = 10 # RRF 融合然后截断输出给Reranker的最终条数，默认为15，过大可能增加精排负担，过小可能丢失有效候选

# RRF_2WAY_AXIS: 2-way 模式的融合轴（仅当 RRF_METHOD = "2way" 时生效）
#   "by_index"     — 按索引类型分组：定位组(BM25+Dense) RRF → 前5, 内容组(BM25+Dense) RRF → 前5
#   "by_retriever" — 按检索器分组：BM25组(定位+正文) RRF → 前5, Dense组(定位+正文) RRF → 前5
RRF_2WAY_AXIS = "by_index"

# ==================== Reranker ====================
# 模型名称，对应 rerank/reranker.py 中的 RERANKER_REGISTRY
# 可选: "qwen3"（默认，区分度最强）, "bge"（备选，BPU 部署更友好）
RERANKER_MODEL = "qwen3"
RERANKER_MODEL_PATH = "/home/moga/models/reranker/Qwen3-Reranker-0.6B"
# 备选模型路径: /home/moga/models/reranker/bge-reranker-v2-m3
RERANKER_DEVICE = "cuda"
RERANKER_BATCH_SIZE = 15  # 精排批大小；切分后文档 ≤512 字符，padding 开销大幅降低；batch=15 匹配 RRF_TOP_K，一次批处理完成精排；显存占用 reserved=8.7GB/11.9GB (73%)

# Reranker 精排后最终输出条数
RERANK_TOP_K = 5 # 可以等于RERANKER_BATCH_SIZE，由下游决定保留哪些chunk注入LLM

# ==================== API 服务 ====================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 18400
