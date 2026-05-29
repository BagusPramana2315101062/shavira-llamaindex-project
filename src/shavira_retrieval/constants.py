"""Konstanta default untuk eksperimen retrieval SHAVIRA."""

DEFAULT_JSONL_FILES = [
    "jdih_undiksha_ac_id.jsonl",
    "upttik_undiksha_ac_id.jsonl",
    "undiksha_ac_id_pmb.jsonl",
    "undiksha_ac_id_tentang_undiksha.jsonl",
]

DEFAULT_VALIDATION_FILE = "Dataset_Validasi_Retrieval_SHAVIRA.xlsx"

DEFAULT_EVAL_K = [5, 10]
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_CANDIDATE_K = 50
DEFAULT_RRF_K = 10
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_EMBED_MAX_LENGTH = 512
DEFAULT_EMBED_BATCH_SIZE = 8
DEFAULT_BM25_MODE = "default"
DEFAULT_INDEX_CACHE_DIR = "storage/faiss_index_cache"

REQUIRED_VALIDATION_COLUMNS = {"ID", "Context", "Question", "Answer"}
RETRIEVAL_METHODS = ["BM25", "BGE_M3_FAISS", "HYBRID_RRF"]
