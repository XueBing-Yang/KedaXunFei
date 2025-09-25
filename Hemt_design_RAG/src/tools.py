import re
from typing import List
import json
import numpy as np
import requests
from src.hemt_config import BASE_URL, EMBEDDING_INDEX_PATH, EMBEDDING_META_PATH
import faiss

def extract_json_list(text: str) -> List[str]:
    """尽量稳健地从 LLM 输出中提取 JSON 列表"""
    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if not m:
        m = re.search(r"(\[[\s\S]*?\])", text)
    if m:
        try:
            arr = json.loads(m.group(1))
            if isinstance(arr, list):
                return arr
        except Exception:
            pass
    return []

def strip_refs(s: str) -> str:
    return re.sub(r"\[\^\d+\^\]", "", s).strip()

def get_embedding(text: str, model_name: str = "gl_embedding") -> np.ndarray:
    url = f"{BASE_URL}/embedding"
    payload = {"text": text, "model_name": model_name}
    r = requests.post(url, json=payload)
    emb = r.json()["result"]
    return np.asarray(emb, dtype="float32")


index = faiss.read_index(EMBEDDING_INDEX_PATH)
id2meta = {}
with open(EMBEDDING_META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        m = json.loads(line)
        id2meta[int(m["vid"])] = m

def search(query: str, k: int = 5):
    q = get_embedding(query).reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)                 # 和建库时一致：余弦相似
    D, I = index.search(q, k)
    results = []
    for score, vid in zip(D[0], I[0]):
        if vid == -1: continue
        m = id2meta.get(int(vid))
        if not m:   continue
        results.append({
            "score": float(score),
            "doc_id":  m["doc_id"],
            "chunk": m["chunk_id"],
            "text":  m["text"],
        })
    return results