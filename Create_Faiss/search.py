import os, json, requests, numpy as np, faiss

BASE_URL   = "http://10.43.107.28:10110"
INDEX_PATH = "data/sac.index"
META_PATH  = "data/meta.jsonl"
MODEL_NAME = "gl_embedding"

def get_embedding(text: str, model_name: str = "gl_embedding") -> np.ndarray:
    url = f"{BASE_URL}/embedding"
    payload = {"text": text, "model_name": model_name}
    r = requests.post(url, json=payload)
    emb = r.json()["result"]
    return np.asarray(emb, dtype="float32")

# 载入索引与侧表
index = faiss.read_index(INDEX_PATH)
id2meta = {}
with open(META_PATH, "r", encoding="utf-8") as f:
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

if __name__ == "__main__":
    query = "the material of the buffer in hemt"
    hits = search(query, k=5)
    print(hits)
    print(f"Query: {query}\nTop-{len(hits)} results:")
    for i, h in enumerate(hits, 1):
        snippet = h["text"].replace("\n", " ")
        print(f"[{i}] score={h['score']:.4f} file={h['doc_id']} chunk={h['chunk']}")
        print("    " + snippet)
