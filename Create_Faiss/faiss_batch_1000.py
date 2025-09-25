from cgitb import text
import os
import requests
import json
import time
import statistics
import re
from pathlib import Path
import faiss, numpy as np
from tqdm import tqdm

BASE_URL = "http://10.43.107.28:10110"
TEST_TIMES = 50
BATCH_FILES = 1000  # 每处理1000个文件，持久化一次

def get_embedding(text: str, model_name: str = "gl_embedding") -> np.ndarray:
    url = f"{BASE_URL}/embedding"
    payload = {"text": text, "model_name": model_name}
    r = requests.post(url, json=payload)
    r.raise_for_status()
    emb = r.json()["result"]
    return np.asarray(emb, dtype="float32")

def test_embedding():
    url = f"{BASE_URL}/embedding"
    test_cases = [
        {"text": "深度学习框架", "model_name": "gl_embedding"},
    ]

    for case in test_cases:
        durations = []
        for _ in range(TEST_TIMES):
            start_time = time.time()
            response = requests.post(url, json=case)
            end_time = time.time()
            durations.append((end_time - start_time)*1000)

        avg_time = statistics.mean(durations)
        std_dev = statistics.stdev(durations) if len(durations) > 1 else 0

        print(f"Embedding Test - Model: {case['model_name']}")
        print(f"Average Time: {avg_time:.2f}ms ± {std_dev:.2f}ms")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}\n")

def _flush_batch(index, INDEX_PATH, batch_vecs, batch_ids, batch_metas, meta_fh):
    """把当前批次的向量与元数据落盘；批次按文件数触发。"""
    if not batch_vecs:
        return 0
    Vb = np.vstack(batch_vecs).astype("float32")
    faiss.normalize_L2(Vb)  # 余弦相似：先归一化，用内积
    ids = np.asarray(batch_ids, dtype="int64")
    index.add_with_ids(Vb, ids)

    # 覆盖保存索引
    faiss.write_index(index, INDEX_PATH)

    # 追加写入 meta（jsonl）
    for meta in batch_metas:
        meta_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")

    return len(batch_vecs)

if __name__ == "__main__":
    input_dir = "Embedding/chunks"
    OUT_DIR = "Embedding/data"
    os.makedirs(OUT_DIR, exist_ok=True)

    INDEX_PATH = os.path.join(OUT_DIR, "sac.index")
    META_PATH  = os.path.join(OUT_DIR, "meta.jsonl")

    # 如需从零开始，先清空旧产物
    if os.path.exists(META_PATH):
        os.remove(META_PATH)
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)

    index = None
    dim = None

    # 按“文件批”缓冲；这些向量可能来自这1000个文件中的多个chunk
    batch_vecs, batch_ids, batch_metas = [], [], []

    # 计数器
    files_in_batch = 0
    total_files_processed = 0
    total_vectors_written = 0
    next_vid = 0  # 全局向量ID

    with open(META_PATH, 'w', encoding="utf-8") as meta_fh:
        # 遍历文件（仅处理普通文件）
        for file in tqdm(os.listdir(input_dir)):
            file_path = os.path.join(input_dir, file)
            if not os.path.isfile(file_path):
                continue

            # ——开始处理一个文件——
            with open(file_path, 'r', encoding="utf-8") as f:
                chunks = f.read()
                chunk_list = [c for c in chunks.lstrip('\ufeff\r\n ').split("Chunks(md_content):\n") if c.strip()]
                doc_id = Path(file).stem

                for i, raw in enumerate(chunk_list, start=1):
                    t = raw.strip()
                    if not t:
                        continue
                    try:
                        v = get_embedding(t, model_name="gl_embedding")
                    except Exception as e:
                        print(f"[WARN] embed failed @ {file} chunk#{i}: {e}")
                        continue

                    # 索引懒初始化
                    if index is None:
                        dim = v.shape[0]
                        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

                    # 加入当前文件批的缓冲
                    batch_vecs.append(v.astype("float32"))
                    batch_ids.append(next_vid)
                    batch_metas.append({
                        "vid": int(next_vid),
                        "source_file": file,
                        "doc_id": doc_id,
                        "chunk_id": i,
                        "text": t,
                        "char_len": len(t)
                    })
                    next_vid += 1

            # ——该文件处理完毕——
            files_in_batch += 1
            total_files_processed += 1

            # 满1000个文件 -> flush
            if files_in_batch >= BATCH_FILES:
                n = _flush_batch(index, INDEX_PATH, batch_vecs, batch_ids, batch_metas, meta_fh)
                total_vectors_written += n
                print(f"[FLUSH] Files={BATCH_FILES} (total_files={total_files_processed}), "
                      f"wrote {n} vectors (total_vectors={total_vectors_written}), index -> {INDEX_PATH}")
                # 清空当前批缓冲并重置计数
                batch_vecs.clear()
                batch_ids.clear()
                batch_metas.clear()
                files_in_batch = 0

        # 处理尾批（不足1000文件的剩余）
        if files_in_batch > 0:
            n = _flush_batch(index, INDEX_PATH, batch_vecs, batch_ids, batch_metas, meta_fh)
            total_vectors_written += n
            print(f"[FLUSH] Tail files={files_in_batch} (total_files={total_files_processed}), "
                  f"wrote {n} vectors (total_vectors={total_vectors_written}), index -> {INDEX_PATH}")

    if total_vectors_written == 0:
        raise RuntimeError("No embedding collected")

    print(f"Done. Total files: {total_files_processed}, total vectors: {total_vectors_written}")
    print(f"FAISS index saved -> {INDEX_PATH}")
    print(f"Metadata saved -> {META_PATH}")
