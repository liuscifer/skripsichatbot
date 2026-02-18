# main.py
import json
from extract_pdf import extract_blocks   # <-- pakai extractor baru yang menghasilkan Block
from adaptive_chunking import label_blocks, build_chunks

pdf_path = "data/IPAS.pdf"

# 1) PDF -> Blocks (heading + paragraph + steps sudah dipisah)
blocks = extract_blocks(pdf_path)

# 2) Labeling (cue pattern + category)
labeled_blocks = label_blocks(blocks)

# 3) Adaptive chunking
chunks = build_chunks(labeled_blocks)

# 4) Save chunks to TXT (enak dibaca manusia)
out_path = "chunks.txt"
with open(out_path, "w", encoding="utf-8") as f:
    for ch in chunks:
        f.write(f"{ch.chunk_id}\n")
        f.write(f"HEADINGS: {ch.meta.get('headings', [])}\n")
        f.write(f"BLOCK_IDS: {ch.block_ids}\n")
        f.write(f"CUE_PATTERNS: {ch.cue_patterns}\n")
        f.write(f"CATEGORIES: {ch.categories}\n")
        f.write("CONTENT:\n")
        f.write(ch.content)
        f.write("\n" + ("-" * 80) + "\n")

print(f"Saved {len(chunks)} chunks to {out_path}")

# 5) (Optional) Save chunks to JSONL (bagus untuk tahap embedding/RAG)
jsonl_path = "chunks.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for ch in chunks:
        row = {
            "chunk_id": ch.chunk_id,
            "headings": ch.meta.get("headings", []),
            "block_ids": ch.block_ids,
            "cue_patterns": ch.cue_patterns,
            "categories": ch.categories,
            "content": ch.content,
            "meta": ch.meta,
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved {len(chunks)} chunks to {jsonl_path}")

# 6) (Optional) Print a small preview only (biar terminal gak banjir)
print("\n=== PREVIEW (first 5 chunks) ===")
for ch in chunks[:5]:
    print(ch.chunk_id, "| headings:", ch.meta.get("headings", []))
    print(ch.content[:500], "...\n")
    print("-" * 40)