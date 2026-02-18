import re
import pdfplumber
from adaptive_chunking import Block

# --- patterns to remove noise lines ---
NOISE_LINE_PATTERNS = [
    r"^Sumber:\s*.*$",
    r"^Gambar\s*\d+(\.\d+)?\s+.*$",
    r"^\d+\s+Ilmu Pengetahuan.*$",
    r"^Bab\s+\d+\s*\|.*$",
    r"^\d+\s*$",  # page number alone
]

def is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return any(re.match(p, s, flags=re.IGNORECASE) for p in NOISE_LINE_PATTERNS)

# --- heading detection (SD book friendly) ---
HEADING_PATTERNS = [
    r"^Bab\s+\d+",
    r"^Topik\s+[A-Z]\s*:",
    r"^\d+\.\s+\S+",                # 1. Cahaya ...
    r"^(Lakukan Bersama|Mari Refleksikan|Belajar Lebih Lanjut|Kosakata Baru)\b",
    r"^(Ayo,|Ayo )\b",
]

def is_heading_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s.split()) <= 12:
        return any(re.match(p, s, flags=re.IGNORECASE) for p in HEADING_PATTERNS)
    return False

# --- numbered steps (instruction list items) ---
STEP_PAT = re.compile(r"^\s*(\d+)\.\s+.+$")

def extract_blocks(pdf_path: str) -> list[Block]:
    blocks: list[Block] = []
    bid = 1

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [ln.rstrip() for ln in text.split("\n")]

            buffer: list[str] = []

            def flush_paragraph():
                nonlocal bid, buffer
                if buffer:
                    para = " ".join(x.strip() for x in buffer if x.strip())
                    if len(para) >= 20:
                        blocks.append(Block(id=f"P{bid}", text=para, block_type="paragraph"))
                        bid += 1
                    buffer = []

            for ln in lines:
                ln = ln.strip()

                if is_noise_line(ln):
                    continue

                # empty line -> paragraph boundary
                if ln == "":
                    flush_paragraph()
                    continue

                # heading line -> its own block
                if is_heading_line(ln):
                    flush_paragraph()
                    blocks.append(Block(id=f"H{bid}", text=ln, block_type="heading"))
                    bid += 1
                    continue

                # numbered step -> split as its own paragraph block
                if STEP_PAT.match(ln):
                    flush_paragraph()
                    blocks.append(Block(id=f"S{bid}", text=ln, block_type="paragraph"))
                    bid += 1
                    continue

                buffer.append(ln)

            flush_paragraph()

    return blocks


if __name__ == "__main__":
    bs = extract_blocks("data/ipas.pdf")
    print("blocks:", len(bs))
    for b in bs[:30]:
        print(b.id, b.block_type, b.text[:80])
