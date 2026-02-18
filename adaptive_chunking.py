"""
Rule-based Adaptive Chunking (Function- & Structure-aware) for SD textbooks
- Detect cue patterns from text (NO embedding needed for chunk boundaries)
- Map cue patterns -> categories (KONSEP/NARASI/INSTRUKSI/EVALUASI)
- Build adaptive chunks based on structural + functional rules
- Optional: after chunking, you can embed each chunk for RAG (not shown here)

You can copy-paste this into a .py file and run it.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import re

# =============================
# 1) DATA STRUCTURES
# =============================

@dataclass
class Block:
    id: str
    text: str
    block_type: str = "paragraph"
    meta: Optional[Dict] = None


@dataclass
class LabeledBlock:
    id: str
    text: str
    block_type: str
    cue_pattern: str
    category: str
    meta: Optional[Dict] = None


@dataclass
class Chunk:
    chunk_id: str
    texts: List[str]
    block_ids: List[str]
    cue_patterns: List[str]
    categories: List[str]
    meta: Dict

    @property
    def content(self) -> str:
        return "\n".join(self.texts)


# =============================
# 2) CUE PATTERNS
# =============================
CP_HEADING = "NEW_SUBTOPIC_HEADING"
CP_IMPERATIVE = "IMPERATIVE_TASK"
CP_EVALUATIVE = "EVALUATIVE_QUESTION"
CP_DEFINITION = "DEFINITION"
CP_CAUSE_EFFECT = "CAUSE_EFFECT"
CP_EXAMPLE = "EXAMPLE_ILLUSTRATION"
CP_NARRATIVE = "NARRATIVE_SEQUENCE"
CP_FACT = "FACT_EXPLANATION"
CP_INTRO = "PROMPT_INTRO"
# TAMBAHIN CP_ACTIVITY MISALNYA CP_ACTIVITY = "ACTIVITY_HEADING" -- HEADING DARI AKTIVITAS DI BUKU

# =============================
# 3) LEXICAL LISTS
# =============================

def normalize_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


IMPERATIVE_VERBS = {
    "amati", "perhatikan", "lakukan", "siapkan", "diskusikan",
    "tuliskan", "jawablah", "sebutkan", "urutkan", "pasangkan",
    "buatlah", "bacalah", "cermati", "pahami", "gunakan",
    "kerjakan", "cobalah", "tentukan", "prediksi", "isilah"
}

EVAL_TRIGGERS = {
    "latihan", "soal", "ujian",
    "jawablah", "isilah", "pasangkan", "urutkan", "pilihlah",
    "lengkapilah", "teka-teki silang", "diagram venn"
}

PROMPT_WORDS = {
    "tahukah", "yuk", "mari", "pernahkah", "selamat belajar", "masih ingat"
}

ACTIVITY_HEADINGS = {
    "lakukan bersama",
    "mari mencoba",
    "mari refleksikan",
    "mari refleksi",
    "ayo,",
    "ayo ",
    "kosakata baru",
    "belajar lebih lanjut",
}


# =============================
# 4) HEADING DETECTION
# =============================

def is_numbered_line(t: str) -> bool:
    return bool(re.match(r"^\s*\d+\.\s+", t))


def numbered_line_is_heading(t: str) -> bool:
    """
    Decide if '1. ...' is a CONTENT heading or an INSTRUCTION step.
    Rule: if the 2nd token is an imperative verb => instruction step (NOT heading).
    Otherwise => heading.
    """
    tl = t.lower().strip()
    words = tl.split()
    if len(words) < 2:
        return False
    second = words[1]
    return second not in IMPERATIVE_VERBS


def is_heading_text(t: str) -> bool:
    t = normalize_text(t)
    tl = t.lower()

    # "Bab", "Topik" headings
    if re.match(r"^\s*(bab|topik)\s+\w+", tl):
        return True

    # Activity headings like "Lakukan Bersama", "Kosakata Baru"
    if any(h in tl for h in ACTIVITY_HEADINGS):
        return True

    # Numbered headings
    if is_numbered_line(t):
        return numbered_line_is_heading(t)

    # short title-like line
    if len(t.split()) <= 6 and not re.search(r"[.!?]$", t):
        return True

    return False


def starts_with_imperative(t: str) -> bool:
    t = normalize_text(t).lower()
    first = re.split(r"\s+", t, maxsplit=1)[0]
    first = re.sub(r"[^\w-]", "", first)
    return first in IMPERATIVE_VERBS


def looks_like_evaluative(t: str) -> bool:
    tl = normalize_text(t).lower()
    return any(trg in tl for trg in EVAL_TRIGGERS)


# =============================
# 5) CUE PATTERN DETECTION
# =============================

def detect_cue_pattern(block: Block) -> str:
    t = normalize_text(block.text)
    tl = t.lower()

    # A) Activity/Content headings
    if is_heading_text(t):
        # refleksi => evaluasi
        if "refleksikan" in tl or "refleksi" in tl:
            return CP_EVALUATIVE
        # activity heading => imperative context
        if any(h in tl for h in ACTIVITY_HEADINGS):
            return CP_IMPERATIVE
        return CP_HEADING

    # B) Numbered instruction steps (1. siapkan..., 2. lakukan...)
    if is_numbered_line(t) and not numbered_line_is_heading(t):
        return CP_IMPERATIVE

    # C) Imperative sentences
    if starts_with_imperative(t) or tl.startswith("ayo,") or tl.startswith("ayo "):
        if looks_like_evaluative(t):
            return CP_EVALUATIVE
        return CP_IMPERATIVE

    # D) Evaluative (Latihan/Soal)
    if looks_like_evaluative(t):
        return CP_EVALUATIVE

    # E) Prompt intro
    if "?" in t and any(p in tl for p in PROMPT_WORDS):
        return CP_INTRO
    if any(p in tl for p in PROMPT_WORDS):
        return CP_INTRO

    # F) Definition
    if re.search(r"\b(adalah|merupakan|yaitu|disebut)\b", tl):
        return CP_DEFINITION

    # G) Cause-effect
    if re.search(r"\b(karena|sehingga|oleh karena itu)\b", tl):
        return CP_CAUSE_EFFECT

    # H) Example
    if re.search(r"\b(misalnya|contohnya|seperti)\b", tl):
        return CP_EXAMPLE

    # I) Narrative
    if re.search(r"\b(dahulu|suatu hari|kemudian|tinggallah)\b", tl):
        return CP_NARRATIVE

    return CP_FACT


def cue_to_category(cue: str) -> str:
    if cue == CP_IMPERATIVE:
        return "INSTRUKSI"
    if cue == CP_EVALUATIVE:
        return "EVALUASI"
    if cue == CP_NARRATIVE:
        return "NARASI"
    return "KONSEP"


def label_blocks(blocks: List[Block]) -> List[LabeledBlock]:
    out: List[LabeledBlock] = []
    for b in blocks:
        cue = detect_cue_pattern(b)
        cat = cue_to_category(cue)
        out.append(
            LabeledBlock(
                id=b.id,
                text=normalize_text(b.text),
                block_type=b.block_type,
                cue_pattern=cue,
                category=cat,
                meta=b.meta or {},
            )
        )
    return out


# =============================
# 6) CHUNKING RULE
# =============================

def should_start_new_chunk(prev: Optional[LabeledBlock], cur: LabeledBlock) -> bool:
    if prev is None:
        return True

    # Always break when switching between Concept/Narrative and Activity (Instr/Eval)
    if prev.category in ("INSTRUKSI", "EVALUASI") and cur.category not in ("INSTRUKSI", "EVALUASI"):
        return True
    if cur.category in ("INSTRUKSI", "EVALUASI") and prev.category not in ("INSTRUKSI", "EVALUASI"):
        return True

    # Keep consecutive INSTRUKSI together
    if prev.category == "INSTRUKSI" and cur.category == "INSTRUKSI":
        return False

    # Keep consecutive EVALUASI together
    if prev.category == "EVALUASI" and cur.category == "EVALUASI":
        return False

    # For concepts, break on new content heading cue
    if cur.cue_pattern == CP_HEADING:
        return True

    # Default: break when category changes
    if prev.category != cur.category:
        return True

    return False


# =============================
# 7) BUILD CHUNKS
# =============================

def build_chunks(labeled: List[LabeledBlock]) -> List[Chunk]:
    chunks: List[Chunk] = []

    current_activity: Optional[str] = None  # None | INSTRUKSI | EVALUASI
    current_texts: List[str] = []
    current_ids: List[str] = []
    current_cues: List[str] = []
    current_cats: List[str] = []
    current_meta: Dict = {"headings": []}

    last: Optional[LabeledBlock] = None
    chunk_counter = 1

    for lb in labeled:
        # --- handle headings as metadata, but FORCE break if we already have content ---
        if lb.cue_pattern == CP_HEADING:
            # If there is running content, close it so headings don't "pile up"
            if current_texts:
                chunks.append(
                    Chunk(
                        chunk_id=f"CH{chunk_counter:04d}",
                        texts=current_texts,
                        block_ids=current_ids,
                        cue_patterns=current_cues,
                        categories=current_cats,
                        meta=current_meta,
                    )
                )
                chunk_counter += 1
                current_texts, current_ids, current_cues, current_cats = [], [], [], []
                current_meta = {"headings": []}

            current_activity = None
            current_meta["headings"].append(lb.text)
            last = lb
            continue

        # --- lock activity context ---
        if lb.cue_pattern == CP_IMPERATIVE:
            current_activity = "INSTRUKSI"
        elif lb.cue_pattern == CP_EVALUATIVE:
            current_activity = "EVALUASI"

        if current_activity:
            lb.category = current_activity

        # --- split chunk if needed ---
        new_chunk = should_start_new_chunk(last, lb)
        if new_chunk and current_texts:
            chunks.append(
                Chunk(
                    chunk_id=f"CH{chunk_counter:04d}",
                    texts=current_texts,
                    block_ids=current_ids,
                    cue_patterns=current_cues,
                    categories=current_cats,
                    meta=current_meta,
                )
            )
            chunk_counter += 1
            current_texts, current_ids, current_cues, current_cats = [], [], [], []
            current_meta = {"headings": []}

        # --- add block ---
        current_texts.append(lb.text)
        current_ids.append(lb.id)
        current_cues.append(lb.cue_pattern)
        current_cats.append(lb.category)
        last = lb

    # --- final flush ---
    if current_texts:
        chunks.append(
            Chunk(
                chunk_id=f"CH{chunk_counter:04d}",
                texts=current_texts,
                block_ids=current_ids,
                cue_patterns=current_cues,
                categories=current_cats,
                meta=current_meta,
            )
        )

    return chunks
