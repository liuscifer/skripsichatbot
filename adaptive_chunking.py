"""
Rule-based Adaptive Chunking (Function- & Structure-aware) for SD textbooks (Kelas 5)

Fixes included:
1) CP_GLOSSARY for "Kosakata Baru" (NOT activity/instruction)
2) Better numbered line handling: headings vs steps vs numbered questions
3) Reflection question lists become EVALUASI
4) Activity context can UNLOCK when content cues appear
5) Avoid mutating LabeledBlock
6) Tighter heading detection
7) Merge split-step / split-question blocks using continuation heuristic:
   - If previous block is INSTRUKSI and current looks like continuation => force INSTRUKSI
   - If previous block is EVALUASI and current looks like continuation => force EVALUASI
8) Optional: filter obvious book boilerplate/metadata (ISBN, Kementerian, etc.) as CP_META
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
CP_GLOSSARY = "GLOSSARY_BOX"
CP_META = "BOOK_META"


# =============================
# 3) NORMALIZATION + LEXICAL LISTS
# =============================

def normalize_text(t: str) -> str:
    t = (t or "").replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


IMPERATIVE_VERBS = {
    "amati", "perhatikan", "lakukan", "siapkan", "diskusikan",
    "tuliskan", "jawablah", "sebutkan", "urutkan", "pasangkan",
    "buatlah", "bacalah", "cermati", "pahami", "gunakan",
    "kerjakan", "cobalah", "tentukan", "prediksi", "isilah",
    "lengkapilah", "tebaklah", "ukur", "gambar", "hitung",
    "jelaskan", "uraikan", "pilihlah", "lingkarilah",
}

EVAL_TRIGGERS = {
    "latihan", "soal", "ujian",
    "jawablah", "isilah", "pasangkan", "urutkan", "pilihlah",
    "lengkapilah", "teka-teki silang", "diagram venn",
    "refleksikan", "refleksi", "mari refleksikan", "mari refleksi",
    "evaluasi", "penilaian",
}

PROMPT_WORDS = {
    "tahukah", "yuk", "mari", "pernahkah", "selamat belajar", "masih ingat"
}

ACTIVITY_HEADINGS = {
    "lakukan bersama",
    "mari mencoba",
    "ayo,",
    "ayo ",
    "belajar lebih lanjut",
    "projek", "proyek",
}

GLOSSARY_HEADINGS = {"kosakata baru"}

QUESTION_WORDS = {
    "apa", "mengapa", "bagaimana", "kapan", "siapa",
    "di mana", "dimana", "kenapa",
}

STEP_PREFIXES = (
    "saat kalian",
    "ketika kalian",
    "jika kalian",
    "bila kalian",
    "setelah itu",
    "kemudian",
    "lalu",
)

CONTENT_CUES = {CP_DEFINITION, CP_CAUSE_EFFECT, CP_EXAMPLE, CP_NARRATIVE, CP_FACT, CP_INTRO}


# =============================
# 4) HELPERS
# =============================

def looks_like_book_meta(t: str) -> bool:
    tl = normalize_text(t).lower()
    if not tl:
        return False

    patterns = [
        r"\bisbn\b",
        r"\bkementerian\b",
        r"\brepublik indonesia\b",
        r"\bkemendikbud\b",
        r"\bpenulis\b",
        r"\bjil\.\b",
        r"\bhak cipta\b",
        r"\b(pusat|balai) kurikulum\b",
    ]
    if any(re.search(p, tl) for p in patterns):
        return True

    letters = re.sub(r"[^A-Za-z]", "", t)
    if len(letters) >= 30:
        upper_ratio = sum(ch.isupper() for ch in letters) / max(1, len(letters))
        if upper_ratio > 0.85:
            return True

    return False


def is_numbered_line(t: str) -> bool:
    return bool(re.match(r"^\s*\d+\.\s+", t))


def strip_number_prefix(t: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", t).strip()


def looks_like_numbered_question(t: str) -> bool:
    t_norm = normalize_text(t)
    if not is_numbered_line(t_norm):
        return False
    after = strip_number_prefix(t_norm).lower()

    if "?" in t_norm:
        return True

    for qw in QUESTION_WORDS:
        if after.startswith(qw + " ") or after == qw:
            return True

    return False


def starts_with_imperative(t: str) -> bool:
    t = normalize_text(t).lower()
    if not t:
        return False
    first = re.split(r"\s+", t, maxsplit=1)[0]
    first = re.sub(r"[^\w-]", "", first)

    if first in IMPERATIVE_VERBS:
        return True

    if first.endswith("lah") and first[:-3] in IMPERATIVE_VERBS:
        return True

    if first.startswith(("meng", "meny", "men", "mem", "me")):
        return True

    return False


def contains_imperative_anywhere(t: str) -> bool:
    tl = normalize_text(t).lower()
    if not tl:
        return False
    words = re.findall(r"[a-zA-Z-]+", tl)
    for w in words:
        w = re.sub(r"[^\w-]", "", w)
        if w in IMPERATIVE_VERBS:
            return True
        if w.endswith("lah") and w[:-3] in IMPERATIVE_VERBS:
            return True
    return False


def numbered_line_is_heading(t: str) -> bool:
    t_norm = normalize_text(t)
    if not is_numbered_line(t_norm):
        return False

    if looks_like_numbered_question(t_norm):
        return False

    after = strip_number_prefix(t_norm).lower()
    toks = after.split()

    for tok in toks[:3]:
        tok_clean = re.sub(r"[^\w-]", "", tok)
        if tok_clean in IMPERATIVE_VERBS:
            return False
        if tok_clean.endswith("lah") and tok_clean[:-3] in IMPERATIVE_VERBS:
            return False

    for pref in STEP_PREFIXES:
        if after.startswith(pref):
            return False

    return len(toks) <= 6


def is_heading_text(t: str) -> bool:
    t = normalize_text(t)
    tl = t.lower()
    if not t:
        return False

    if re.match(r"^\s*(bab|topik)\s+\w+", tl):
        return True

    if any(tl.startswith(h) for h in GLOSSARY_HEADINGS):
        return True

    if any(tl.startswith(h) for h in ACTIVITY_HEADINGS):
        return True

    if is_numbered_line(t):
        return numbered_line_is_heading(t)

    short = len(t.split()) <= 6 and not re.search(r"[.!?]$", t)
    has_definition_word = bool(re.search(r"\b(adalah|merupakan|yaitu|disebut)\b", tl))
    has_question = "?" in t
    if short and not has_definition_word and not has_question:
        return True

    return False


def looks_like_evaluative(t: str) -> bool:
    tl = normalize_text(t).lower()
    if not tl:
        return False
    if looks_like_numbered_question(t):
        return True
    return any(trg in tl for trg in EVAL_TRIGGERS)


def looks_like_continuation(prev_text: str, cur_text: str) -> bool:
    """
    Detect if cur_text is likely a continuation of prev_text (PDF extraction split).
    Common signs:
    - prev doesn't end with strong punctuation
    - cur starts with lowercase / continuation word
    - cur is relatively short fragment
    """
    prev = normalize_text(prev_text)
    cur = normalize_text(cur_text)
    if not prev or not cur:
        return False

    # prev ends with punctuation => less likely continuation
    if re.search(r"[.!?…:]$", prev):
        return False

    cur_l = cur.lstrip()
    # starts with lowercase letter
    starts_lower = bool(re.match(r"^[a-z]", cur_l))
    # or starts with continuation connectors
    starts_connector = cur_l.lower().startswith((
        "dan ", "atau ", "serta ", "yang ", "hasil ", "kemudian", "lalu", "setelah", "karena", "sehingga"
    ))

    shortish = len(cur.split()) <= 25  # continuation fragments are often short

    return (starts_lower or starts_connector) and shortish


# =============================
# 6) CUE PATTERN DETECTION
# =============================

def detect_cue_pattern(block: Block) -> str:
    t = normalize_text(block.text)
    tl = t.lower()

    # if "kosakata baru" in tl:
    #     return CP_GLOSSARY

    if looks_like_book_meta(t):
        return CP_META

    if is_heading_text(t):
        if any(tl.startswith(h) for h in GLOSSARY_HEADINGS):
            return CP_GLOSSARY

        if "refleksikan" in tl or "refleksi" in tl:
            return CP_EVALUATIVE

        if any(tl.startswith(h) for h in ACTIVITY_HEADINGS):
            return CP_IMPERATIVE

        if looks_like_numbered_question(t):
            return CP_EVALUATIVE

        return CP_HEADING

    if is_numbered_line(t) and not numbered_line_is_heading(t):
        if looks_like_numbered_question(t):
            return CP_EVALUATIVE
        return CP_IMPERATIVE

    if starts_with_imperative(t) or tl.startswith("ayo,") or tl.startswith("ayo "):
        if looks_like_evaluative(t):
            return CP_EVALUATIVE
        return CP_IMPERATIVE

    if looks_like_evaluative(t):
        return CP_EVALUATIVE

    if any(p in tl for p in PROMPT_WORDS):
        return CP_INTRO

    if re.search(r"\b(adalah|merupakan|yaitu|disebut)\b", tl):
        return CP_DEFINITION

    if re.search(r"\b(karena|sehingga|oleh karena itu)\b", tl):
        return CP_CAUSE_EFFECT

    if re.search(r"\b(misalnya|contohnya|seperti)\b", tl):
        return CP_EXAMPLE

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
    if cue == CP_GLOSSARY:
        return "KONSEP"
    if cue == CP_META:
        return "META"
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
# 7) CHUNK BOUNDARY RULES
# =============================

def should_start_new_chunk(prev: Optional[LabeledBlock], cur: LabeledBlock) -> bool:
    if prev is None:
        return True

    if prev.category == "INSTRUKSI" and cur.category == "INSTRUKSI":
        return False

    if prev.category == "EVALUASI" and cur.category == "EVALUASI":
        return False

    if prev.category != cur.category:
        return True

    return False


# =============================
# 8) BUILD CHUNKS
# =============================

def build_chunks(labeled: List[LabeledBlock], skip_meta_blocks: bool = True) -> List[Chunk]:
    chunks: List[Chunk] = []

    current_activity: Optional[str] = None  # None | INSTRUKSI | EVALUASI

    current_texts: List[str] = []
    current_ids: List[str] = []
    current_cues: List[str] = []
    current_cats: List[str] = []
    current_meta: Dict = {"headings": []}

    last_effective: Optional[LabeledBlock] = None
    last_text_in_chunk: str = ""
    chunk_counter = 1

    def flush():
        nonlocal chunk_counter, current_texts, current_ids, current_cues, current_cats, current_meta
        nonlocal last_effective, last_text_in_chunk
        if not current_texts:
            return
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
        last_effective = None
        last_text_in_chunk = ""

    for lb in labeled:
        if skip_meta_blocks and lb.cue_pattern == CP_META:
            continue

        # Headings become metadata (not content)
        if lb.cue_pattern == CP_HEADING:
            flush()
            current_activity = None
            current_meta["headings"].append(lb.text)
            continue

        # Glossary is content, but unlock activity mode
        if lb.cue_pattern == CP_GLOSSARY:
            current_activity = None

        # Lock/unlock activity context
        if lb.cue_pattern == CP_IMPERATIVE:
            current_activity = "INSTRUKSI"
        elif lb.cue_pattern == CP_EVALUATIVE:
            current_activity = "EVALUASI"
        elif lb.cue_pattern in CONTENT_CUES:
            current_activity = None

        forced_cue = lb.cue_pattern
        forced_cat = lb.category

        # --- Continuation heuristic (THE KEY FIX for your CH0036 & CH0038) ---
        if last_effective is not None and looks_like_continuation(last_text_in_chunk, lb.text):
            if last_effective.category == "INSTRUKSI":
                forced_cue = CP_IMPERATIVE
                forced_cat = "INSTRUKSI"
            elif last_effective.category == "EVALUASI":
                forced_cue = CP_EVALUATIVE
                forced_cat = "EVALUASI"

        # Merge split-step blocks in INSTRUKSI mode (even if "kemudian" triggers narrative)
        if current_activity == "INSTRUKSI":
            if forced_cue in (CP_NARRATIVE, CP_FACT, CP_INTRO) and contains_imperative_anywhere(lb.text):
                forced_cue = CP_IMPERATIVE
                forced_cat = "INSTRUKSI"

        # Keep evaluasi if looks question-ish in evaluasi mode
        if current_activity == "EVALUASI":
            if forced_cue in (CP_FACT, CP_INTRO) and ("?" in lb.text or looks_like_evaluative(lb.text)):
                forced_cue = CP_EVALUATIVE
                forced_cat = "EVALUASI"

        effective_category = forced_cat
        if current_activity in ("INSTRUKSI", "EVALUASI"):
            if forced_cue in (CP_IMPERATIVE, CP_EVALUATIVE):
                effective_category = current_activity

        effective_lb = LabeledBlock(
            id=lb.id,
            text=lb.text,
            block_type=lb.block_type,
            cue_pattern=forced_cue,
            category=effective_category,
            meta=lb.meta,
        )

        if should_start_new_chunk(last_effective, effective_lb) and current_texts:
            flush()

        current_texts.append(effective_lb.text)
        current_ids.append(effective_lb.id)
        current_cues.append(effective_lb.cue_pattern)
        current_cats.append(effective_lb.category)

        if effective_lb.meta:
            current_meta.setdefault("blocks_meta", []).append({effective_lb.id: effective_lb.meta})

        last_effective = effective_lb
        last_text_in_chunk = effective_lb.text

    flush()
    return chunks


# =============================
# OPTIONAL: debug printer
# =============================

def print_chunks(chunks: List[Chunk]) -> None:
    for ch in chunks:
        print(ch.chunk_id)
        print("HEADINGS:", ch.meta.get("headings", []))
        print("BLOCK_IDS:", ch.block_ids)
        print("CUE_PATTERNS:", ch.cue_patterns)
        print("CATEGORIES:", ch.categories)
        print("CONTENT:\n", ch.content)
        print("-" * 80)
