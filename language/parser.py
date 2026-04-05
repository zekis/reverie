"""spaCy-based gap detection, scene parsing, key extraction.

One shared nlp instance. WH detection via POS tags, not word lists.
"""

import numpy as np
import spacy

from config import EMBEDDING_DIM, WH_TAGS


_nlp = spacy.load("en_core_web_sm")


def _is_wh(tok) -> bool:
    return tok.tag_ in WH_TAGS


def extract_keys(text: str) -> list[str]:
    """Nouns, verbs, proper nouns. No WH-words. No be/have/do. Length > 2."""
    doc = _nlp(text)
    return [
        t.lemma_.lower() for t in doc
        if t.pos_ in ("NOUN", "PROPN", "VERB")
        and not _is_wh(t)
        and t.lemma_.lower() not in ("be", "have", "do")
        and len(t.lemma_) > 2
    ]


def parse_gap(question: str, embedder) -> dict:
    """Detect gap role, extract subject, produce query + subject vectors."""
    doc = _nlp(question)
    role, head, root, subj = None, None, None, None
    given: list[str] = []
    for t in doc:
        if t.dep_ == "ROOT":
            root = t.lemma_.lower()
        if _is_wh(t):
            if not role:
                role, head = t.dep_, t.head.lemma_.lower()
            continue
        if t.pos_ in ("DET", "PUNCT", "AUX", "CCONJ", "SCONJ"):
            continue
        if t.dep_ in ("det", "punct", "aux", "auxpass", "cc", "mark"):
            continue
        lem = t.lemma_.lower()
        if t.dep_ in ("nsubj", "nsubjpass") and not subj:
            subj = lem
        if len(lem) > 1:
            given.append(lem)
    # Hollow-verb redirect: "what is X" → attr gap, not subject
    if role == "nsubj" and root in ("be", "do", "have"):
        role = "attr"
    # Boolean detection: no WH found, copular/aux root
    is_boolean = role is None and root in ("be", "do", "have")
    qv = embedder.embed(question)
    sv = embedder.embed(f"{question} : {subj}") if subj else qv
    return {
        "role": role or "attr",
        "root": root or "",
        "given": given,
        "subject": subj,
        "is_boolean": is_boolean,
        "query_vec": qv,
        "subject_vec": sv,
    }


def parse_scene(text: str, embedder) -> dict:
    """Extract slots, embed each contextually, return scene with keys + vec."""
    doc = _nlp(text)
    wh_heads = {t.head.i for t in doc if t.dep_ == "det" and _is_wh(t)}
    to_embed: list[str] = []
    info: list[tuple] = []
    for tok in doc:
        if tok.dep_ in ("det", "punct", "aux", "auxpass", "cc", "mark"):
            continue
        role, gap = None, False
        if tok.dep_ == "ROOT":
            role = "ROOT"
        elif tok.dep_ in ("nsubj", "nsubjpass"):
            role, gap = "nsubj", _is_wh(tok)
        elif tok.dep_ == "dobj":
            role, gap = "dobj", _is_wh(tok) or tok.i in wh_heads
        elif tok.dep_ == "attr":
            role, gap = "attr", _is_wh(tok) or tok.i in wh_heads
        elif tok.dep_ == "acomp":
            role = "acomp"
        elif tok.dep_ == "nummod":
            role = "nummod"
        elif tok.dep_ == "pobj":
            role = f"prep_{tok.head.text.lower()}"
        elif tok.dep_ == "advmod" and _is_wh(tok):
            role, gap = "advmod", True
        elif tok.dep_ == "compound" and len(tok.lemma_) > 2:
            role = "compound"
        if role:
            sub = sorted(tok.subtree, key=lambda t: t.i)
            span = " ".join(
                t.text for t in sub
                if t.dep_ not in ("det", "punct", "aux", "auxpass")
                and t.pos_ not in ("DET", "PUNCT")
                and not _is_wh(t)
            ) or tok.text
            if not gap:
                to_embed.append(f"{text} : {span}")
                info.append((role, span, False, len(to_embed) - 1))
            else:
                info.append((role, span, True, -1))
    vecs = embedder.embed_batch(to_embed) if to_embed else []
    slots = []
    for role, span, is_gap, vi in info:
        v = (np.zeros(EMBED_DIM, dtype=np.float32)
             if is_gap else vecs[vi])
        slots.append({
            "role": role, "text": span,
            "vec": v.tobytes(), "is_gap": is_gap,
        })
    non_gap = [vecs[s[3]] for s in info if not s[2] and s[3] >= 0]
    scene_vec = (np.mean(non_gap, axis=0) if non_gap
                 else np.zeros(EMBEDDING_DIM, dtype=np.float32))
    return {
        "slots": slots,
        "source_text": text,
        "vec": scene_vec.tobytes(),
        "keys": extract_keys(text),
    }


EMBED_DIM = EMBEDDING_DIM  # alias used in parse_scene
