"""Chunk papers (title + abstract) and embed them with bge-small (M5.1).

Builds the vector-RAG corpus. Each paper's searchable text is `title` + a blank
line + `abstract` (the abstract already falls back to the title when missing —
see M3.1, `has_abstract` marks those). Most abstracts fit in a single chunk; the
word-window splitter only kicks in for the long ones, with overlap so a fact
straddling a boundary still lands in one window.

Chunks are embedded with `BAAI/bge-small-en-v1.5` (384-dim) via
sentence-transformers, L2-normalized so Qdrant cosine == dot product (M5.2).

Outputs two row-aligned files (gitignored under data/):
  - data/processed/chunks.jsonl   one JSON object per chunk (id + metadata + text)
  - data/processed/embeddings.npy float32 [n_chunks, 384], row i == chunk i

bge passages need no instruction prefix; only *queries* get the
"Represent this sentence..." prefix (that lives in the M5.2 search side).

    python src/vector/embed.py                       # chunk + embed all papers
    python src/vector/embed.py --limit 50            # quick smoke run
    python src/vector/embed.py --no-embed            # chunk only (no model load)
    python src/vector/embed.py --max-words 256 --overlap 32 --batch-size 64

Reads:  data/interim/papers_normalized.jsonl  (M3.2)
Writes: data/processed/chunks.jsonl
        data/processed/embeddings.npy
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

MAX_WORDS = 256  # bge-small caps at 512 tokens; 256 words leaves comfortable headroom
OVERLAP = 32  # words shared between consecutive chunks
BATCH_SIZE = 64


def chunk_text(text: str, max_words: int = MAX_WORDS, overlap: int = OVERLAP) -> list[str]:
    """Split on whitespace into overlapping <=max_words windows.

    Short text (the common case for abstracts) returns a single chunk unchanged.
    """
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()] if text.strip() else []
    step = max_words - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + max_words]))
        if i + max_words >= len(words):
            break
    return chunks


def build_chunks(
    records: list[dict], max_words: int = MAX_WORDS, overlap: int = OVERLAP
) -> list[dict]:
    """Turn paper records into chunk records (title+abstract -> windows + metadata)."""
    chunks = []
    for rec in records:
        title = (rec.get("title") or "").strip()
        abstract = (rec.get("abstract") or "").strip()
        body = f"{title}\n\n{abstract}".strip() if abstract else title
        for idx, ctext in enumerate(chunk_text(body, max_words, overlap)):
            chunks.append(
                {
                    "chunk_id": f"{rec['paper_id']}-{idx}",
                    "paper_id": rec["paper_id"],
                    "chunk_index": idx,
                    "title": title,
                    "year": rec.get("year"),
                    "cited_by_count": rec.get("cited_by_count"),
                    "has_abstract": rec.get("has_abstract"),
                    "text": ctext,
                }
            )
    return chunks


def embed_texts(
    texts: list[str], model_name: str = config.EMBEDDING_MODEL, batch_size: int = BATCH_SIZE
):
    """Embed texts with bge-small, L2-normalized. Returns float32 [n, dim]."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vectors.astype("float32")


def load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_chunks(chunks: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        for c in chunks:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")


def main() -> int:
    args = sys.argv[1:]

    def opt(flag: str, default):
        return type(default)(args[args.index(flag) + 1]) if flag in args else default

    max_words = opt("--max-words", MAX_WORDS)
    overlap = opt("--overlap", OVERLAP)
    batch_size = opt("--batch-size", BATCH_SIZE)
    limit = opt("--limit", 0)
    no_embed = "--no-embed" in args

    in_path = os.path.join(config.DATA_INTERIM, "papers_normalized.jsonl")
    chunks_path = os.path.join(config.DATA_PROCESSED, "chunks.jsonl")
    vectors_path = os.path.join(config.DATA_PROCESSED, "embeddings.npy")
    if not os.path.exists(in_path):
        print(f"missing {in_path} -- run src/ingest/normalize.py first", file=sys.stderr)
        return 1

    records = load_records(in_path)
    if limit:
        records = records[:limit]
    chunks = build_chunks(records, max_words, overlap)
    write_chunks(chunks, chunks_path)
    multi = sum(1 for c in chunks if c["chunk_index"] > 0)
    print(
        f"Chunked {len(records):,} papers -> {len(chunks):,} chunks "
        f"({multi:,} from multi-chunk papers) -> {chunks_path}"
    )

    if no_embed:
        print("--no-embed: skipping embedding pass")
        return 0

    import numpy as np

    print(f"Embedding {len(chunks):,} chunks with {config.EMBEDDING_MODEL} ...")
    vectors = embed_texts([c["text"] for c in chunks], config.EMBEDDING_MODEL, batch_size)
    np.save(vectors_path, vectors)
    print(f"Wrote embeddings {vectors.shape} (dim={vectors.shape[1]}) -> {vectors_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
