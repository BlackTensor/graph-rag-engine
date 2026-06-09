"""Index the M5.1 chunks + embeddings into Qdrant (M5.2).

Reads the row-aligned artifacts from `src/vector/embed.py`:
  - data/processed/chunks.jsonl   chunk metadata + text (row i)
  - data/processed/embeddings.npy float32 [n, 384], row i

and upserts them into a Qdrant collection (cosine distance — the vectors are
already L2-normalized, so cosine == dot). The Qdrant point id is the row index
(an int, as Qdrant requires); `chunk_id`/`paper_id` live in the payload, which
also carries the chunk `text` so search can hand context straight to the LLM
(M8) without a second lookup.

Upsert with stable integer ids is idempotent — re-running overwrites the same
points, never duplicates. `--reset` drops the collection first for a clean
rebuild (e.g. after re-chunking).

    python src/vector/index.py                # build/refresh the collection
    python src/vector/index.py --reset        # drop + rebuild from scratch
    python src/vector/index.py --batch-size 512

Reads:  data/processed/chunks.jsonl   (M5.1)
        data/processed/embeddings.npy (M5.1)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

BATCH_SIZE = 512


def load_chunks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def make_points(chunks: list[dict], vectors) -> list:
    """Row-aligned chunks + vectors -> Qdrant PointStructs (id = row index)."""
    from qdrant_client.models import PointStruct

    if len(chunks) != len(vectors):
        raise ValueError(f"chunks ({len(chunks)}) != vectors ({len(vectors)}) -- re-run embed.py")
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        points.append(PointStruct(id=i, vector=vec.tolist(), payload=chunk))
    return points


def ensure_collection(client, collection: str, dim: int, reset: bool) -> None:
    """Create the collection (cosine), optionally dropping an existing one first."""
    from qdrant_client.models import Distance, VectorParams

    if reset and client.collection_exists(collection):
        client.delete_collection(collection)
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def index(client, chunks: list[dict], vectors, collection: str, batch_size: int, reset: bool):
    ensure_collection(client, collection, int(vectors.shape[1]), reset)
    points = make_points(chunks, vectors)
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection, points=points[i : i + batch_size])
    return client.count(collection_name=collection, exact=True).count


def main() -> int:
    args = sys.argv[1:]
    reset = "--reset" in args
    batch_size = int(args[args.index("--batch-size") + 1]) if "--batch-size" in args else BATCH_SIZE

    chunks_path = os.path.join(config.DATA_PROCESSED, "chunks.jsonl")
    vectors_path = os.path.join(config.DATA_PROCESSED, "embeddings.npy")
    for p in (chunks_path, vectors_path):
        if not os.path.exists(p):
            print(f"missing {p} -- run src/vector/embed.py first", file=sys.stderr)
            return 1

    import numpy as np
    from qdrant_client import QdrantClient

    chunks = load_chunks(chunks_path)
    vectors = np.load(vectors_path)
    print(f"Loaded {len(chunks):,} chunks + vectors {vectors.shape}")

    client = QdrantClient(url=config.QDRANT_URL)
    total = index(client, chunks, vectors, config.QDRANT_COLLECTION, batch_size, reset)
    print(f"Indexed -> Qdrant collection '{config.QDRANT_COLLECTION}' now holds {total:,} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
