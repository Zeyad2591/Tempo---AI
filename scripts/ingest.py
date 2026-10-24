"""Index the private documents in data/docs into the vector store (T-29).

python -m scripts.ingest            # uses DATABASE_URL (Tiger) if set
python -m scripts.ingest --reset    # empty the table first
"""

from __future__ import annotations

import argparse
import asyncio

from tempo.config import load_settings
from tempo.core.factory import build_embedder, build_vector_store
from tempo.telemetry.vectors import TigerVectorStore, chunk


async def main(reset: bool) -> None:
    s = load_settings()
    store = build_vector_store(s, build_embedder(s))
    if reset and isinstance(store, TigerVectorStore):
        await store.clear()
    items: list[tuple[str, str]] = []
    for path in sorted((s.data_dir / "docs").glob("*.md")):
        for c in chunk(path.read_text(encoding="utf-8")):
            items.append((path.stem, c))
    n = await store.add(items)
    docs = len({d for d, _ in items})
    print(f"Indexed {n} chunks from {docs} documents into {type(store).__name__}")
    hits = await store.search("How long do customers have to ask for a refund?", k=1)
    if hits:
        print(f"Check: top hit '{hits[0].doc}' score {hits[0].score:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    asyncio.run(main(ap.parse_args().reset))
