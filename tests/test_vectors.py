from tempo.telemetry.embeddings import HashEmbedder
from tempo.telemetry.vectors import MemoryVectorStore, chunk


def test_chunk_packs_paragraphs():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    parts = chunk(text, size=500)
    assert len(parts) > 1
    assert all(len(p) < 900 for p in parts)


async def test_memory_store_finds_the_right_document():
    store = MemoryVectorStore(HashEmbedder(256))
    await store.add(
        [
            ("refund", "Customers can request a refund within 45 days of delivery."),
            ("leave", "Employees get 27 days of annual leave plus bank holidays."),
        ]
    )
    hits = await store.search("how many days of annual leave do employees get", k=1)
    assert hits[0].doc == "leave"
    assert await store.count() == 2
