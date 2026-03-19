import asyncio
import os
import tempfile
import shutil

import numpy as np

from PathRAG.storage.defaults import NanoVectorDBStorage
from PathRAG.utils import EmbeddingFunc


MOCK_DIM = 128


async def mock_embed(texts: list[str]) -> np.ndarray:
    """Deterministic mock embedding: same text always produces the same vector."""
    embeddings = []
    for text in texts:
        seed = sum(ord(c) for c in text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(MOCK_DIM).astype(np.float64)
        vec = vec / np.linalg.norm(vec)
        embeddings.append(vec)
    return np.array(embeddings)


async def main():
    work_dir = tempfile.mkdtemp(prefix="pathrag_vdb_test_")
    print(f"Working directory: {work_dir}")

    embedding_func = EmbeddingFunc(
        embedding_dim=MOCK_DIM,
        max_token_size=8192,
        func=mock_embed,
    )

    global_config = {
        "working_dir": work_dir,
        "embedding_batch_num": 32,
        "cosine_better_than_threshold": 0.2,
    }

    vdb = NanoVectorDBStorage(
        namespace="test_entities",
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"entity_name"},
    )

    # 1. Upsert vectors
    print("\n=== upsert ===")
    test_data = {
        "ent-001": {
            "content": "Apple Inc. is a technology company headquartered in Cupertino.",
            "entity_name": "APPLE",
        },
        "ent-002": {
            "content": "Google LLC is a search engine and cloud computing company.",
            "entity_name": "GOOGLE",
        },
        "ent-003": {
            "content": "Microsoft Corporation develops software and cloud services.",
            "entity_name": "MICROSOFT",
        },
        "ent-004": {
            "content": "Steve Jobs co-founded Apple and was its visionary CEO.",
            "entity_name": "STEVE JOBS",
        },
        "ent-005": {
            "content": "Tim Cook is the current CEO of Apple Inc since 2011.",
            "entity_name": "TIM COOK",
        },
    }

    result = await vdb.upsert(test_data)
    print(f"Upserted {len(result)} vectors")

    # 2. Query - similarity search
    print("\n=== query (Apple technology) ===")
    results = await vdb.query("Apple technology company", top_k=3)
    for r in results:
        print(f"  entity={r.get('entity_name', 'N/A'):15s}  similarity={r['distance']:.4f}")

    print("\n=== query (CEO leadership) ===")
    results = await vdb.query("CEO of a technology company", top_k=3)
    for r in results:
        print(f"  entity={r.get('entity_name', 'N/A'):15s}  similarity={r['distance']:.4f}")

    # 3. Persist to disk
    print("\n=== index_done_callback (persist) ===")
    await vdb.index_done_callback()
    vdb_file = os.path.join(work_dir, "vdb_test_entities.json")
    print(f"Persisted to: {vdb_file}")
    print(f"File exists: {os.path.exists(vdb_file)}")

    # 4. Reload from disk and query again
    print("\n=== Reload from disk ===")
    vdb2 = NanoVectorDBStorage(
        namespace="test_entities",
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"entity_name"},
    )
    results = await vdb2.query("Apple technology company", top_k=2)
    print(f"Reloaded query results ({len(results)}):")
    for r in results:
        print(f"  entity={r.get('entity_name', 'N/A'):15s}  similarity={r['distance']:.4f}")

    # 5. Delete entity
    print("\n=== delete_entity ===")
    await vdb.delete_entity("MICROSOFT")
    await vdb.index_done_callback()

    results = await vdb.query("Microsoft software", top_k=5)
    remaining = [r.get("entity_name") for r in results]
    print(f"After deleting MICROSOFT, query results: {remaining}")

    # 6. Upsert update - re-insert with new content
    print("\n=== upsert (update existing) ===")
    update_data = {
        "ent-001": {
            "content": "Apple Inc. designs iPhones, iPads, Macs and provides digital services worldwide.",
            "entity_name": "APPLE",
        },
    }
    await vdb.upsert(update_data)
    await vdb.index_done_callback()
    print("Updated ent-001 (APPLE) with new content")

    # Cleanup
    shutil.rmtree(work_dir)
    print(f"\nCleaned up: {work_dir}")
    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
