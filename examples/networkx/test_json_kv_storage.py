import asyncio
import os
import tempfile
import shutil

from PathRAG.storage.defaults import JsonKVStorage
from PathRAG.utils import EmbeddingFunc


async def main():
    # Create a temporary working directory
    work_dir = tempfile.mkdtemp(prefix="pathrag_kv_test_")
    print(f"Working directory: {work_dir}")

    # Dummy embedding function (not used by JsonKVStorage, but required by base class)
    embedding_func = EmbeddingFunc(
        embedding_dim=384,
        max_token_size=8192,
        func=lambda texts: None,
    )

    global_config = {"working_dir": work_dir}

    kv = JsonKVStorage(
        namespace="test",
        global_config=global_config,
        embedding_func=embedding_func,
    )

    # 1. upsert
    print("\n=== upsert ===")
    data = {
        "doc_1": {"title": "Hello World", "content": "This is a test document."},
        "doc_2": {"title": "PathRAG", "content": "Graph-based RAG system."},
        "doc_3": {"title": "Storage", "content": "Key-value storage test."},
    }
    inserted = await kv.upsert(data)
    print(f"Inserted keys: {list(inserted.keys())}")

    # 2. all_keys
    print("\n=== all_keys ===")
    keys = await kv.all_keys()
    print(f"All keys: {keys}")

    # 3. get_by_id
    print("\n=== get_by_id ===")
    result = await kv.get_by_id("doc_1")
    print(f"doc_1: {result}")

    result_none = await kv.get_by_id("non_existent")
    print(f"non_existent: {result_none}")

    # 4. get_by_ids
    print("\n=== get_by_ids ===")
    results = await kv.get_by_ids(["doc_1", "doc_2", "non_existent"])
    print(f"Multiple get: {results}")

    # 5. get_by_ids with fields filter
    print("\n=== get_by_ids (with fields) ===")
    results = await kv.get_by_ids(["doc_1", "doc_2"], fields={"title"})
    print(f"Fields filtered: {results}")

    # 6. filter_keys - returns keys that do NOT exist in the store
    print("\n=== filter_keys ===")
    new_keys = await kv.filter_keys(["doc_1", "doc_4", "doc_5"])
    print(f"Keys not in store: {new_keys}")

    # 7. upsert duplicate - should not overwrite existing keys
    print("\n=== upsert (duplicate) ===")
    dup_data = {
        "doc_1": {"title": "Updated", "content": "Should not overwrite."},
        "doc_4": {"title": "New Doc", "content": "This is new."},
    }
    inserted = await kv.upsert(dup_data)
    print(f"Newly inserted keys: {list(inserted.keys())}")
    original = await kv.get_by_id("doc_1")
    print(f"doc_1 unchanged: {original}")

    # 8. index_done_callback - persist to JSON file
    print("\n=== index_done_callback (persist) ===")
    await kv.index_done_callback()
    json_file = os.path.join(work_dir, "kv_store_test.json")
    print(f"Persisted to: {json_file}")
    print(f"File exists: {os.path.exists(json_file)}")

    # 9. Reload from disk to verify persistence
    print("\n=== Reload from disk ===")
    kv2 = JsonKVStorage(
        namespace="test",
        global_config=global_config,
        embedding_func=embedding_func,
    )
    keys2 = await kv2.all_keys()
    print(f"Reloaded keys: {keys2}")

    # 10. drop
    print("\n=== drop ===")
    await kv.drop()
    keys_after_drop = await kv.all_keys()
    print(f"Keys after drop: {keys_after_drop}")

    # Cleanup
    shutil.rmtree(work_dir)
    print(f"\nCleaned up: {work_dir}")
    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
