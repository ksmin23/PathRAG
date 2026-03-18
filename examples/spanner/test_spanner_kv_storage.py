"""
SpannerKVStorage Test Suite

Tests CRUD operations on SpannerKVStorage.

Prerequisites:
  - Set GCP credentials (GOOGLE_APPLICATION_CREDENTIALS or gcloud auth)
  - Set Spanner config in examples/spanner/.env or as environment variables:
      SPANNER_INSTANCE=<instance-id>
      SPANNER_DATABASE=<database-id>
  - pip install -e .
  - pip install google-cloud-spanner

Usage:
  python examples/spanner/test_spanner_kv_storage.py            # run tests (no cleanup)
  python examples/spanner/test_spanner_kv_storage.py --cleanup  # run tests then cleanup
  python examples/spanner/test_spanner_kv_storage.py --cleanup-only  # cleanup only
"""

import argparse
import asyncio
import json
import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


SPANNER_INSTANCE = os.environ.get("SPANNER_INSTANCE")
SPANNER_DATABASE = os.environ.get("SPANNER_DATABASE")

NAMESPACE = "test_kv"


def _check_env():
    if not SPANNER_INSTANCE or not SPANNER_DATABASE:
        print("[SKIP] SPANNER_INSTANCE and SPANNER_DATABASE must be set.")
        print("  export SPANNER_INSTANCE=<your-instance-id>")
        print("  export SPANNER_DATABASE=<your-database-id>")
        return False
    print(f"  Spanner Instance: {SPANNER_INSTANCE}")
    print(f"  Spanner Database: {SPANNER_DATABASE}")
    return True


def _create_storage(namespace: str = NAMESPACE):
    """Create a SpannerKVStorage instance for testing."""
    from PathRAG.spanner_kv_storage import SpannerKVStorage
    from PathRAG.utils import EmbeddingFunc
    import numpy as np

    dummy_embedding = EmbeddingFunc(
        embedding_dim=3,
        max_token_size=100,
        func=lambda texts: np.zeros((len(texts), 3)),
    )

    return SpannerKVStorage(
        namespace=namespace,
        global_config={
            "spanner_instance_id": SPANNER_INSTANCE,
            "spanner_database_id": SPANNER_DATABASE,
        },
        embedding_func=dummy_embedding,
    )


# ===================================================================
# Step 1: Upsert & basic reads
# ===================================================================
async def test_step1_upsert_and_read():
    """Test upsert, get_by_id, get_by_ids."""
    print("=" * 70)
    print("STEP 1: Upsert & Read Operations")
    print("=" * 70)

    if not _check_env():
        return

    kv = _create_storage()

    # --- Upsert data ---
    print("\n--- Upserting 3 entries ---")
    data = {
        "doc-001": {
            "content": "Apple is a tech company.",
            "tokens": 6,
            "full_doc_id": "full-001",
            "chunk_order_index": 0,
        },
        "doc-002": {
            "content": "Google is a search engine company.",
            "tokens": 7,
            "full_doc_id": "full-002",
            "chunk_order_index": 0,
        },
        "doc-003": {
            "content": "Microsoft develops Windows OS.",
            "tokens": 5,
            "full_doc_id": "full-003",
            "chunk_order_index": 0,
        },
    }
    left = await kv.upsert(data)
    print(f"  Inserted keys: {list(left.keys())}")

    # --- get_by_id ---
    print("\n--- get_by_id ---")
    for key in ["doc-001", "doc-002", "doc-003", "non-existent"]:
        val = await kv.get_by_id(key)
        if val:
            print(f"  {key}: content={val.get('content', 'N/A')}")
        else:
            print(f"  {key}: None")

    # --- get_by_ids ---
    print("\n--- get_by_ids (all fields) ---")
    results = await kv.get_by_ids(["doc-001", "doc-003", "non-existent"])
    for key, val in zip(["doc-001", "doc-003", "non-existent"], results):
        print(f"  {key}: {json.dumps(val) if val else None}")

    # --- get_by_ids with fields filter ---
    print("\n--- get_by_ids (fields={content, tokens}) ---")
    results = await kv.get_by_ids(
        ["doc-001", "doc-002"], fields={"content", "tokens"}
    )
    for key, val in zip(["doc-001", "doc-002"], results):
        print(f"  {key}: {json.dumps(val) if val else None}")

    print("\n[OK] Upsert & Read test passed\n")


# ===================================================================
# Step 2: Upsert skip-existing behaviour
# ===================================================================
async def test_step2_upsert_skip_existing():
    """Test that upsert only inserts new keys (skips existing)."""
    print("=" * 70)
    print("STEP 2: Upsert Skip-Existing Behaviour")
    print("=" * 70)

    if not _check_env():
        return

    kv = _create_storage()

    # Try upserting a mix of existing + new keys
    print("\n--- Upserting mix of existing and new keys ---")
    data = {
        "doc-001": {
            "content": "MODIFIED Apple content (should NOT be written)",
            "tokens": 99,
            "full_doc_id": "full-001",
            "chunk_order_index": 0,
        },
        "doc-004": {
            "content": "Amazon is an e-commerce company.",
            "tokens": 6,
            "full_doc_id": "full-004",
            "chunk_order_index": 0,
        },
    }
    left = await kv.upsert(data)
    print(f"  Actually inserted keys: {list(left.keys())}")

    # Verify doc-001 was NOT overwritten
    val = await kv.get_by_id("doc-001")
    print(f"\n  doc-001 content (should be original): {val.get('content', 'N/A')}")
    assert "MODIFIED" not in val.get("content", ""), "doc-001 should NOT be overwritten!"

    # Verify doc-004 was inserted
    val = await kv.get_by_id("doc-004")
    print(f"  doc-004 content: {val.get('content', 'N/A')}")

    print("\n[OK] Upsert Skip-Existing test passed\n")


# ===================================================================
# Step 3: all_keys & filter_keys
# ===================================================================
async def test_step3_keys():
    """Test all_keys and filter_keys."""
    print("=" * 70)
    print("STEP 3: all_keys & filter_keys")
    print("=" * 70)

    if not _check_env():
        return

    kv = _create_storage()

    # --- all_keys ---
    print("\n--- all_keys ---")
    keys = await kv.all_keys()
    print(f"  Keys ({len(keys)}): {keys}")

    # --- filter_keys (return keys NOT in storage) ---
    print("\n--- filter_keys ---")
    candidates = ["doc-001", "doc-002", "doc-999", "doc-888"]
    missing = await kv.filter_keys(candidates)
    print(f"  Candidates: {candidates}")
    print(f"  Missing (not in storage): {missing}")
    assert "doc-999" in missing, "doc-999 should be missing"
    assert "doc-888" in missing, "doc-888 should be missing"
    assert "doc-001" not in missing, "doc-001 should exist"

    print("\n[OK] Keys test passed\n")


# ===================================================================
# Step 4: drop (clear all data)
# ===================================================================
async def test_step4_drop():
    """Test drop to clear all data."""
    print("=" * 70)
    print("STEP 4: Drop (Clear All Data)")
    print("=" * 70)

    if not _check_env():
        return

    kv = _create_storage()

    keys_before = await kv.all_keys()
    print(f"\n  Keys before drop: {len(keys_before)}")

    await kv.drop()

    keys_after = await kv.all_keys()
    print(f"  Keys after drop: {len(keys_after)}")
    assert len(keys_after) == 0, "All keys should be removed after drop"

    print("\n[OK] Drop test passed\n")


# ===================================================================
# Step 5: Cleanup (drop table from Spanner)
# ===================================================================
async def test_step5_cleanup():
    """Drop the test KV table from Spanner."""
    print("=" * 70)
    print("STEP 5: Cleanup Test Data")
    print("=" * 70)

    if not _check_env():
        return

    from google.cloud import spanner

    client = spanner.Client()
    instance = client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)

    table_name = f"{NAMESPACE}_kv"

    print(f"\n  Dropping table '{table_name}'...")
    op = database.update_ddl([f"DROP TABLE IF EXISTS {table_name}"])
    op.result()

    print("\n[OK] Cleanup completed\n")


# ===================================================================
# Main entry point
# ===================================================================
async def main():
    parser = argparse.ArgumentParser(description="SpannerKVStorage Test Suite")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Run cleanup after tests to drop test table",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only run cleanup (skip all tests)",
    )
    args = parser.parse_args()

    print()
    print("*" * 70)
    print("  SpannerKVStorage Test Suite")
    print("*" * 70)
    print()

    if args.cleanup_only:
        await test_step5_cleanup()
    else:
        await test_step1_upsert_and_read()
        await test_step2_upsert_skip_existing()
        await test_step3_keys()
        await test_step4_drop()

        if args.cleanup:
            await test_step5_cleanup()

    print("*" * 70)
    print("  All SpannerKVStorage tests completed!")
    print("*" * 70)


if __name__ == "__main__":
    asyncio.run(main())
