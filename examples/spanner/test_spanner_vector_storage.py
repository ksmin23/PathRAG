"""
SpannerVectorDBStorage + SpannerGraphStorage Integration Test Suite

Tests SpannerVectorDBStorage CRUD operations and verifies full PathRAG
integration with both SpannerVectorDBStorage and SpannerGraphStorage.

Prerequisites:
  - Set GCP credentials (GOOGLE_APPLICATION_CREDENTIALS or gcloud auth)
  - Set Spanner config in examples/spanner/.env or as environment variables:
      SPANNER_INSTANCE=<instance-id>
      SPANNER_DATABASE=<database-id>
  - Set LLM API key: GEMINI_API_KEY or OPENAI_API_KEY
  - pip install -e .
  - pip install google-cloud-spanner

Usage:
  python examples/spanner/test_spanner_vector_storage.py            # run tests (no cleanup)
  python examples/spanner/test_spanner_vector_storage.py --cleanup  # run tests then cleanup
  python examples/spanner/test_spanner_vector_storage.py --cleanup-only  # cleanup only
"""

import argparse
import asyncio
import os

import numpy as np
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

SPANNER_INSTANCE = os.environ.get("SPANNER_INSTANCE")
SPANNER_DATABASE = os.environ.get("SPANNER_DATABASE")

VECTOR_TEST_NAMESPACE = "test_vec"


def _check_env():
    if not SPANNER_INSTANCE or not SPANNER_DATABASE:
        print("[SKIP] SPANNER_INSTANCE and SPANNER_DATABASE must be set.")
        print("  export SPANNER_INSTANCE=<your-instance-id>")
        print("  export SPANNER_DATABASE=<your-database-id>")
        return False
    print(f"  Spanner Instance: {SPANNER_INSTANCE}")
    print(f"  Spanner Database: {SPANNER_DATABASE}")
    return True


def _get_llm_config():
    """Resolve LLM configuration from environment variables."""
    if os.environ.get("GEMINI_API_KEY"):
        default_llm = "gemini/gemini-2.5-flash"
        default_embed = "gemini/gemini-embedding-001"
        default_dim = 3072
    elif os.environ.get("OPENAI_API_KEY"):
        default_llm = "gpt-4o-mini"
        default_embed = "text-embedding-3-small"
        default_dim = 1536
    else:
        return None, None, None, None

    llm_model = os.environ.get("LLM_MODEL_NAME", default_llm)
    embedding_model = os.environ.get("EMBEDDING_MODEL_NAME", default_embed)
    embedding_dim = int(os.environ.get("EMBEDDING_DIM", default_dim))
    tiktoken_model = llm_model

    print(f"  LLM Model: {llm_model}")
    print(f"  Embedding Model: {embedding_model} (dim={embedding_dim})")
    return llm_model, embedding_model, embedding_dim, tiktoken_model


def _create_mock_embedding_func(dim=128):
    """Create a deterministic mock embedding function for unit tests."""
    from PathRAG.utils import EmbeddingFunc

    async def _mock_embed(texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            seed = sum(ord(c) for c in text) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(dim).astype(np.float64)
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)
        return np.array(embeddings)

    return EmbeddingFunc(embedding_dim=dim, max_token_size=8192, func=_mock_embed)


def _create_vector_storage(namespace: str = VECTOR_TEST_NAMESPACE, meta_fields=None):
    """Create a SpannerVectorDBStorage instance for testing."""
    from PathRAG.spanner_vector_storage import SpannerVectorDBStorage

    if meta_fields is None:
        meta_fields = set()

    embedding_func = _create_mock_embedding_func(dim=128)

    return SpannerVectorDBStorage(
        namespace=namespace,
        global_config={
            "spanner_instance_id": SPANNER_INSTANCE,
            "spanner_database_id": SPANNER_DATABASE,
            "embedding_batch_num": 32,
        },
        embedding_func=embedding_func,
        meta_fields=meta_fields,
    )


# ===================================================================
# Step 1: Schema creation & vector upsert
# ===================================================================
async def test_step1_upsert():
    """Test vector upsert with embeddings."""
    print("=" * 70)
    print("STEP 1: Vector Upsert")
    print("=" * 70)

    if not _check_env():
        return

    vdb = _create_vector_storage(meta_fields={"entity_name"})

    # Insert test data
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

    print(f"\n--- Inserting {len(test_data)} vectors ---")
    result = await vdb.upsert(test_data)
    print(f"  Inserted {len(result)} vectors")

    # Verify by counting rows in Spanner
    from google.cloud.spanner_v1 import param_types

    with vdb._database.snapshot() as snap:
        rows = list(
            snap.execute_sql(f"SELECT COUNT(*) FROM {vdb._table_name}")
        )
        count = rows[0][0]
    print(f"  Total rows in table '{vdb._table_name}': {count}")

    assert count == len(test_data), f"Expected {len(test_data)} rows, got {count}"

    print("\n[OK] Vector Upsert test passed\n")


# ===================================================================
# Step 2: Vector similarity query
# ===================================================================
async def test_step2_query():
    """Test vector similarity search using COSINE_DISTANCE."""
    print("=" * 70)
    print("STEP 2: Vector Similarity Query")
    print("=" * 70)

    if not _check_env():
        return

    vdb = _create_vector_storage(meta_fields={"entity_name"})

    # Query for Apple-related content
    print("\n--- Query: 'Apple technology company' ---")
    results = await vdb.query("Apple technology company", top_k=3)
    print(f"  Found {len(results)} results:")
    for r in results:
        print(
            f"    id={r['id']}, entity_name={r.get('entity_name', 'N/A')}, "
            f"similarity={r['distance']:.4f}"
        )
        print(f"      content: {r['content'][:80]}...")

    # Query for person-related content
    print("\n--- Query: 'CEO of a technology company' ---")
    results = await vdb.query("CEO of a technology company", top_k=3)
    print(f"  Found {len(results)} results:")
    for r in results:
        print(
            f"    id={r['id']}, entity_name={r.get('entity_name', 'N/A')}, "
            f"similarity={r['distance']:.4f}"
        )
        print(f"      content: {r['content'][:80]}...")

    print("\n[OK] Vector Query test passed\n")


# ===================================================================
# Step 3: Upsert (update existing vectors)
# ===================================================================
async def test_step3_upsert_update():
    """Test that upserting existing IDs updates the vectors."""
    print("=" * 70)
    print("STEP 3: Upsert Update (overwrite existing)")
    print("=" * 70)

    if not _check_env():
        return

    vdb = _create_vector_storage(meta_fields={"entity_name"})

    # Update APPLE entity with new content
    update_data = {
        "ent-001": {
            "content": "Apple Inc. designs iPhones, iPads, Macs and provides digital services worldwide.",
            "entity_name": "APPLE",
        },
    }

    print("\n--- Updating ent-001 (APPLE) ---")
    result = await vdb.upsert(update_data)
    print(f"  Updated {len(result)} vector(s)")

    # Verify the content was updated
    from google.cloud.spanner_v1 import param_types

    with vdb._database.snapshot() as snap:
        rows = list(
            snap.execute_sql(
                f"SELECT content FROM {vdb._table_name} WHERE id = @id",
                params={"id": "ent-001"},
                param_types={"id": param_types.STRING},
            )
        )
    updated_content = rows[0][0]
    print(f"  Updated content: {updated_content[:80]}...")

    assert "iPhones" in updated_content, "Content was not updated"

    # Verify total count hasn't changed
    with vdb._database.snapshot() as snap:
        rows = list(
            snap.execute_sql(f"SELECT COUNT(*) FROM {vdb._table_name}")
        )
    print(f"  Total rows (should still be 5): {rows[0][0]}")

    print("\n[OK] Upsert Update test passed\n")


# ===================================================================
# Step 4: Delete operations
# ===================================================================
async def test_step4_delete():
    """Test entity and relation deletion."""
    print("=" * 70)
    print("STEP 4: Delete Operations")
    print("=" * 70)

    if not _check_env():
        return

    vdb = _create_vector_storage(meta_fields={"entity_name"})

    # Count before deletion
    with vdb._database.snapshot() as snap:
        rows = list(
            snap.execute_sql(f"SELECT COUNT(*) FROM {vdb._table_name}")
        )
    print(f"\n  Rows before deletion: {rows[0][0]}")

    # Delete an entity
    print("\n--- Deleting entity 'MICROSOFT' (id: ent-003) ---")
    from PathRAG.utils import compute_mdhash_id

    # The delete_entity method computes the hash internally, but our test data
    # uses literal IDs. So we delete directly by ID for this test.
    from google.cloud import spanner as spanner_lib

    with vdb._database.batch() as batch:
        batch.delete(
            table=vdb._table_name,
            keyset=spanner_lib.KeySet(keys=[["ent-003"]]),
        )
    print("  Deleted ent-003")

    # Count after deletion
    with vdb._database.snapshot() as snap:
        rows = list(
            snap.execute_sql(f"SELECT COUNT(*) FROM {vdb._table_name}")
        )
    count_after = rows[0][0]
    print(f"  Rows after deletion: {count_after}")

    assert count_after == 4, f"Expected 4 rows, got {count_after}"

    print("\n[OK] Delete test passed\n")


# ===================================================================
# Step 5: Full PathRAG integration (SpannerGraphStorage + SpannerVectorDBStorage)
# ===================================================================
async def test_step5_pathrag_full_spanner():
    """Test PathRAG end-to-end with both Spanner graph and vector storage."""
    import tempfile
    import shutil

    print("=" * 70)
    print("STEP 5: PathRAG Full Integration")
    print("       (SpannerGraphStorage + SpannerVectorDBStorage)")
    print("=" * 70)

    if not _check_env():
        return

    llm_model, embedding_model, embedding_dim, tiktoken_model = _get_llm_config()
    if llm_model is None:
        print("  [SKIP] No API key found (GEMINI_API_KEY or OPENAI_API_KEY).")
        return

    from PathRAG import PathRAG, QueryParam

    work_dir = tempfile.mkdtemp(prefix="pathrag_spanner_full_test_")
    try:
        print(f"\n  Working dir: {work_dir}")

        rag = PathRAG(
            working_dir=work_dir,
            llm_model_name=llm_model,
            embedding_model_name=embedding_model,
            embedding_dim=embedding_dim,
            tiktoken_model_name=tiktoken_model,
            graph_storage="SpannerGraphStorage",
            vector_storage="SpannerVectorDBStorage",
            addon_params={
                "spanner_instance_id": SPANNER_INSTANCE,
                "spanner_database_id": SPANNER_DATABASE,
            },
            chunk_token_size=200,
            chunk_overlap_token_size=50,
        )

        # Verify storage types
        graph = rag.chunk_entity_relation_graph
        entities_vdb = rag.entities_vdb
        relationships_vdb = rag.relationships_vdb
        chunks_vdb = rag.chunks_vdb

        print(f"\n  Graph storage type: {type(graph).__name__}")
        print(f"  Entities VDB type:  {type(entities_vdb).__name__}")
        print(f"  Relations VDB type: {type(relationships_vdb).__name__}")
        print(f"  Chunks VDB type:    {type(chunks_vdb).__name__}")

        # Index sample documents
        sample_docs = [
            """
            Apple Inc. is an American multinational technology company
            headquartered in Cupertino, California. Apple was founded on
            April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne.
            Tim Cook has been the CEO of Apple since August 2011.
            """,
            """
            Google LLC is an American multinational corporation focusing on
            search engine technology and cloud computing. Sundar Pichai has
            been the CEO of Google since October 2015. Google was founded
            on September 4, 1998, by Larry Page and Sergey Brin.
            """,
        ]

        print(f"\n--- Indexing {len(sample_docs)} documents ---")
        await rag.ainsert(sample_docs)
        print("  Indexing complete!")

        # Check graph storage
        nodes = await graph.nodes()
        edges_list = await graph.edges()
        print(f"\n  Extracted entities (graph): {len(nodes)}")
        print(f"  Extracted relationships (graph): {len(edges_list)}")

        print("\n--- Entities in Spanner Graph ---")
        for name in list(nodes)[:10]:
            data = await graph.get_node(name)
            if data:
                print(f"  {name}: type={data.get('entity_type', 'N/A')}")

        print("\n--- Relationships in Spanner Graph ---")
        for src, tgt in list(edges_list)[:10]:
            data = await graph.get_edge(src, tgt)
            if data:
                print(f"  {src} --[{data.get('keywords', 'N/A')}]--> {tgt}")

        # Check vector storage - entity search
        print("\n--- Entity Vector Search ---")
        entity_results = await entities_vdb.query("Apple technology company", top_k=5)
        print(f"  Query: 'Apple technology company'  ({len(entity_results)} results)")
        for r in entity_results:
            print(
                f"    entity={r.get('entity_name', 'N/A')}, "
                f"similarity={r['distance']:.4f}"
            )

        # Check vector storage - relationship search
        print("\n--- Relationship Vector Search ---")
        rel_results = await relationships_vdb.query("founded the company", top_k=5)
        print(f"  Query: 'founded the company'  ({len(rel_results)} results)")
        for r in rel_results:
            print(
                f"    src={r.get('src_id', 'N/A')} -> tgt={r.get('tgt_id', 'N/A')}, "
                f"similarity={r['distance']:.4f}"
            )

        # Full RAG query
        print("\n--- Full RAG Query ---")
        query = "What is the relationship between Steve Jobs and Apple?"
        print(f"  Q: {query}")
        response = await rag.aquery(
            query,
            param=QueryParam(mode="hybrid", top_k=10),
        )
        display = response[:500] + "..." if len(response) > 500 else response
        print(f"  A: {display}")

        # Second query
        print()
        query2 = "Who founded Google?"
        print(f"  Q: {query2}")
        response2 = await rag.aquery(
            query2,
            param=QueryParam(mode="hybrid", top_k=10),
        )
        display2 = response2[:500] + "..." if len(response2) > 500 else response2
        print(f"  A: {display2}")

        print("\n[OK] Full PathRAG Integration test passed\n")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ===================================================================
# Step 6: Cleanup
# ===================================================================
async def test_step6_cleanup():
    """Clean up all test tables from Spanner."""
    print("=" * 70)
    print("STEP 6: Cleanup Test Data")
    print("=" * 70)

    if not _check_env():
        return

    from google.cloud import spanner

    client = spanner.Client(disable_builtin_metrics=True)
    instance = client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)

    # Vector tables created during tests
    vector_tables = [
        f"vdb_{VECTOR_TEST_NAMESPACE}",
        "vdb_entities",
        "vdb_relationships",
        "vdb_chunks",
    ]

    # Graph resources (from PathRAG integration)
    graph_name = "pathrag_chunk_entity_relation"
    graph_node_table = "chunk_entity_relation_nodes"
    graph_edge_table = "chunk_entity_relation_edges"

    # Drop property graph first
    print(f"\n  Dropping property graph '{graph_name}'...")
    try:
        op = database.update_ddl([f"DROP PROPERTY GRAPH IF EXISTS {graph_name}"])
        op.result()
        print("    Done.")
    except Exception as e:
        print(f"    Skipped: {e}")

    # Drop graph edge table (FK dependency before node table)
    print(f"  Dropping graph edge table '{graph_edge_table}'...")
    try:
        op = database.update_ddl([f"DROP TABLE IF EXISTS {graph_edge_table}"])
        op.result()
        print("    Done.")
    except Exception as e:
        print(f"    Skipped: {e}")

    # Drop graph node table
    print(f"  Dropping graph node table '{graph_node_table}'...")
    try:
        op = database.update_ddl([f"DROP TABLE IF EXISTS {graph_node_table}"])
        op.result()
        print("    Done.")
    except Exception as e:
        print(f"    Skipped: {e}")

    # Drop vector tables
    for table in vector_tables:
        print(f"  Dropping vector table '{table}'...")
        try:
            op = database.update_ddl([f"DROP TABLE IF EXISTS {table}"])
            op.result()
            print("    Done.")
        except Exception as e:
            print(f"    Skipped: {e}")

    print("\n[OK] Cleanup completed\n")


# ===================================================================
# Main entry point
# ===================================================================
async def main():
    parser = argparse.ArgumentParser(
        description="SpannerVectorDBStorage + SpannerGraphStorage Integration Test Suite"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Run cleanup after tests to drop test tables",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only run cleanup (skip all tests)",
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Only run unit tests (steps 1-4, no LLM required)",
    )
    args = parser.parse_args()

    print()
    print("*" * 70)
    print("  SpannerVectorDBStorage + SpannerGraphStorage Test Suite")
    print("*" * 70)
    print()

    if args.cleanup_only:
        await test_step6_cleanup()
    elif args.unit_only:
        await test_step1_upsert()
        await test_step2_query()
        await test_step3_upsert_update()
        await test_step4_delete()
        if args.cleanup:
            await test_step6_cleanup()
    else:
        # Unit tests (Spanner only, no LLM)
        await test_step1_upsert()
        await test_step2_query()
        await test_step3_upsert_update()
        await test_step4_delete()

        # Full integration (Spanner + LLM)
        await test_step5_pathrag_full_spanner()

        if args.cleanup:
            await test_step6_cleanup()

    print("*" * 70)
    print("  All tests completed!")
    print("*" * 70)


if __name__ == "__main__":
    asyncio.run(main())
