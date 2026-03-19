"""
Spanner All-Storage Integration Test Suite

Tests PathRAG end-to-end with all three Spanner storage backends:
  - SpannerKVStorage       (key-value: full_docs, text_chunks, llm_response_cache)
  - SpannerVectorDBStorage (vector: entities, relationships, chunks)
  - SpannerGraphStorage    (graph: chunk_entity_relation)

Prerequisites:
  - Set GCP credentials (GOOGLE_APPLICATION_CREDENTIALS or gcloud auth)
  - Set Spanner config in examples/spanner/.env or as environment variables:
      SPANNER_INSTANCE=<instance-id>
      SPANNER_DATABASE=<database-id>
  - Set LLM API key: GEMINI_API_KEY or OPENAI_API_KEY
  - pip install -e .
  - pip install google-cloud-spanner

Usage:
  python examples/spanner/test_spanner_all_storage.py            # run tests (no cleanup)
  python examples/spanner/test_spanner_all_storage.py --cleanup  # run tests then cleanup
  python examples/spanner/test_spanner_all_storage.py --cleanup-only  # cleanup only
  python examples/spanner/test_spanner_all_storage.py --unit-only     # skip LLM integration
"""

import argparse
import asyncio
import json
import os
import tempfile
import shutil

import numpy as np
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

SPANNER_INSTANCE = os.environ.get("SPANNER_INSTANCE")
SPANNER_DATABASE = os.environ.get("SPANNER_DATABASE")

# Namespace prefix for integration test — avoids collision with individual tests
NS_PREFIX = "integ"


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


def _global_config(**overrides):
    """Build global_config dict for Spanner storage instances."""
    cfg = {
        "spanner_instance_id": SPANNER_INSTANCE,
        "spanner_database_id": SPANNER_DATABASE,
        "embedding_batch_num": 32,
    }
    cfg.update(overrides)
    return cfg


# ===================================================================
# Step 1: Individual storage smoke tests (no LLM required)
# ===================================================================
async def test_step1_kv_storage():
    """Smoke test SpannerKVStorage standalone."""
    print("=" * 70)
    print("STEP 1: SpannerKVStorage Smoke Test")
    print("=" * 70)

    if not _check_env():
        return

    from PathRAG.storage.spanner import SpannerKVStorage

    kv = SpannerKVStorage(
        namespace=f"{NS_PREFIX}_kv_smoke",
        global_config=_global_config(),
        embedding_func=_create_mock_embedding_func(),
    )

    # Upsert
    data = {
        "key-1": {"content": "Hello World", "score": 1.0},
        "key-2": {"content": "Foo Bar", "score": 2.0},
    }
    left = await kv.upsert(data)
    print(f"\n  Upserted: {list(left.keys())}")

    # Read back
    val = await kv.get_by_id("key-1")
    print(f"  get_by_id('key-1'): {json.dumps(val)}")
    assert val["content"] == "Hello World"

    vals = await kv.get_by_ids(["key-1", "key-2", "missing"], fields={"content"})
    print(f"  get_by_ids (content only): {[json.dumps(v) if v else None for v in vals]}")
    assert vals[2] is None

    # all_keys / filter_keys
    keys = await kv.all_keys()
    print(f"  all_keys: {keys}")
    assert set(keys) == {"key-1", "key-2"}

    missing = await kv.filter_keys(["key-1", "key-999"])
    print(f"  filter_keys(['key-1','key-999']): {missing}")
    assert missing == {"key-999"}

    # drop
    await kv.drop()
    assert len(await kv.all_keys()) == 0
    print("  drop: OK (0 keys)")

    print("\n[OK] KV Storage smoke test passed\n")


async def test_step2_vector_storage():
    """Smoke test SpannerVectorDBStorage standalone."""
    print("=" * 70)
    print("STEP 2: SpannerVectorDBStorage Smoke Test")
    print("=" * 70)

    if not _check_env():
        return

    from PathRAG.storage.spanner import SpannerVectorDBStorage

    embed_func = _create_mock_embedding_func(dim=128)
    vdb = SpannerVectorDBStorage(
        namespace=f"{NS_PREFIX}_vec_smoke",
        global_config=_global_config(),
        embedding_func=embed_func,
        meta_fields={"entity_name"},
    )

    test_data = {
        "v-001": {"content": "Apple is a technology company.", "entity_name": "APPLE"},
        "v-002": {"content": "Google is a search engine.", "entity_name": "GOOGLE"},
        "v-003": {"content": "Microsoft makes Windows.", "entity_name": "MICROSOFT"},
    }

    print(f"\n  Upserting {len(test_data)} vectors...")
    await vdb.upsert(test_data)

    results = await vdb.query("Apple technology", top_k=2)
    print(f"  Query 'Apple technology': {len(results)} results")
    for r in results:
        print(f"    {r['id']}: entity={r.get('entity_name','N/A')}, dist={r['distance']:.4f}")

    print("\n[OK] Vector Storage smoke test passed\n")


async def test_step3_graph_storage():
    """Smoke test SpannerGraphStorage standalone."""
    print("=" * 70)
    print("STEP 3: SpannerGraphStorage Smoke Test")
    print("=" * 70)

    if not _check_env():
        return

    from PathRAG.storage.spanner import SpannerGraphStorage

    graph = SpannerGraphStorage(
        namespace=f"{NS_PREFIX}_graph_smoke",
        global_config=_global_config(),
    )

    # Insert nodes & edge
    await graph.upsert_node("NODE_A", {"entity_type": "ORG", "description": "Org A"})
    await graph.upsert_node("NODE_B", {"entity_type": "PERSON", "description": "Person B"})
    await graph.upsert_edge("NODE_B", "NODE_A", {
        "weight": "1.5", "description": "works at", "keywords": "employment",
    })
    print(f"\n  Nodes: {await graph.nodes()}")
    print(f"  Edges: {await graph.edges()}")

    assert await graph.has_node("NODE_A")
    assert await graph.has_edge("NODE_B", "NODE_A")

    degree = await graph.node_degree("NODE_A")
    print(f"  NODE_A degree: {degree}")

    edge_data = await graph.get_edge("NODE_B", "NODE_A")
    print(f"  Edge B->A: {json.dumps(edge_data)}")

    print("\n[OK] Graph Storage smoke test passed\n")


# ===================================================================
# Step 4: Full PathRAG integration (all 3 Spanner backends + LLM)
# ===================================================================
async def test_step4_pathrag_full_integration():
    """PathRAG end-to-end with SpannerKVStorage + SpannerVectorDBStorage + SpannerGraphStorage."""
    print("=" * 70)
    print("STEP 4: Full PathRAG Integration")
    print("       (SpannerKVStorage + SpannerVectorDBStorage + SpannerGraphStorage)")
    print("=" * 70)

    if not _check_env():
        return

    llm_model, embedding_model, embedding_dim, tiktoken_model = _get_llm_config()
    if llm_model is None:
        print("  [SKIP] No API key found (GEMINI_API_KEY or OPENAI_API_KEY).")
        return

    from PathRAG import PathRAG, QueryParam

    work_dir = tempfile.mkdtemp(prefix="pathrag_spanner_integ_")
    try:
        print(f"\n  Working dir: {work_dir}")

        rag = PathRAG(
            working_dir=work_dir,
            llm_model_name=llm_model,
            embedding_model_name=embedding_model,
            embedding_dim=embedding_dim,
            tiktoken_model_name=tiktoken_model,
            kv_storage="SpannerKVStorage",
            vector_storage="SpannerVectorDBStorage",
            graph_storage="SpannerGraphStorage",
            addon_params={
                "spanner_instance_id": SPANNER_INSTANCE,
                "spanner_database_id": SPANNER_DATABASE,
            },
            chunk_token_size=200,
            chunk_overlap_token_size=50,
            # Disable LLM response cache. The cache uses "mode" as the key
            # and stores all cached responses as a nested JSON dict under that
            # key (see handle_cache / save_to_cache in utils.py). On every
            # cache read or write the entire dict must be fetched from the
            # KV store, updated in memory, and written back. This
            # read-modify-write pattern is fine for local JsonKVStorage but
            # adds unnecessary round-trips with a remote backend like Spanner.
            enable_llm_cache=False,
        )

        # --- Verify storage types ---
        print("\n--- Storage backends ---")
        print(f"  KV (full_docs):    {type(rag.full_docs).__name__}")
        print(f"  KV (text_chunks):  {type(rag.text_chunks).__name__}")
        print(f"  KV (llm_cache):    {type(rag.llm_response_cache).__name__}")
        print(f"  Vector (entities): {type(rag.entities_vdb).__name__}")
        print(f"  Vector (rels):     {type(rag.relationships_vdb).__name__}")
        print(f"  Vector (chunks):   {type(rag.chunks_vdb).__name__}")
        print(f"  Graph:             {type(rag.chunk_entity_relation_graph).__name__}")

        assert "Spanner" in type(rag.full_docs).__name__
        assert "Spanner" in type(rag.entities_vdb).__name__
        assert "Spanner" in type(rag.chunk_entity_relation_graph).__name__

        # --- Index documents ---
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

        # --- Verify KV storage ---
        print("\n--- KV Storage verification ---")
        full_doc_keys = await rag.full_docs.all_keys()
        text_chunk_keys = await rag.text_chunks.all_keys()
        print(f"  full_docs keys: {len(full_doc_keys)}")
        print(f"  text_chunks keys: {len(text_chunk_keys)}")
        assert len(full_doc_keys) > 0, "full_docs should have entries"
        assert len(text_chunk_keys) > 0, "text_chunks should have entries"

        # Read a text chunk to verify content
        if text_chunk_keys:
            sample_chunk = await rag.text_chunks.get_by_id(text_chunk_keys[0])
            print(f"  Sample chunk content: {sample_chunk.get('content', '')[:80]}...")

        # --- Verify Graph storage ---
        print("\n--- Graph Storage verification ---")
        graph = rag.chunk_entity_relation_graph
        nodes = await graph.nodes()
        edges_list = await graph.edges()
        print(f"  Entities: {len(nodes)}")
        print(f"  Relationships: {len(edges_list)}")

        print("\n  Entities:")
        for name in list(nodes)[:10]:
            data = await graph.get_node(name)
            if data:
                print(f"    {name}: type={data.get('entity_type', 'N/A')}")

        print("\n  Relationships:")
        for src, tgt in list(edges_list)[:10]:
            data = await graph.get_edge(src, tgt)
            if data:
                print(f"    {src} --[{data.get('keywords', 'N/A')}]--> {tgt}")

        # --- Verify Vector storage ---
        print("\n--- Vector Storage verification ---")
        entity_results = await rag.entities_vdb.query("Apple technology company", top_k=5)
        print(f"  Entity search 'Apple technology company': {len(entity_results)} results")
        for r in entity_results:
            print(f"    entity={r.get('entity_name','N/A')}, dist={r['distance']:.4f}")

        rel_results = await rag.relationships_vdb.query("founded the company", top_k=5)
        print(f"  Relationship search 'founded the company': {len(rel_results)} results")
        for r in rel_results:
            print(f"    {r.get('src_id','N/A')} -> {r.get('tgt_id','N/A')}, dist={r['distance']:.4f}")

        # --- Full RAG query ---
        print("\n--- RAG Queries ---")
        queries = [
            "What is the relationship between Steve Jobs and Apple?",
            "Who founded Google and when?",
            "Compare the leadership of Apple and Google.",
        ]
        for q in queries:
            print(f"\n  Q: {q}")
            response = await rag.aquery(
                q, param=QueryParam(mode="hybrid", top_k=10),
            )
            display = response[:300] + "..." if len(response) > 300 else response
            print(f"  A: {display}")

        print("\n[OK] Full PathRAG Integration test passed\n")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ===================================================================
# Step 5: Cleanup all test resources
# ===================================================================
async def test_step5_cleanup():
    """Drop all test tables and graphs created during integration tests."""
    print("=" * 70)
    print("STEP 5: Cleanup All Test Resources")
    print("=" * 70)

    if not _check_env():
        return

    from google.cloud import spanner

    client = spanner.Client(disable_builtin_metrics=True)
    instance = client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)

    # --- Property graphs (must be dropped before tables they reference) ---
    graphs_to_drop = [
        f"pathrag_{NS_PREFIX}_graph_smoke",
        "pathrag_chunk_entity_relation",
    ]
    for g in graphs_to_drop:
        print(f"\n  Dropping property graph '{g}'...")
        try:
            op = database.update_ddl([f"DROP PROPERTY GRAPH IF EXISTS {g}"])
            op.result()
            print("    Done.")
        except Exception as e:
            print(f"    Skipped: {e}")

    # --- Tables (order: edges before nodes due to FK constraints) ---
    tables_to_drop = [
        # Smoke test tables
        f"{NS_PREFIX}_kv_smoke_kv",
        f"vdb_{NS_PREFIX}_vec_smoke",
        f"{NS_PREFIX}_graph_smoke_edges",
        f"{NS_PREFIX}_graph_smoke_nodes",
        # PathRAG integration tables — KV
        "full_docs_kv",
        "text_chunks_kv",
        "llm_response_cache_kv",
        # PathRAG integration tables — Vector
        "vdb_entities",
        "vdb_relationships",
        "vdb_chunks",
        # PathRAG integration tables — Graph
        "chunk_entity_relation_edges",
        "chunk_entity_relation_nodes",
    ]
    for table in tables_to_drop:
        print(f"  Dropping table '{table}'...")
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
        description="Spanner All-Storage Integration Test Suite"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Run cleanup after tests to drop all test resources",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only run cleanup (skip all tests)",
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Only run unit tests (steps 1-3, no LLM required)",
    )
    args = parser.parse_args()

    print()
    print("*" * 70)
    print("  Spanner All-Storage Integration Test Suite")
    print("  (SpannerKVStorage + SpannerVectorDBStorage + SpannerGraphStorage)")
    print("*" * 70)
    print()

    if args.cleanup_only:
        await test_step5_cleanup()
    elif args.unit_only:
        await test_step1_kv_storage()
        await test_step2_vector_storage()
        await test_step3_graph_storage()
        if args.cleanup:
            await test_step5_cleanup()
    else:
        # Unit tests (Spanner only, no LLM)
        await test_step1_kv_storage()
        await test_step2_vector_storage()
        await test_step3_graph_storage()

        # Full integration (Spanner + LLM)
        await test_step4_pathrag_full_integration()

        if args.cleanup:
            await test_step5_cleanup()

    print("*" * 70)
    print("  All tests completed!")
    print("*" * 70)


if __name__ == "__main__":
    asyncio.run(main())
