"""
SpannerGraphStorage Test Suite

Tests CRUD operations and GQL graph traversal on SpannerGraphStorage.

Prerequisites:
  - Set GCP credentials (GOOGLE_APPLICATION_CREDENTIALS or gcloud auth)
  - Set Spanner config in examples/spanner/.env or as environment variables:
      SPANNER_INSTANCE=<instance-id>
      SPANNER_DATABASE=<database-id>
  - pip install -e .
  - pip install google-cloud-spanner

Usage:
  python examples/spanner/test_spanner_graph_storage.py
"""

import asyncio
import json
import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


SPANNER_INSTANCE = os.environ.get("SPANNER_INSTANCE")
SPANNER_DATABASE = os.environ.get("SPANNER_DATABASE")


def _check_env():
    if not SPANNER_INSTANCE or not SPANNER_DATABASE:
        print("[SKIP] SPANNER_INSTANCE and SPANNER_DATABASE must be set.")
        print("  export SPANNER_INSTANCE=<your-instance-id>")
        print("  export SPANNER_DATABASE=<your-database-id>")
        return False
    print(f"  Spanner Instance: {SPANNER_INSTANCE}")
    print(f"  Spanner Database: {SPANNER_DATABASE}")
    return True


def _create_storage(namespace: str = "test"):
    """Create a SpannerGraphStorage instance for testing."""
    from PathRAG.spanner_graph_storage import SpannerGraphStorage

    return SpannerGraphStorage(
        namespace=namespace,
        global_config={
            "spanner_instance_id": SPANNER_INSTANCE,
            "spanner_database_id": SPANNER_DATABASE,
        },
    )


# ===================================================================
# Step 1: Schema creation & basic node CRUD
# ===================================================================
async def test_step1_node_crud():
    """Test node insert, read, update, and delete."""
    print("=" * 70)
    print("STEP 1: Node CRUD Operations")
    print("=" * 70)

    if not _check_env():
        return

    graph = _create_storage("test_crud")

    # --- Insert nodes ---
    print("\n--- Inserting nodes ---")
    await graph.upsert_node(
        "APPLE",
        {
            "entity_type": "ORGANIZATION",
            "description": "American multinational technology company",
            "source_id": "chunk-001",
        },
    )
    await graph.upsert_node(
        "STEVE JOBS",
        {
            "entity_type": "PERSON",
            "description": "Co-founder of Apple Inc.",
            "source_id": "chunk-001",
        },
    )
    await graph.upsert_node(
        "TIM COOK",
        {
            "entity_type": "PERSON",
            "description": "Current CEO of Apple Inc.",
            "source_id": "chunk-002",
        },
    )
    print("  Inserted 3 nodes: APPLE, STEVE JOBS, TIM COOK")

    # --- Read nodes ---
    print("\n--- Reading nodes ---")
    for name in ["APPLE", "STEVE JOBS", "TIM COOK"]:
        exists = await graph.has_node(name)
        data = await graph.get_node(name)
        print(f"  {name}: exists={exists}, data={json.dumps(data)}")

    # --- Check non-existent node ---
    exists = await graph.has_node("NON_EXISTENT")
    data = await graph.get_node("NON_EXISTENT")
    print(f"  NON_EXISTENT: exists={exists}, data={data}")

    # --- List all nodes ---
    all_nodes = await graph.nodes()
    print(f"\n--- All nodes ({len(all_nodes)}) ---")
    for n in all_nodes:
        print(f"  {n}")

    # --- Update node ---
    print("\n--- Updating APPLE description ---")
    await graph.upsert_node(
        "APPLE",
        {
            "entity_type": "ORGANIZATION",
            "description": "Multinational tech company headquartered in Cupertino, CA",
            "source_id": "chunk-001",
        },
    )
    updated = await graph.get_node("APPLE")
    print(f"  Updated: {json.dumps(updated)}")

    print("\n[OK] Node CRUD test passed\n")


# ===================================================================
# Step 2: Edge CRUD & GQL traversal
# ===================================================================
async def test_step2_edge_crud():
    """Test edge insert, read, and GQL-based graph traversal."""
    print("=" * 70)
    print("STEP 2: Edge CRUD & GQL Traversal")
    print("=" * 70)

    if not _check_env():
        return

    graph = _create_storage("test_crud")

    # --- Insert edges ---
    print("\n--- Inserting edges ---")
    await graph.upsert_edge(
        "STEVE JOBS",
        "APPLE",
        {
            "weight": "3.0",
            "description": "Co-founded Apple in 1976",
            "keywords": "founding, co-founder",
            "source_id": "chunk-001",
        },
    )
    await graph.upsert_edge(
        "TIM COOK",
        "APPLE",
        {
            "weight": "2.5",
            "description": "CEO of Apple since 2011",
            "keywords": "CEO, leadership",
            "source_id": "chunk-002",
        },
    )
    print("  Inserted 2 edges: STEVE JOBS->APPLE, TIM COOK->APPLE")

    # --- Read edges (GQL) ---
    print("\n--- Reading edges (via GQL) ---")
    edge1 = await graph.get_edge("STEVE JOBS", "APPLE")
    print(f"  STEVE JOBS -> APPLE: {json.dumps(edge1)}")

    edge2 = await graph.get_edge("TIM COOK", "APPLE")
    print(f"  TIM COOK -> APPLE: {json.dumps(edge2)}")

    # --- Check edge existence (GQL) ---
    print("\n--- Edge existence (via GQL) ---")
    print(f"  STEVE JOBS -> APPLE: {await graph.has_edge('STEVE JOBS', 'APPLE')}")
    print(f"  APPLE -> STEVE JOBS: {await graph.has_edge('APPLE', 'STEVE JOBS')}")
    print(f"  TIM COOK -> STEVE JOBS: {await graph.has_edge('TIM COOK', 'STEVE JOBS')}")

    # --- List all edges ---
    all_edges = await graph.edges()
    print(f"\n--- All edges ({len(all_edges)}) ---")
    for src, tgt in all_edges:
        print(f"  {src} -> {tgt}")

    print("\n[OK] Edge CRUD test passed\n")


# ===================================================================
# Step 3: Graph traversal (degree, in/out edges)
# ===================================================================
async def test_step3_graph_traversal():
    """Test GQL-based degree and edge traversal operations."""
    print("=" * 70)
    print("STEP 3: Graph Traversal (GQL)")
    print("=" * 70)

    if not _check_env():
        return

    graph = _create_storage("test_crud")

    # --- Node degree (via GQL) ---
    print("\n--- Node degree (via GQL) ---")
    for name in ["APPLE", "STEVE JOBS", "TIM COOK"]:
        degree = await graph.node_degree(name)
        print(f"  {name}: degree={degree}")

    # --- Edge degree ---
    print("\n--- Edge degree ---")
    edge_deg = await graph.edge_degree("STEVE JOBS", "APPLE")
    print(f"  STEVE JOBS <-> APPLE: edge_degree={edge_deg}")

    # --- Outgoing edges (GQL) ---
    print("\n--- Outgoing edges (via GQL) ---")
    for name in ["STEVE JOBS", "TIM COOK", "APPLE"]:
        out_edges = await graph.get_node_edges(name)
        print(f"  {name} -> {out_edges}")

    # --- Incoming edges (GQL) ---
    print("\n--- Incoming edges (via GQL) ---")
    for name in ["APPLE", "STEVE JOBS"]:
        in_edges = await graph.get_node_in_edges(name)
        print(f"  {name} <- {in_edges}")

    # --- PageRank approximation ---
    print("\n--- PageRank (normalised degree) ---")
    for name in ["APPLE", "STEVE JOBS", "TIM COOK"]:
        pr = await graph.get_pagerank(name)
        print(f"  {name}: pagerank={pr:.4f}")

    print("\n[OK] Graph Traversal test passed\n")


# ===================================================================
# Step 4: Node deletion (cascading edge removal)
# ===================================================================
async def test_step4_node_deletion():
    """Test node deletion with cascading edge removal."""
    print("=" * 70)
    print("STEP 4: Node Deletion (Cascade)")
    print("=" * 70)

    if not _check_env():
        return

    graph = _create_storage("test_crud")

    nodes_before = await graph.nodes()
    edges_before = await graph.edges()
    print(f"\nBefore deletion:")
    print(f"  Nodes ({len(nodes_before)}): {nodes_before}")
    print(f"  Edges ({len(edges_before)}): {edges_before}")

    # Delete STEVE JOBS — should also remove the edge STEVE JOBS->APPLE
    print("\n--- Deleting 'STEVE JOBS' ---")
    await graph.delete_node("STEVE JOBS")

    nodes_after = await graph.nodes()
    edges_after = await graph.edges()
    print(f"\nAfter deletion:")
    print(f"  Nodes ({len(nodes_after)}): {nodes_after}")
    print(f"  Edges ({len(edges_after)}): {edges_after}")

    exists = await graph.has_node("STEVE JOBS")
    print(f"\n  STEVE JOBS exists: {exists}")

    print("\n[OK] Node Deletion test passed\n")


# ===================================================================
# Step 5: Full PathRAG integration with SpannerGraphStorage
# ===================================================================
async def test_step5_pathrag_integration():
    """Test PathRAG end-to-end with SpannerGraphStorage as graph backend."""
    from PathRAG import PathRAG, QueryParam
    from PathRAG.utils import EmbeddingFunc
    import numpy as np
    import tempfile
    import shutil

    print("=" * 70)
    print("STEP 5: PathRAG Integration with SpannerGraphStorage")
    print("=" * 70)

    if not _check_env():
        return

    llm_model, embedding_model, embedding_dim, tiktoken_model = _get_llm_config()
    if llm_model is None:
        print("  [SKIP] No API key found (GEMINI_API_KEY or OPENAI_API_KEY).")
        return

    work_dir = tempfile.mkdtemp(prefix="pathrag_spanner_test_")
    try:
        rag = PathRAG(
            working_dir=work_dir,
            llm_model_name=llm_model,
            embedding_model_name=embedding_model,
            embedding_dim=embedding_dim,
            tiktoken_model_name=tiktoken_model,
            graph_storage="SpannerGraphStorage",
            addon_params={
                "spanner_instance_id": SPANNER_INSTANCE,
                "spanner_database_id": SPANNER_DATABASE,
            },
            chunk_token_size=200,
            chunk_overlap_token_size=50,
        )

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

        graph = rag.chunk_entity_relation_graph
        nodes = await graph.nodes()
        edges_list = await graph.edges()
        print(f"\n  Extracted entities: {len(nodes)}")
        print(f"  Extracted relationships: {len(edges_list)}")

        print("\n--- Entities ---")
        for name in list(nodes)[:10]:
            data = await graph.get_node(name)
            if data:
                print(f"  {name}: type={data.get('entity_type', 'N/A')}")

        print("\n--- Relationships ---")
        for src, tgt in list(edges_list)[:10]:
            data = await graph.get_edge(src, tgt)
            if data:
                print(f"  {src} --[{data.get('keywords', 'N/A')}]--> {tgt}")

        # Query
        print("\n--- Query ---")
        query = "What is the relationship between Steve Jobs and Apple?"
        response = await rag.aquery(
            query,
            param=QueryParam(mode="hybrid", top_k=10),
        )
        print(f"  Q: {query}")
        display = response[:300] + "..." if len(response) > 300 else response
        print(f"  A: {display}")

        print("\n[OK] PathRAG Integration test passed\n")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ===================================================================
# Step 6: Cleanup test data
# ===================================================================
async def test_step6_cleanup():
    """Clean up test tables and graph from Spanner."""
    print("=" * 70)
    print("STEP 6: Cleanup Test Data")
    print("=" * 70)

    if not _check_env():
        return

    from google.cloud import spanner

    client = spanner.Client()
    instance = client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)

    graph_name = "pathrag_test_crud"
    node_table = "test_crud_nodes"
    edge_table = "test_crud_edges"

    ddls = [
        f"DROP PROPERTY GRAPH IF EXISTS {graph_name}",
    ]
    print(f"\n  Dropping property graph '{graph_name}'...")
    op = database.update_ddl(ddls)
    op.result()

    ddls = [
        f"DROP TABLE IF EXISTS {edge_table}",
    ]
    print(f"  Dropping edge table '{edge_table}'...")
    op = database.update_ddl(ddls)
    op.result()

    ddls = [
        f"DROP TABLE IF EXISTS {node_table}",
    ]
    print(f"  Dropping node table '{node_table}'...")
    op = database.update_ddl(ddls)
    op.result()

    print("\n[OK] Cleanup completed\n")


# ===================================================================
# Helper
# ===================================================================
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


# ===================================================================
# Main entry point
# ===================================================================
async def main():
    print()
    print("*" * 70)
    print("  SpannerGraphStorage Test Suite")
    print("*" * 70)
    print()

    # CRUD tests (Spanner required, no LLM)
    await test_step1_node_crud()
    await test_step2_edge_crud()
    await test_step3_graph_traversal()
    await test_step4_node_deletion()

    # Full PathRAG integration (Spanner + LLM required)
    await test_step5_pathrag_integration()

    # Cleanup test tables
    await test_step6_cleanup()

    print("*" * 70)
    print("  All SpannerGraphStorage tests completed!")
    print("*" * 70)


if __name__ == "__main__":
    asyncio.run(main())
