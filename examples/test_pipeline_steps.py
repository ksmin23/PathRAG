"""
PathRAG Pipeline Step-by-Step Test Suite

Tests each stage of the PathRAG pipeline as described in the
PathRAG_Technical_Specification.md. Each step can be run independently.

Prerequisites:
  - Set API keys in examples/.env (or export them as environment variables)
  - pip install -e .  (install PathRAG package)

Usage:
  python examples/test_pipeline_steps.py
"""

import asyncio
import os
import json
import shutil
import tempfile
from collections import defaultdict
import networkx as nx
from dotenv import load_dotenv, find_dotenv

# Auto-discover and load .env file (does not override existing env vars)
load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------

def _get_llm_config():
    """Resolve LLM / embedding configuration from environment variables.

    If LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, or EMBEDDING_DIM are set in the
    environment (or .env file), those values take precedence.  Otherwise,
    sensible defaults are chosen based on which API key is available.

    Returns:
        (llm_model, embedding_model, embedding_dim, tiktoken_model)
        tiktoken_model is derived from LLM_MODEL_NAME and passed to
        tiktoken for token counting.  Falls back to cl100k_base encoding
        when the model is not recognized by tiktoken (e.g. Gemini,
        Bedrock, Ollama models).
    """
    # Choose defaults based on available API key
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

    # Use LLM_MODEL_NAME as the tiktoken model name.
    # tiktoken recognises OpenAI model names (e.g. gpt-4o-mini) directly.
    # For non-OpenAI models (e.g. gemini/gemini-2.5-flash),
    # _get_tiktoken_encoder() falls back to cl100k_base automatically.
    tiktoken_model = llm_model

    print(f"  LLM Model: {llm_model}")
    print(f"  Embedding Model: {embedding_model} (dim={embedding_dim})")
    print(f"  Tiktoken Model: {tiktoken_model}")
    return llm_model, embedding_model, embedding_dim, tiktoken_model


SAMPLE_DOCUMENTS = [
    """
    Apple Inc. is an American multinational technology company headquartered in
    Cupertino, California. Apple was founded on April 1, 1976, by Steve Jobs,
    Steve Wozniak, and Ronald Wayne. The company designs, develops, and sells
    consumer electronics, computer software, and online services.
    Tim Cook has been the CEO of Apple since August 2011, succeeding Steve Jobs.
    Under Cook's leadership, Apple launched Apple Watch, AirPods, and Apple Vision Pro.
    """,
    """
    Steve Jobs was an American business magnate, inventor, and investor. He was
    the co-founder, chairman, and CEO of Apple Inc. Jobs also co-founded and
    served as the chairman of Pixar Animation Studios. He was a board member at
    The Walt Disney Company following the acquisition of Pixar by Disney.
    Jobs is widely recognized as a pioneer of the personal computer revolution
    and for his influential career in the computer and consumer electronics fields.
    """,
    """
    Google LLC is an American multinational corporation and technology company
    focusing on online advertising, search engine technology, cloud computing,
    and artificial intelligence. Sundar Pichai has been the CEO of Google since
    October 2015 and of its parent company Alphabet Inc. since December 2019.
    Google was founded on September 4, 1998, by Larry Page and Sergey Brin
    while they were Ph.D. students at Stanford University in California.
    """,
]


# ===================================================================
# Step 1: Chunking test
# ===================================================================
def test_step1_chunking():
    """Test token-based document chunking."""
    from PathRAG.operate import chunking_by_token_size

    print("=" * 70)
    print("STEP 1: Chunking")
    print("=" * 70)

    # Use LLM_MODEL_NAME for tiktoken token counting.
    # Falls back to cl100k_base for non-OpenAI models (e.g. Gemini).
    tiktoken_model = os.environ.get("LLM_MODEL_NAME", "gpt-4o")

    for i, doc in enumerate(SAMPLE_DOCUMENTS):
        chunks = chunking_by_token_size(
            doc.strip(),
            overlap_token_size=50,
            max_token_size=200,
            tiktoken_model=tiktoken_model,
        )
        print(f"\n--- Document {i+1} ---")
        print(f"  Original length: {len(doc.strip())} chars")
        print(f"  Chunks created: {len(chunks)}")
        for j, chunk in enumerate(chunks):
            print(f"  Chunk {j}: tokens={chunk['tokens']}, "
                  f"order={chunk['chunk_order_index']}, "
                  f"content={chunk['content'][:80]}...")

    print("\n[OK] Chunking test passed\n")


# ===================================================================
# Step 2: Path Finding algorithm test (no LLM required)
# ===================================================================
async def test_step2_path_finding():
    """Test multi-hop path finding on a knowledge graph.

    Runs entirely on a local NetworkX graph — no LLM calls needed.
    """
    from PathRAG.operate import find_paths_and_edges_with_stats, bfs_weighted_paths

    print("=" * 70)
    print("STEP 2: Path Finding & Pruning")
    print("=" * 70)

    # Build a sample knowledge graph
    G = nx.Graph()
    edges = [
        ("STEVE JOBS", "APPLE"),
        ("STEVE JOBS", "PIXAR"),
        ("APPLE", "TIM COOK"),
        ("APPLE", "CUPERTINO"),
        ("PIXAR", "DISNEY"),
        ("DISNEY", "ENTERTAINMENT"),
        ("TIM COOK", "APPLE WATCH"),
        ("GOOGLE", "SUNDAR PICHAI"),
        ("GOOGLE", "ALPHABET"),
        ("GOOGLE", "STANFORD UNIVERSITY"),
        ("LARRY PAGE", "GOOGLE"),
        ("SERGEY BRIN", "GOOGLE"),
        ("LARRY PAGE", "STANFORD UNIVERSITY"),
    ]
    G.add_edges_from(edges)

    print(f"\nGraph nodes: {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")

    # Simulate entities retrieved by vector search
    target_nodes = ["STEVE JOBS", "TIM COOK", "APPLE", "PIXAR"]

    print(f"\nTarget nodes (simulated vector search results): {target_nodes}")

    # 2-1. DFS-based path discovery
    result, path_stats, one_hop, two_hop, three_hop = (
        await find_paths_and_edges_with_stats(G, target_nodes)
    )

    print(f"\n--- Path Statistics ---")
    print(f"  1-hop paths: {path_stats['1-hop']}")
    print(f"  2-hop paths: {path_stats['2-hop']}")
    print(f"  3-hop paths: {path_stats['3-hop']}")

    print(f"\n--- 1-hop Paths ---")
    for p in one_hop[:5]:
        print(f"  {' -> '.join(p)}")

    print(f"\n--- 2-hop Paths ---")
    for p in two_hop[:5]:
        print(f"  {' -> '.join(p)}")

    print(f"\n--- 3-hop Paths ---")
    for p in three_hop[:5]:
        print(f"  {' -> '.join(p)}")

    # 2-2. BFS-based path weighting and pruning
    print(f"\n--- BFS Weighted Path Pruning ---")
    threshold = 0.3
    alpha = 0.8

    for (src, tgt), data in list(result.items())[:3]:
        paths = data["paths"]
        if not paths:
            continue
        weighted = bfs_weighted_paths(G, paths, src, tgt, threshold, alpha)
        print(f"\n  {src} -> {tgt}: ({len(paths)} path(s))")
        for path, weight in sorted(weighted, key=lambda x: x[1], reverse=True)[:5]:
            status = "KEEP" if weight > threshold else "PRUNE"
            print(f"    [{status}] weight={weight:.3f}  {' -> '.join(path)}")

    print("\n[OK] Path Finding test passed\n")


# ===================================================================
# Step 3: Custom KG insertion test (no LLM required)
# ===================================================================
async def test_step3_custom_kg_insert():
    """Test inserting a manually constructed knowledge graph into PathRAG storage.

    Uses a dummy embedding function — no LLM calls needed.
    """
    from PathRAG import PathRAG, QueryParam
    from PathRAG.utils import EmbeddingFunc
    import numpy as np

    print("=" * 70)
    print("STEP 3: Custom KG Insert")
    print("=" * 70)

    work_dir = tempfile.mkdtemp(prefix="pathrag_test_")

    # Dummy embedding function (no LLM calls)
    async def dummy_embedding(texts: list[str]) -> np.ndarray:
        return np.random.randn(len(texts), 384).astype(np.float32)

    try:
        rag = PathRAG(
            working_dir=work_dir,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=dummy_embedding,
            ),
            embedding_dim=384,
        )

        custom_kg = {
            "chunks": [
                {
                    "content": "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
                    "source_id": "chunk-apple-founding",
                },
                {
                    "content": "Steve Jobs co-founded Pixar and served as its chairman until Disney acquired it.",
                    "source_id": "chunk-jobs-pixar",
                },
            ],
            "entities": [
                {
                    "entity_name": "Apple Inc.",
                    "entity_type": "ORGANIZATION",
                    "description": "American multinational technology company",
                    "source_id": "chunk-apple-founding",
                },
                {
                    "entity_name": "Steve Jobs",
                    "entity_type": "PERSON",
                    "description": "Co-founder and former CEO of Apple Inc.",
                    "source_id": "chunk-apple-founding",
                },
                {
                    "entity_name": "Pixar",
                    "entity_type": "ORGANIZATION",
                    "description": "Animation studio co-founded by Steve Jobs",
                    "source_id": "chunk-jobs-pixar",
                },
            ],
            "relationships": [
                {
                    "src_id": "Steve Jobs",
                    "tgt_id": "Apple Inc.",
                    "description": "Steve Jobs co-founded Apple Inc. in 1976",
                    "keywords": "founding, co-founder, leadership",
                    "weight": 3.0,
                    "source_id": "chunk-apple-founding",
                },
                {
                    "src_id": "Steve Jobs",
                    "tgt_id": "Pixar",
                    "description": "Steve Jobs co-founded Pixar Animation Studios",
                    "keywords": "co-founder, animation, chairman",
                    "weight": 2.0,
                    "source_id": "chunk-jobs-pixar",
                },
            ],
        }

        await rag.ainsert_custom_kg(custom_kg)

        # Verify insertion results
        graph = rag.chunk_entity_relation_graph
        nodes = await graph.nodes()
        edges_list = await graph.edges()
        print(f"\nInserted nodes: {len(nodes)}")
        print(f"Inserted edges: {len(edges_list)}")

        print("\n--- Nodes ---")
        for node_name in nodes:
            node_data = await graph.get_node(node_name)
            print(f"  {node_name}: type={node_data.get('entity_type', 'N/A')}, "
                  f"desc={node_data.get('description', 'N/A')[:60]}")

        print("\n--- Edges ---")
        for src, tgt in edges_list:
            edge_data = await graph.get_edge(src, tgt)
            print(f"  {src} -> {tgt}: "
                  f"weight={edge_data.get('weight', 'N/A')}, "
                  f"keywords={edge_data.get('keywords', 'N/A')}")

        print("\n[OK] Custom KG Insert test passed\n")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ===================================================================
# Step 4: Full Indexing Pipeline test (LLM required)
# ===================================================================
async def test_step4_full_indexing():
    """Test the full indexing pipeline: document -> chunking -> entity extraction -> KG.

    Requires a valid LLM API key.
    """
    from PathRAG import PathRAG, QueryParam

    print("=" * 70)
    print("STEP 4: Full Indexing Pipeline")
    print("=" * 70)

    llm_model, embedding_model, embedding_dim, tiktoken_model = _get_llm_config()
    if llm_model is None:
        print("  [SKIP] No API key found (GEMINI_API_KEY or OPENAI_API_KEY).")
        print("  Set an API key to run this test.")
        return

    work_dir = tempfile.mkdtemp(prefix="pathrag_full_test_")
    try:
        rag = PathRAG(
            working_dir=work_dir,
            llm_model_name=llm_model,
            embedding_model_name=embedding_model,
            embedding_dim=embedding_dim,
            tiktoken_model_name=tiktoken_model,
            chunk_token_size=200,
            chunk_overlap_token_size=50,
        )

        # Index only the first document (for speed)
        print(f"\nIndexing document...")
        await rag.ainsert(SAMPLE_DOCUMENTS[0])
        print(f"Indexing complete!")

        # Inspect results
        graph = rag.chunk_entity_relation_graph
        nodes = await graph.nodes()
        edges_list = await graph.edges()
        print(f"\nExtracted entities: {len(nodes)}")
        print(f"Extracted relationships: {len(edges_list)}")

        print("\n--- Extracted Entities ---")
        for node_name in list(nodes)[:10]:
            node_data = await graph.get_node(node_name)
            if node_data:
                print(f"  {node_name}: type={node_data.get('entity_type', 'N/A')}")

        print("\n--- Extracted Relationships ---")
        for src, tgt in list(edges_list)[:10]:
            edge_data = await graph.get_edge(src, tgt)
            if edge_data:
                print(f"  {src} --[{edge_data.get('keywords', 'N/A')}]--> {tgt}")

        print("\n[OK] Full Indexing test passed\n")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ===================================================================
# Step 5: Full Query Pipeline test (LLM required)
# ===================================================================
async def test_step5_full_query():
    """Test the end-to-end query pipeline.

    Keyword Extraction -> Dual Context Retrieval -> Path Finding -> LLM Response
    """
    from PathRAG import PathRAG, QueryParam

    print("=" * 70)
    print("STEP 5: Full Query Pipeline")
    print("=" * 70)

    llm_model, embedding_model, embedding_dim, tiktoken_model = _get_llm_config()
    if llm_model is None:
        print("  [SKIP] No API key found. Set an API key to run this test.")
        return

    work_dir = tempfile.mkdtemp(prefix="pathrag_query_test_")
    try:
        rag = PathRAG(
            working_dir=work_dir,
            llm_model_name=llm_model,
            embedding_model_name=embedding_model,
            embedding_dim=embedding_dim,
            tiktoken_model_name=tiktoken_model,
            chunk_token_size=300,
            chunk_overlap_token_size=50,
        )

        # Index all documents
        print(f"\nIndexing {len(SAMPLE_DOCUMENTS)} documents...")
        await rag.ainsert(SAMPLE_DOCUMENTS)
        print("Indexing complete!")

        # --- 5-1: Retrieve context only (no final LLM response) ---
        print("\n--- 5-1: Context retrieval (only_need_context=True) ---")
        query = "What is the relationship between Steve Jobs and Apple?"
        context = await rag.aquery(
            query,
            param=QueryParam(
                mode="hybrid",
                only_need_context=True,
                top_k=10,
            ),
        )
        print(f"  Query: {query}")
        print(f"  Context length: {len(context)} chars")
        print(f"  Context (first 500 chars):\n{context[:500]}")

        # --- 5-2: Inspect final prompt ---
        print("\n--- 5-2: Final prompt inspection (only_need_prompt=True) ---")
        prompt = await rag.aquery(
            query,
            param=QueryParam(
                mode="hybrid",
                only_need_prompt=True,
                top_k=10,
            ),
        )
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  Prompt (first 500 chars):\n{prompt[:500]}")

        # --- 5-3: Full response generation ---
        print("\n--- 5-3: Full response generation ---")
        queries = [
            "What is the relationship between Steve Jobs and Apple?",
            "Who founded Google and what is their background?",
            "Compare the leadership of Apple and Google.",
        ]
        for q in queries:
            print(f"\n  Q: {q}")
            response = await rag.aquery(
                q,
                param=QueryParam(mode="hybrid", top_k=10),
            )
            # Truncate response for display
            display = response[:300] + "..." if len(response) > 300 else response
            print(f"  A: {display}")

        print("\n[OK] Full Query Pipeline test passed\n")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ===================================================================
# Step 6: Graph Storage direct manipulation test (no LLM required)
# ===================================================================
async def test_step6_graph_storage():
    """Test CRUD operations on NetworkXStorage directly."""
    from PathRAG.storage import NetworkXStorage
    from PathRAG.utils import EmbeddingFunc
    import numpy as np

    print("=" * 70)
    print("STEP 6: Graph Storage Operations")
    print("=" * 70)

    work_dir = tempfile.mkdtemp(prefix="pathrag_graph_test_")

    async def dummy_embedding(texts: list[str]) -> np.ndarray:
        return np.random.randn(len(texts), 384).astype(np.float32)

    try:
        graph = NetworkXStorage(
            namespace="test_graph",
            global_config={"working_dir": work_dir},
            embedding_func=EmbeddingFunc(
                embedding_dim=384, max_token_size=8192, func=dummy_embedding
            ),
        )

        # Insert nodes
        await graph.upsert_node(
            '"APPLE"',
            node_data={
                "entity_type": "ORGANIZATION",
                "description": "American multinational technology company",
                "source_id": "chunk-001",
            },
        )
        await graph.upsert_node(
            '"STEVE JOBS"',
            node_data={
                "entity_type": "PERSON",
                "description": "Co-founder of Apple Inc.",
                "source_id": "chunk-001",
            },
        )
        await graph.upsert_node(
            '"TIM COOK"',
            node_data={
                "entity_type": "PERSON",
                "description": "Current CEO of Apple Inc.",
                "source_id": "chunk-002",
            },
        )

        # Insert edges
        await graph.upsert_edge(
            '"STEVE JOBS"',
            '"APPLE"',
            edge_data={
                "weight": 3.0,
                "description": "Co-founded Apple in 1976",
                "keywords": "founding, co-founder",
                "source_id": "chunk-001",
            },
        )
        await graph.upsert_edge(
            '"TIM COOK"',
            '"APPLE"',
            edge_data={
                "weight": 2.5,
                "description": "CEO of Apple since 2011",
                "keywords": "CEO, leadership",
                "source_id": "chunk-002",
            },
        )

        # Query the graph
        print(f"\nNode count: {len(await graph.nodes())}")
        print(f"Edge count: {len(await graph.edges())}")

        node = await graph.get_node('"APPLE"')
        print(f'\n"APPLE" node: {json.dumps(node, indent=2)}')

        edge = await graph.get_edge('"STEVE JOBS"', '"APPLE"')
        print(f'\n"STEVE JOBS" -> "APPLE" edge: {json.dumps(edge, indent=2)}')

        degree = await graph.node_degree('"APPLE"')
        print(f'\n"APPLE" node degree: {degree}')

        node_edges = await graph.get_node_edges('"APPLE"')
        print(f'"APPLE" connected edges: {node_edges}')

        # Persist graph to disk
        await graph.index_done_callback()
        print(f"\nGraph saved to: {work_dir}")

        print("\n[OK] Graph Storage test passed\n")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ===================================================================
# Step 7: Entity deletion test (no LLM required)
# ===================================================================
async def test_step7_entity_deletion():
    """Test deleting a specific entity from the knowledge graph."""
    from PathRAG import PathRAG
    from PathRAG.utils import EmbeddingFunc
    import numpy as np

    print("=" * 70)
    print("STEP 7: Entity Deletion")
    print("=" * 70)

    work_dir = tempfile.mkdtemp(prefix="pathrag_delete_test_")

    async def dummy_embedding(texts: list[str]) -> np.ndarray:
        return np.random.randn(len(texts), 384).astype(np.float32)

    try:
        rag = PathRAG(
            working_dir=work_dir,
            embedding_func=EmbeddingFunc(
                embedding_dim=384, max_token_size=8192, func=dummy_embedding
            ),
            embedding_dim=384,
        )

        custom_kg = {
            "chunks": [
                {"content": "Test chunk content.", "source_id": "src-1"},
            ],
            "entities": [
                {
                    "entity_name": "EntityA",
                    "entity_type": "TEST",
                    "description": "Test entity A",
                    "source_id": "src-1",
                },
                {
                    "entity_name": "EntityB",
                    "entity_type": "TEST",
                    "description": "Test entity B",
                    "source_id": "src-1",
                },
            ],
            "relationships": [
                {
                    "src_id": "EntityA",
                    "tgt_id": "EntityB",
                    "description": "A is related to B",
                    "keywords": "test relationship",
                    "weight": 1.0,
                    "source_id": "src-1",
                },
            ],
        }

        await rag.ainsert_custom_kg(custom_kg)

        graph = rag.chunk_entity_relation_graph
        nodes_before = await graph.nodes()
        edges_before = await graph.edges()
        print(f"\nBefore deletion - nodes: {len(nodes_before)}, edges: {len(edges_before)}")
        print(f"  Node list: {list(nodes_before)}")

        # Delete EntityA
        await rag.adelete_by_entity("EntityA")

        nodes_after = await graph.nodes()
        edges_after = await graph.edges()
        print(f"\nAfter deletion - nodes: {len(nodes_after)}, edges: {len(edges_after)}")
        print(f"  Node list: {list(nodes_after)}")

        print("\n[OK] Entity Deletion test passed\n")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ===================================================================
# Main entry point
# ===================================================================
async def main():
    print()
    print("*" * 70)
    print("  PathRAG Pipeline Step-by-Step Tests")
    print("*" * 70)
    print()

    # Tests that run without an LLM
    test_step1_chunking()
    await test_step2_path_finding()
    await test_step3_custom_kg_insert()
    await test_step6_graph_storage()
    await test_step7_entity_deletion()

    # Tests that require an LLM API key
    await test_step4_full_indexing()
    await test_step5_full_query()

    print("*" * 70)
    print("  All tests completed!")
    print("*" * 70)


if __name__ == "__main__":
    asyncio.run(main())
