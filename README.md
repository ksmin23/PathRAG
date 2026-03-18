# PathRAG - Knowledge Graph-based RAG System

PathRAG (Path-based Retrieval Augmented Generation) is an advanced approach to knowledge retrieval and generation that combines the power of knowledge graphs with large language models (LLMs).

**Paper**: [PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths](https://arxiv.org/abs/2502.14902)

## What is PathRAG?

### Core Concepts

#### Knowledge Graph Integration
PathRAG builds and maintains a knowledge graph from your documents, where:
- **Nodes** represent entities (people, organizations, concepts, locations, etc.)
- **Edges** represent relationships between these entities
- **Properties** store additional information about entities and relationships

#### Path-based Retrieval
Unlike traditional RAG systems that rely solely on vector similarity:
1. PathRAG identifies relevant paths through the knowledge graph
2. These paths provide contextual connections between entities
3. The system can follow logical relationships to find information not directly mentioned

#### Hybrid Search
PathRAG combines multiple search strategies:
- **Vector search** for semantic similarity
- **Graph traversal** for relationship-based connections
- **Entity-centric retrieval** for focused information about specific entities

#### Advantages Over Traditional RAG
- **Relational understanding**: Captures relationships between concepts, not just similarity
- **Explainability**: Provides clear paths showing how information is connected
- **Reduced hallucinations**: Grounds responses in explicit knowledge connections
- **Complex reasoning**: Can answer multi-hop questions requiring several logical steps

### How PathRAG Works

1. **Document Processing**: Documents are chunked, entities and relationships are extracted, and a knowledge graph is constructed
2. **Query Processing**: Queries are analyzed to identify key entities, and relevant paths in the knowledge graph are identified
3. **Response Generation**: Retrieved context from multiple paths is synthesized by the LLM into grounded responses

## Project Structure

```
PathRAG/
├── PathRAG/                        # Core library (pip install -e .)
│   ├── PathRAG.py                  # Main PathRAG class
│   ├── __init__.py                 # Package exports
│   ├── base.py                     # Base classes (StorageNameSpace, BaseKV/Vector/GraphStorage)
│   ├── llm.py                      # LLM integrations (OpenAI, LiteLLM, HuggingFace, etc.)
│   ├── operate.py                  # Graph operations
│   ├── prompt.py                   # Prompt templates
│   ├── utils.py                    # Utilities
│   │
│   └── storage/                    # Storage backend implementations
│       ├── __init__.py             # Re-exports default implementations
│       ├── defaults.py             # JsonKVStorage, NanoVectorDBStorage, NetworkXStorage
│       └── spanner/                # Google Cloud Spanner backend
│           ├── __init__.py         # Re-exports Spanner classes
│           ├── kv.py               # SpannerKVStorage
│           ├── vector.py           # SpannerVectorDBStorage
│           └── graph.py            # SpannerGraphStorage
│
├── web_app/                        # Web application
│   ├── backend/                    # FastAPI backend server
│   │   ├── main.py                 # Server entry point
│   │   ├── sample.env              # Environment variable template
│   │   ├── api/                    # API routes (auth, chats, documents, etc.)
│   │   └── models/                 # Database models
│   ├── frontend/                   # React frontend
│   │   ├── package.json
│   │   ├── public/
│   │   └── src/
│   ├── scripts/                    # Start scripts (start.sh, start-api.sh, start-ui.sh)
│   ├── INSTALLATION.md             # Web app installation & deployment guide
│   ├── QUICKSTART.md               # Web app user guide
│   └── API_REFERENCE.md            # REST API endpoints & data models
│
├── examples/                       # Usage examples
│   ├── networkx/                   # NetworkX (default) storage examples
│   └── spanner/                    # Google Cloud Spanner storage examples
│
├── docs/                           # Documentation
│   ├── PathRAG_Technical_Specification.md
│   ├── spanner_setup_guide.md
│   └── spanner_graph_storage_plan.md
│
├── setup.py                        # Package setup (with extras_require)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Quick Start

### Prerequisites

- Python 3.10+
- `pip install -e .` (core only)

Optional extras:
```bash
pip install -e ".[spanner]"       # + Google Cloud Spanner storage
pip install -e ".[huggingface]"   # + HuggingFace models (torch, transformers)
pip install -e ".[ollama]"        # + Ollama local models
pip install -e ".[litellm]"       # + LiteLLM multi-provider support
pip install -e ".[vllm]"          # + vLLM inference
pip install -e ".[api]"           # + Web application (FastAPI, etc.)
pip install -e ".[all]"           # Everything
```

### Command Line (Library Only)

#### With OpenAI

```python
import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import gpt_4o_mini_complete

WORKING_DIR = "./your_working_dir"
os.environ["OPENAI_API_KEY"] = "your_api_key"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
)

with open("./text.txt") as f:
    rag.insert(f.read())

print(rag.query("your_question", param=QueryParam(mode="hybrid")))
```

#### With LiteLLM (Multi-Provider Support)

PathRAG uses [LiteLLM](https://docs.litellm.ai/docs/providers) for both LLM and embedding calls, so you can use **any supported provider** (OpenAI, Gemini, Bedrock, Anthropic, Ollama, etc.) by simply changing the model name.

```python
import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import litellm_complete

WORKING_DIR = "./your_working_dir"
os.environ["GEMINI_API_KEY"] = "your_gemini_api_key"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=litellm_complete,
    llm_model_name="gemini/gemini-2.5-flash",
    # Embedding model configuration.
    # Set embedding_model_name and embedding_dim to match the chosen model.
    # Examples:
    #   OpenAI  : embedding_model_name="text-embedding-3-small",  embedding_dim=1536
    #   OpenAI  : embedding_model_name="text-embedding-3-large",  embedding_dim=3072
    #   Gemini  : embedding_model_name="gemini/gemini-embedding-001", embedding_dim=3072
    #   Bedrock : embedding_model_name="bedrock/amazon.titan-embed-text-v2:0", embedding_dim=1024
    embedding_model_name="gemini/gemini-embedding-001",
    embedding_dim=3072,
)

with open("./text.txt") as f:
    rag.insert(f.read())

print(rag.query("your_question", param=QueryParam(mode="hybrid")))
```

#### With Ollama (Local Models)

```python
import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import ollama_model_complete, ollama_embed
from PathRAG.utils import EmbeddingFunc

WORKING_DIR = "./your_working_dir"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen3:0.6b",
    llm_model_max_token_size=8192,
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts, embed_model="nomic-embed-text",
            host="http://localhost:11434",
        ),
    ),
)

with open("your_data_file", "r", encoding="utf-8") as f:
    rag.insert(f.read())

print(rag.query("your_question", param=QueryParam(mode="hybrid")))
```

#### With HuggingFace Models

```python
import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import hf_model_complete, hf_embedding
from PathRAG.utils import EmbeddingFunc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

WORKING_DIR = "./your_working_dir"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Load embedding model
embed_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
embed_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,
    llm_model_name="Qwen/Qwen3-0.6B",
    llm_model_max_token_size=8192,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=8192,
        func=lambda texts: hf_embedding(texts, embed_tokenizer, embed_model),
    ),
)

with open("your_data_file", "r", encoding="utf-8") as f:
    rag.insert(f.read())

print(rag.query("your_question", param=QueryParam(mode="hybrid")))
```

#### Batch Insert

```python
import os

folder_path = "your_folder_path"
txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
for file_name in txt_files:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        rag.insert(file.read())
```

### Web Application

For the full web application with chat UI, document management, and knowledge graph visualization:

- **Installation & Deployment**: [web_app/INSTALLATION.md](web_app/INSTALLATION.md)
- **User Guide**: [web_app/QUICKSTART.md](web_app/QUICKSTART.md)
- **API Reference**: [web_app/API_REFERENCE.md](web_app/API_REFERENCE.md)

**Quick start:**
```bash
# Start both backend and frontend
chmod +x web_app/scripts/start.sh
./web_app/scripts/start.sh
```

## Storage Backends

### Default (Development/Demo)

| Storage | Implementation | Description |
|---------|---------------|-------------|
| Vector | NanoVectorDB | Local file-based vector store |
| Graph | NetworkX | Local in-memory graph |
| Key-Value | JsonKVStorage | Local file-based JSON storage |

> These are suitable for demonstration and development only. Not recommended for production use with large datasets.

See `examples/networkx/` for usage examples.

### Google Cloud Spanner (Production)

PathRAG includes native Spanner storage backends for production deployments:

```bash
pip install -e ".[spanner]"
```

| Storage | Implementation | Description |
|---------|---------------|-------------|
| Vector | `SpannerVectorDBStorage` | Spanner native vector search with COSINE_DISTANCE |
| Graph | `SpannerGraphStorage` | Spanner native GraphDB with GQL queries |
| Key-Value | `SpannerKVStorage` | Spanner as key-value backend |

```python
rag = PathRAG(
    working_dir=work_dir,
    llm_model_name="gemini/gemini-2.5-flash",
    embedding_model_name="gemini/gemini-embedding-001",
    embedding_dim=3072,
    kv_storage="SpannerKVStorage",
    vector_storage="SpannerVectorDBStorage",
    graph_storage="SpannerGraphStorage",
    addon_params={
        "spanner_instance_id": "your-instance",
        "spanner_database_id": "your-database",
    },
)
```

- **Setup Guide**: [docs/spanner_setup_guide.md](docs/spanner_setup_guide.md)
- **Examples**: `examples/spanner/`

### Adding Custom Storage Backends

New backends can be added under `PathRAG/storage/<backend>/` with implementations for KV, vector, and/or graph storage. See [docs/spanner_graph_storage_plan.md](docs/spanner_graph_storage_plan.md#storage-package-architecture) for the step-by-step guide.

### Other Production Options

- **Vector Databases**: PostgreSQL (pgvector), Pinecone, DataStax, Azure Cognitive Search
- **Graph Databases**: Neo4j, ArangoDB, Apache AGE (PostgreSQL extension), CosmosDB
- **Document Databases**: MongoDB, Cassandra, CosmosDB

## Configuration

You can adjust the relevant parameters in `base.py` and `operate.py`:

| Parameter | File | Description |
|-----------|------|-------------|
| `top_k` | `base.py` | Number of nodes retrieved |
| `alpha` | `operate.py` | Decay rate of information propagation along edges |
| `threshold` | `operate.py` | Pruning threshold |

## Evaluation

### Dataset
The dataset used in PathRAG can be downloaded from [TommyChien/UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain).

### Eval Metrics
<details>
<summary>Evaluation Prompt</summary>

```python
You will evaluate two answers to the same question based on five criteria: **Comprehensiveness**, **Diversity**, **logicality**, **Coherence**, **Relevance**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **logicality**: How logically does the answer respond to all parts of the question?
- **Coherence**: How well does the answer maintain internal logical connections between its parts, ensuring a smooth and consistent structure?
- **Relevance**: How relevant is the answer to the question, staying focused and addressing the intended topic or issue?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why.

Here is the question:
{query}

Here are the two answers:
**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the five criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:
{{
  "Comprehensiveness": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "Diversity": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "logicality": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "Coherence": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "Relevance": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }}
}}
```
</details>

## Use Cases

PathRAG is particularly effective for:

### Knowledge-Intensive Applications
- **Research assistance**: Connecting findings across multiple papers and sources
- **Legal document analysis**: Identifying relationships between cases, statutes, and legal concepts
- **Medical knowledge systems**: Connecting symptoms, conditions, treatments, and research

### Complex Information Retrieval
- **Multi-hop question answering**: "What treatments were developed based on research by scientists who studied under Marie Curie?"
- **Contextual understanding**: Understanding how different parts of a document relate to each other
- **Exploratory research**: Discovering unexpected connections between concepts

### Enterprise Knowledge Management
- **Corporate knowledge bases**: Connecting information across departments and documents
- **Compliance and regulation**: Tracking relationships between policies, regulations, and procedures
- **Institutional memory**: Preserving and accessing organizational knowledge

## Limitations and Considerations

- **Knowledge graph quality**: The system's effectiveness depends on the quality of entity and relationship extraction
- **Computational complexity**: Graph operations can be more resource-intensive than simple vector searches
- **Domain specificity**: May require domain-specific entity extraction for specialized fields
- **Storage limitations**: The default storage options (NanoVectorDB, NetworkX) are not suitable for large-scale production use

## Authors & Contributors

### PathRAG Core Logic Research Team
- Boyu Chen¹, Zirui Guo², Zidan Yang¹, Yuluo Chen¹, Junze Chen¹, Zhenghao Liu³, Chuan Shi¹, Cheng Yang¹
  1. Beijing University of Posts and Telecommunications
  2. University of Hong Kong
  3. Northeastern University

  Contact: chenbys4@bupt.edu.cn, yangcheng@bupt.edu.cn

### Demo Application Contributor
- Robert Dennyson, Solution Architect, UK
- Contact: robertdennyson@live.in

## Acknowledgements
- PathRAG for the knowledge graph and retrieval augmented generation capabilities
- RSuite for the UI components
- D3.js for the knowledge graph visualization

## License
This project is licensed under the MIT License - see the LICENSE file for details.
