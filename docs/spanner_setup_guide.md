# Spanner Setup Guide for PathRAG

This guide explains how to set up Google Cloud Spanner for use with PathRAG's `SpannerGraphStorage` backend.

## Prerequisites

- An active Google Cloud project with billing enabled
- [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) installed and configured

## 1. Authenticate with Google Cloud

```bash
gcloud auth application-default login
```

## 2. Configure Your Google Cloud Project

```bash
# Set your project ID
export PROJECT_ID=$(gcloud config get-value project)

# Enable the required APIs
gcloud services enable \
  spanner.googleapis.com \
  aiplatform.googleapis.com \
  cloudresourcemanager.googleapis.com
```

### (Optional) Create a Service Account

If you are running PathRAG on a local machine or a VM without a default service account:

```bash
export SERVICE_ACCOUNT="pathrag-spanner-sa"

# Create the service account
gcloud iam service-accounts create $SERVICE_ACCOUNT \
    --description="Service account for PathRAG with Spanner Graph" \
    --display-name="PathRAG Spanner SA"

# Grant Spanner database user role
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/spanner.databaseUser"

# Grant Vertex AI user role (for Gemini LLM / embedding)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

## 3. Create a Spanner Instance and Database

```bash
# Set environment variables
export SPANNER_INSTANCE="pathrag-instance"
export SPANNER_DATABASE="pathrag-database"
export SPANNER_REGION="us-central1"   # Change to your preferred region

# Create the Spanner instance
gcloud spanner instances create $SPANNER_INSTANCE \
  --config=regional-$SPANNER_REGION \
  --description="Spanner instance for PathRAG" \
  --nodes=1 \
  --edition=ENTERPRISE

# Create the database
gcloud spanner databases create $SPANNER_DATABASE \
  --instance=$SPANNER_INSTANCE
```

> **Note:** `SpannerGraphStorage` automatically creates the required tables
> (`{namespace}_nodes`, `{namespace}_edges`) and property graph DDL on first
> initialization. No manual schema setup is needed.

## 4. Install Dependencies

```bash
pip install google-cloud-spanner
```

Or if you are using the PathRAG project:

```bash
pip install -e .
pip install google-cloud-spanner
```

## 5. Configure Environment Variables

Create a `.env` file in your project root (or `examples/` directory):

```bash
# .env
SPANNER_INSTANCE=pathrag-instance
SPANNER_DATABASE=pathrag-database

# LLM configuration (choose one)
GEMINI_API_KEY=your-gemini-api-key
# OPENAI_API_KEY=your-openai-api-key

# Optional overrides
# LLM_MODEL_NAME=gemini/gemini-2.5-flash
# EMBEDDING_MODEL_NAME=gemini/gemini-embedding-001
# EMBEDDING_DIM=3072
```

## 6. Usage

### Basic Usage with PathRAG

```python
from PathRAG import PathRAG, QueryParam

rag = PathRAG(
    working_dir="./data",
    llm_model_name="gemini/gemini-2.5-flash",
    embedding_model_name="gemini/gemini-embedding-001",
    embedding_dim=3072,
    graph_storage="SpannerGraphStorage",
    addon_params={
        "spanner_instance_id": "pathrag-instance",
        "spanner_database_id": "pathrag-database",
    },
)

# Index documents
await rag.ainsert("Your document text here...")

# Query
response = await rag.aquery(
    "Your question here?",
    param=QueryParam(mode="hybrid"),
)
print(response)
```

### Direct SpannerGraphStorage Usage

```python
from PathRAG.spanner_graph_storage import SpannerGraphStorage

graph = SpannerGraphStorage(
    namespace="my_graph",
    global_config={
        "spanner_instance_id": "pathrag-instance",
        "spanner_database_id": "pathrag-database",
    },
)

# Insert nodes
await graph.upsert_node("APPLE", {
    "entity_type": "ORGANIZATION",
    "description": "American multinational technology company",
    "source_id": "chunk-001",
})

# Insert edges
await graph.upsert_edge("STEVE JOBS", "APPLE", {
    "weight": "3.0",
    "description": "Co-founded Apple in 1976",
    "keywords": "founding, co-founder",
    "source_id": "chunk-001",
})

# Query (uses GQL internally)
edges = await graph.get_node_edges("STEVE JOBS")
degree = await graph.node_degree("APPLE")
```

### Run Test Suite

```bash
export SPANNER_INSTANCE=pathrag-instance
export SPANNER_DATABASE=pathrag-database

python examples/test_spanner_graph_storage.py
```

## 7. Schema Overview

`SpannerGraphStorage` automatically creates the following schema:

### Node Table (`{namespace}_nodes`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | STRING(MAX) NOT NULL | Node identifier (Primary Key) |
| `entity_type` | STRING(MAX) | Entity type (e.g., PERSON, ORGANIZATION) |
| `description` | STRING(MAX) | Entity description |
| `source_id` | STRING(MAX) | Source chunk ID |

### Edge Table (`{namespace}_edges`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | STRING(MAX) NOT NULL | Source node ID (Primary Key, FK → nodes) |
| `target_id` | STRING(MAX) NOT NULL | Target node ID (Primary Key, FK → nodes) |
| `weight` | FLOAT64 | Relationship weight |
| `description` | STRING(MAX) | Relationship description |
| `keywords` | STRING(MAX) | Relationship keywords |
| `source_id` | STRING(MAX) | Source chunk ID |

### Property Graph DDL

```sql
CREATE OR REPLACE PROPERTY GRAPH pathrag_{namespace}
  NODE TABLES (
    {namespace}_nodes
      KEY(id)
      LABEL Entity
        PROPERTIES(entity_type, description, source_id)
  )
  EDGE TABLES (
    {namespace}_edges
      KEY(id, target_id)
      SOURCE KEY(id) REFERENCES {namespace}_nodes(id)
      DESTINATION KEY(target_id) REFERENCES {namespace}_nodes(id)
      LABEL Relationship
        PROPERTIES(weight, description, keywords, source_id)
  )
```

## 8. Cleanup

To remove test data and resources:

```bash
# Drop the database
gcloud spanner databases delete $SPANNER_DATABASE \
  --instance=$SPANNER_INSTANCE

# (Optional) Delete the instance
gcloud spanner instances delete $SPANNER_INSTANCE
```

## References

- [Cloud Spanner Documentation](https://cloud.google.com/spanner/docs)
- [Spanner Graph Overview](https://cloud.google.com/spanner/docs/graph/overview)
- [Spanner GQL Reference](https://cloud.google.com/spanner/docs/reference/standard-sql/graph-query-statements)
- [Build GraphRAG applications using Spanner Graph and LangChain](https://cloud.google.com/blog/products/databases/using-spanner-graph-with-langchain-for-graphrag)
- [langchain-google-spanner-python](https://github.com/googleapis/langchain-google-spanner-python)
- [IAM for Spanner](https://cloud.google.com/spanner/docs/iam)
