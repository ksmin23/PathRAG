# Redesign SpannerGraphStorage to Use Spanner's Native GraphDB Features

## Context

The current `SpannerGraphStorage` (v1) treats Spanner as a plain relational store — it uses SQL `SELECT`/`INSERT`/`DELETE` on two tables (`{namespace}_entities`, `{namespace}_relationships`) and only has a minimal `CREATE OR REPLACE PROPERTY GRAPH` DDL without leveraging Spanner's graph capabilities.

The user wants a redesign that fully utilizes Spanner's native GraphDB (property graph + GQL), referencing the `langchain-google-spanner-python` implementation patterns.

**Note:** Spanner property graphs always require separate NODE TABLES and EDGE TABLES — this is a Spanner architectural requirement. The redesign focuses on using GQL for graph traversal and batch mutations for writes, rather than raw SQL.

## Key Changes

### 1. Use GQL queries for graph traversal operations
Replace SQL-based edge/degree queries with GQL via `GRAPH_TABLE` or `GRAPH` queries on the property graph. This leverages Spanner's graph engine for operations like:
- `node_degree` — count edges via GQL pattern matching
- `get_node_edges` / `get_node_in_edges` / `get_node_out_edges` — traverse edges via GQL
- `has_edge` — match edge pattern via GQL
- `get_edge` — retrieve edge properties via GQL
- `edge_degree` — combined degree via GQL

### 2. Use batch mutations instead of DML transactions
Replace `run_in_transaction` + `execute_update` with `database.batch()` + `batch.insert_or_update()` for `upsert_node` and `upsert_edge`, matching the langchain reference pattern. This is more efficient for write operations.

### 3. Improved property graph DDL with explicit labels and properties
Update the `CREATE OR REPLACE PROPERTY GRAPH` DDL to declare explicit LABEL and PROPERTIES clauses, so the graph schema is self-documenting and GQL queries can use typed labels.

### 4. Keep SQL for simple lookups
`has_node`, `get_node`, `nodes` — simple single-table lookups where SQL is cleaner and more efficient than GQL.

## File to Modify

- **`PathRAG/spanner_graph_storage.py`** — complete rewrite

## Implementation Details

### Schema DDL

```sql
CREATE TABLE {node_table} (
  id STRING(MAX) NOT NULL,
  entity_type STRING(MAX),
  description STRING(MAX),
  source_id STRING(MAX),
) PRIMARY KEY (id)

CREATE TABLE {edge_table} (
  id STRING(MAX) NOT NULL,
  target_id STRING(MAX) NOT NULL,
  weight FLOAT64,
  description STRING(MAX),
  keywords STRING(MAX),
  source_id STRING(MAX),
  FOREIGN KEY (id) REFERENCES {node_table}(id),
  FOREIGN KEY (target_id) REFERENCES {node_table}(id),
) PRIMARY KEY (id, target_id)

CREATE OR REPLACE PROPERTY GRAPH {graph_name}
  NODE TABLES (
    {node_table}
      KEY(id)
      LABEL Entity
        PROPERTIES(entity_type, description, source_id)
  )
  EDGE TABLES (
    {edge_table}
      KEY(id, target_id)
      SOURCE KEY(id) REFERENCES {node_table}(id)
      DESTINATION KEY(target_id) REFERENCES {node_table}(id)
      LABEL Relationship
        PROPERTIES(weight, description, keywords, source_id)
  )
```

Key naming changes from v1:
- `node_id` → `id` (matches langchain pattern: `NODE_KEY_COLUMN_NAME = "id"`)
- `source_node_id` → `id`, `target_node_id` → `target_id` (matches langchain: `TARGET_NODE_KEY_COLUMN_NAME = "target_id"`)

### GQL Query Examples

```sql
-- node_degree: count edges (in + out)
GRAPH {graph_name}
MATCH (n:Entity {id: @id})-[r:Relationship]-()
RETURN COUNT(*) AS degree

-- get_node_edges (outgoing):
GRAPH {graph_name}
MATCH (src:Entity {id: @id})-[r:Relationship]->(dst:Entity)
RETURN src.id AS source_id, dst.id AS target_id

-- get_node_in_edges (incoming):
GRAPH {graph_name}
MATCH (src:Entity)-[r:Relationship]->(dst:Entity {id: @id})
RETURN src.id AS source_id, dst.id AS target_id

-- get_edge:
GRAPH {graph_name}
MATCH (s:Entity {id: @src})-[r:Relationship]->(d:Entity {id: @tgt})
RETURN r.weight, r.description, r.keywords, r.source_id
```

### Batch Mutations for Writes

```python
async def upsert_node(self, node_id, node_data):
    with self._database.batch() as batch:
        batch.insert_or_update(
            table=self._node_table,
            columns=("id", "entity_type", "description", "source_id"),
            values=[(node_id, node_data.get("entity_type"), ...)]
        )
```

### Method-by-Method Plan

| Method | Approach |
|--------|----------|
| `has_node` | SQL (simple key lookup) |
| `get_node` | SQL (single row read) |
| `upsert_node` | Batch mutation |
| `delete_node` | Transaction (must delete edges first) |
| `nodes` | SQL (full table scan) |
| `has_edge` | GQL pattern match |
| `get_edge` | GQL with property return |
| `upsert_edge` | Batch mutation (ensure nodes exist first in txn) |
| `node_degree` | GQL undirected pattern count |
| `edge_degree` | Two GQL degree queries summed |
| `get_node_edges` | GQL outgoing pattern |
| `get_node_in_edges` | GQL incoming pattern |
| `get_node_out_edges` | Delegates to `get_node_edges` |
| `edges` | SQL (full edge table scan) |
| `get_pagerank` | GQL degree / SQL total count |
| `index_done_callback` | No-op (Spanner auto-persists) |
| `embed_nodes` | NotImplementedError |

## Verification

1. Ensure the module imports cleanly: `python -c "from PathRAG.spanner_graph_storage import SpannerGraphStorage"`
2. Verify lazy import in PathRAG.py still works
3. Review GQL syntax against Spanner documentation (GRAPH keyword, label syntax, property access)
