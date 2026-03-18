# Redesign SpannerGraphStorage to Use Spanner's Native GraphDB Features

## Context

The current `SpannerGraphStorage` (v1) treats Spanner as a plain relational store — it uses SQL `SELECT`/`INSERT`/`DELETE` on two tables (`{namespace}_entities`, `{namespace}_relationships`) and only has a minimal `CREATE OR REPLACE PROPERTY GRAPH` DDL without leveraging Spanner's graph capabilities.

The user wants a redesign that fully utilizes Spanner's native GraphDB (property graph + GQL), referencing the `langchain-google-spanner-python` implementation patterns.

**Note:** Spanner property graphs always require separate NODE TABLES and EDGE TABLES — this is a Spanner architectural requirement. The redesign focuses on using GQL for graph traversal and batch mutations for writes, rather than raw SQL.

## Configuration

`SpannerGraphStorage` accepts the following configuration via `global_config` dict or environment variables:

| Config key | Environment variable | Description |
|---|---|---|
| `spanner_project_id` | `GOOGLE_CLOUD_PROJECT` | GCP project ID |
| `spanner_instance_id` | `SPANNER_INSTANCE` | Spanner instance ID |
| `spanner_database_id` | `SPANNER_DATABASE` | Spanner database ID |
| `spanner_graph_name` | — | Property graph name (default: `pathrag_{namespace}`) |

Built-in client metrics are disabled by default (`disable_builtin_metrics=True` and `SPANNER_DISABLE_BUILTIN_METRICS=true`) to avoid noisy telemetry errors in the Spanner Python SDK.

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

**Important:** The `id` column must be included in the Entity `PROPERTIES` clause — Spanner does not automatically expose KEY columns as graph properties. Without it, GQL queries like `MATCH (n:Entity {id: @id})` will fail with `Property id is not exposed`.

### 4. Keep SQL for simple lookups
`has_node`, `get_node`, `nodes` — simple single-table lookups where SQL is cleaner and more efficient than GQL.

### 5. GQL requires explicit column aliases for literals
Spanner GQL requires `RETURN` clauses to use explicit aliases for literal values. For example, `RETURN 1` must be written as `RETURN 1 AS result`.

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
        PROPERTIES(id, entity_type, description, source_id)
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
-- has_edge: check edge existence (literal must have alias)
GRAPH {graph_name}
MATCH (s:Entity {id: @src})-[r:Relationship]->(d:Entity {id: @tgt})
RETURN 1 AS result

-- node_degree: count edges (in + out)
GRAPH {graph_name}
MATCH (n:Entity {id: @id})-[r:Relationship]-()
RETURN COUNT(*) AS degree

-- get_node_edges (outgoing):
GRAPH {graph_name}
MATCH (src:Entity {id: @id})-[r:Relationship]->(dst:Entity)
RETURN src.id AS src_id, dst.id AS tgt_id

-- get_node_in_edges (incoming):
GRAPH {graph_name}
MATCH (src:Entity)-[r:Relationship]->(dst:Entity {id: @id})
RETURN src.id AS src_id, dst.id AS tgt_id

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

## Test Suite

The test suite is at `examples/spanner/test_spanner_graph_storage.py` with the following options:

```bash
# Run tests only (no cleanup)
python examples/spanner/test_spanner_graph_storage.py

# Run tests then cleanup test tables
python examples/spanner/test_spanner_graph_storage.py --cleanup

# Cleanup only (skip tests)
python examples/spanner/test_spanner_graph_storage.py --cleanup-only
```

## Verification

1. Ensure the module imports cleanly: `python -c "from PathRAG.spanner_graph_storage import SpannerGraphStorage"`
2. Verify lazy import in PathRAG.py still works
3. Review GQL syntax against Spanner documentation (GRAPH keyword, label syntax, property access)

## Known Issues & Fixes

| Issue | Root Cause | Fix |
|---|---|---|
| `Property id is not exposed` | `id` column missing from Entity `PROPERTIES` clause | Add `id` to `PROPERTIES(id, entity_type, description, source_id)` |
| `A name must be explicitly defined for this column` | GQL `RETURN 1` without alias | Use `RETURN 1 AS result` |
| `the set of resource labels is incomplete, missing (instance_id)` | Spanner SDK built-in metrics telemetry error | Set `disable_builtin_metrics=True` on `spanner.Client()` and env var `SPANNER_DISABLE_BUILTIN_METRICS=true` |
