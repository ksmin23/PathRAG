"""Google Cloud Spanner Graph storage backend for PathRAG.

Implements :class:`PathRAG.base.BaseGraphStorage` using Cloud Spanner's
property-graph feature (GQL queries + batch mutations) as the underlying
persistent store.

Requirements::

    pip install google-cloud-spanner

Configuration (via ``global_config`` dict or environment variables):

    GOOGLE_CLOUD_PROJECT – GCP project ID (or ``spanner_project_id`` in config)
    SPANNER_INSTANCE     – Spanner instance ID
    SPANNER_DATABASE     – Spanner database ID
    SPANNER_GRAPH_NAME   – (optional) property-graph name,
                           defaults to ``pathrag_{namespace}``

Example::

    from PathRAG import PathRAG

    rag = PathRAG(
        working_dir="./data",
        graph_storage="SpannerGraphStorage",
        addon_params={
            "spanner_project_id": "my-project",
            "spanner_instance_id": "my-instance",
            "spanner_database_id": "my-database",
        },
    )
"""

import os
from dataclasses import dataclass
from typing import Union

# Disable Spanner built-in metrics export to avoid noisy telemetry errors.
os.environ.setdefault("SPANNER_DISABLE_BUILTIN_METRICS", "true")

from google.cloud import spanner
from google.cloud.spanner_v1 import param_types

from ...base import BaseGraphStorage
from ...utils import logger


@dataclass
class SpannerGraphStorage(BaseGraphStorage):
    """Spanner Graph backend for PathRAG knowledge-graph storage.

    Two tables are created per namespace:

    * ``{namespace}_nodes`` – node table (KEY: ``id``)
    * ``{namespace}_edges`` – edge table (KEY: ``id``, ``target_id``)

    A Spanner property-graph with explicit LABEL and PROPERTIES clauses
    links the two tables, enabling GQL queries for graph traversal.

    Column naming follows the ``langchain-google-spanner`` convention:
    ``id`` for node keys, ``id``/``target_id`` for edge endpoints.

    All Spanner SDK calls are synchronous.  The async method signatures
    satisfy ``BaseGraphStorage``.
    """

    def __post_init__(self):
        cfg = self.global_config

        self._project_id = (
            cfg.get("spanner_project_id") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        self._instance_id = (
            cfg.get("spanner_instance_id") or os.environ["SPANNER_INSTANCE"]
        )
        self._database_id = (
            cfg.get("spanner_database_id") or os.environ["SPANNER_DATABASE"]
        )
        self._graph_name = cfg.get(
            "spanner_graph_name", f"pathrag_{self.namespace}"
        )

        self._node_table = f"{self.namespace}_nodes"
        self._edge_table = f"{self.namespace}_edges"

        self._client = spanner.Client(
            project=self._project_id,
            disable_builtin_metrics=True,
        )
        self._instance = self._client.instance(self._instance_id)
        self._database = self._instance.database(self._database_id)

        self._ensure_schema()

        logger.info(
            f"SpannerGraphStorage initialised – instance={self._instance_id}, "
            f"database={self._database_id}, graph={self._graph_name}"
        )

    # ------------------------------------------------------------------
    # Schema bootstrapping
    # ------------------------------------------------------------------

    def _table_exists(self, table_name: str) -> bool:
        with self._database.snapshot() as snap:
            rows = list(
                snap.execute_sql(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema = '' AND table_name = @t",
                    params={"t": table_name},
                    param_types={"t": param_types.STRING},
                )
            )
            return rows[0][0] > 0

    def _ensure_schema(self):
        ddl: list[str] = []

        if not self._table_exists(self._node_table):
            ddl.append(
                f"CREATE TABLE {self._node_table} ("
                f"  id STRING(MAX) NOT NULL,"
                f"  entity_type STRING(MAX),"
                f"  description STRING(MAX),"
                f"  source_id STRING(MAX)"
                f") PRIMARY KEY (id)"
            )

        if not self._table_exists(self._edge_table):
            ddl.append(
                f"CREATE TABLE {self._edge_table} ("
                f"  id STRING(MAX) NOT NULL,"
                f"  target_id STRING(MAX) NOT NULL,"
                f"  weight FLOAT64,"
                f"  description STRING(MAX),"
                f"  keywords STRING(MAX),"
                f"  source_id STRING(MAX),"
                f"  CONSTRAINT fk_{self._edge_table}_src"
                f"    FOREIGN KEY (id)"
                f"    REFERENCES {self._node_table}(id),"
                f"  CONSTRAINT fk_{self._edge_table}_tgt"
                f"    FOREIGN KEY (target_id)"
                f"    REFERENCES {self._node_table}(id)"
                f") PRIMARY KEY (id, target_id)"
            )

        ddl.append(
            f"CREATE OR REPLACE PROPERTY GRAPH {self._graph_name}"
            f"  NODE TABLES ("
            f"    {self._node_table}"
            f"      KEY(id)"
            f"      LABEL Entity"
            f"        PROPERTIES(id, entity_type, description, source_id)"
            f"  )"
            f"  EDGE TABLES ("
            f"    {self._edge_table}"
            f"      KEY(id, target_id)"
            f"      SOURCE KEY(id) REFERENCES {self._node_table}(id)"
            f"      DESTINATION KEY(target_id) REFERENCES {self._node_table}(id)"
            f"      LABEL Relationship"
            f"        PROPERTIES(weight, description, keywords, source_id)"
            f"  )"
        )

        if ddl:
            op = self._database.update_ddl(ddl)
            op.result()
            logger.info(f"Spanner schema ensured for graph '{self._graph_name}'")

    # ------------------------------------------------------------------
    # GQL helper
    # ------------------------------------------------------------------

    def _gql(self, query: str, params=None, ptypes=None):
        """Execute a GQL query via GRAPH and return rows as a list."""
        with self._database.snapshot() as snap:
            return list(
                snap.execute_sql(
                    query,
                    params=params or {},
                    param_types=ptypes or {},
                )
            )

    # ------------------------------------------------------------------
    # Node operations (SQL – simple key lookups)
    # ------------------------------------------------------------------

    async def has_node(self, node_id: str) -> bool:
        with self._database.snapshot() as snap:
            rows = list(
                snap.execute_sql(
                    f"SELECT 1 FROM {self._node_table} WHERE id = @id",
                    params={"id": node_id},
                    param_types={"id": param_types.STRING},
                )
            )
            return len(rows) > 0

    async def get_node(self, node_id: str) -> Union[dict, None]:
        with self._database.snapshot() as snap:
            rows = list(
                snap.execute_sql(
                    f"SELECT entity_type, description, source_id "
                    f"FROM {self._node_table} WHERE id = @id",
                    params={"id": node_id},
                    param_types={"id": param_types.STRING},
                )
            )
            if not rows:
                return None
            entity_type, description, source_id = rows[0]
            result = {}
            if entity_type is not None:
                result["entity_type"] = entity_type
            if description is not None:
                result["description"] = description
            if source_id is not None:
                result["source_id"] = source_id
            return result

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        with self._database.batch() as batch:
            batch.insert_or_update(
                table=self._node_table,
                columns=("id", "entity_type", "description", "source_id"),
                values=[
                    (
                        node_id,
                        node_data.get("entity_type"),
                        node_data.get("description"),
                        node_data.get("source_id"),
                    )
                ],
            )

    async def delete_node(self, node_id: str):
        def _txn(transaction):
            transaction.execute_update(
                f"DELETE FROM {self._edge_table} "
                f"WHERE id = @id OR target_id = @id",
                params={"id": node_id},
                param_types={"id": param_types.STRING},
            )
            transaction.execute_update(
                f"DELETE FROM {self._node_table} WHERE id = @id",
                params={"id": node_id},
                param_types={"id": param_types.STRING},
            )

        self._database.run_in_transaction(_txn)
        logger.info(f"Node {node_id} deleted from Spanner graph.")

    async def nodes(self):
        with self._database.snapshot() as snap:
            rows = list(
                snap.execute_sql(f"SELECT id FROM {self._node_table}")
            )
            return [row[0] for row in rows]

    # ------------------------------------------------------------------
    # Edge operations (GQL – graph traversal)
    # ------------------------------------------------------------------

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        rows = self._gql(
            f"GRAPH {self._graph_name} "
            f"MATCH (s:Entity {{id: @src}})-[r:Relationship]->(d:Entity {{id: @tgt}}) "
            f"RETURN 1 AS result",
            params={"src": source_node_id, "tgt": target_node_id},
            ptypes={"src": param_types.STRING, "tgt": param_types.STRING},
        )
        return len(rows) > 0

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        rows = self._gql(
            f"GRAPH {self._graph_name} "
            f"MATCH (s:Entity {{id: @src}})-[r:Relationship]->(d:Entity {{id: @tgt}}) "
            f"RETURN r.weight, r.description, r.keywords, r.source_id",
            params={"src": source_node_id, "tgt": target_node_id},
            ptypes={"src": param_types.STRING, "tgt": param_types.STRING},
        )
        if not rows:
            return None
        weight, description, keywords, source_id = rows[0]
        result = {}
        if weight is not None:
            result["weight"] = weight
        if description is not None:
            result["description"] = description
        if keywords is not None:
            result["keywords"] = keywords
        if source_id is not None:
            result["source_id"] = source_id
        return result

    async def upsert_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_data: dict[str, str],
    ):
        weight = edge_data.get("weight")
        if weight is not None:
            weight = float(weight)

        def _txn(transaction):
            # Ensure both endpoint nodes exist (FK constraint).
            # NOTE: The SELECT-then-INSERT pattern below is safe within a
            # single transaction, but concurrent upsert_edge calls targeting
            # the same node could race: both see "not exists" and attempt an
            # INSERT, causing one to fail with ALREADY_EXISTS.  This is
            # acceptable for now because PathRAG processes edges sequentially.
            # If concurrent edge processing is ever introduced, switch to an
            # atomic INSERT ... WHERE NOT EXISTS DML statement instead.
            for nid in (source_node_id, target_node_id):
                existing = list(
                    transaction.execute_sql(
                        f"SELECT 1 FROM {self._node_table} WHERE id = @id",
                        params={"id": nid},
                        param_types={"id": param_types.STRING},
                    )
                )
                if not existing:
                    transaction.insert(
                        table=self._node_table,
                        columns=["id"],
                        values=[[nid]],
                    )

            transaction.insert_or_update(
                table=self._edge_table,
                columns=["id", "target_id", "weight",
                          "description", "keywords", "source_id"],
                values=[[
                    source_node_id,
                    target_node_id,
                    weight,
                    edge_data.get("description"),
                    edge_data.get("keywords"),
                    edge_data.get("source_id"),
                ]],
            )

        self._database.run_in_transaction(_txn)

    async def node_degree(self, node_id: str) -> int:
        rows = self._gql(
            f"GRAPH {self._graph_name} "
            f"MATCH (n:Entity {{id: @id}})-[r:Relationship]-() "
            f"RETURN COUNT(*) AS degree",
            params={"id": node_id},
            ptypes={"id": param_types.STRING},
        )
        return rows[0][0] if rows else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_deg = await self.node_degree(src_id)
        tgt_deg = await self.node_degree(tgt_id)
        return src_deg + tgt_deg

    async def get_node_edges(self, source_node_id: str):
        if not await self.has_node(source_node_id):
            return None
        rows = self._gql(
            f"GRAPH {self._graph_name} "
            f"MATCH (src:Entity {{id: @id}})-[r:Relationship]->(dst:Entity) "
            f"RETURN src.id AS src_id, dst.id AS tgt_id",
            params={"id": source_node_id},
            ptypes={"id": param_types.STRING},
        )
        return [(r[0], r[1]) for r in rows]

    async def get_node_in_edges(self, source_node_id: str):
        if not await self.has_node(source_node_id):
            return None
        rows = self._gql(
            f"GRAPH {self._graph_name} "
            f"MATCH (src:Entity)-[r:Relationship]->(dst:Entity {{id: @id}}) "
            f"RETURN src.id AS src_id, dst.id AS tgt_id",
            params={"id": source_node_id},
            ptypes={"id": param_types.STRING},
        )
        return [(r[0], r[1]) for r in rows]

    async def get_node_out_edges(self, source_node_id: str):
        return await self.get_node_edges(source_node_id)

    async def edges(self):
        with self._database.snapshot() as snap:
            rows = list(
                snap.execute_sql(
                    f"SELECT id, target_id FROM {self._edge_table}"
                )
            )
            return [(r[0], r[1]) for r in rows]

    # ------------------------------------------------------------------
    # Graph-wide operations
    # ------------------------------------------------------------------

    async def get_pagerank(self, node_id: str) -> float:
        """Approximate PageRank using normalised degree.

        Full iterative PageRank over Spanner is impractical for a real-time
        call, so normalised degree is used as a lightweight proxy.
        """
        degree = await self.node_degree(node_id)
        with self._database.snapshot() as snap:
            rows = list(
                snap.execute_sql(
                    f"SELECT COUNT(*) FROM {self._edge_table}"
                )
            )
            total_edges = rows[0][0] if rows else 1
        return degree / max(total_edges, 1)

    async def index_done_callback(self):
        logger.info(
            f"SpannerGraphStorage index_done_callback "
            f"(graph={self._graph_name})"
        )

    async def find_paths_between(
        self, source_nodes: list[str], max_hops: int = 3
    ) -> dict:
        """Find paths (up to *max_hops*) between *source_nodes* using GQL.

        Returns a dict with keys ``"one_hop"``, ``"two_hop"``, ``"three_hop"``
        where each value is a list of paths (each path is a list of node ids).
        Only paths whose **both endpoints** are in *source_nodes* are returned.
        """
        if not source_nodes:
            return {"one_hop": [], "two_hop": [], "three_hop": []}

        ptypes = {"nodes": param_types.Array(param_types.STRING)}

        one_hop: list[list[str]] = []
        two_hop: list[list[str]] = []
        three_hop: list[list[str]] = []

        # 1-hop
        if max_hops >= 1:
            rows = self._gql(
                f"GRAPH {self._graph_name} "
                f"MATCH (a:Entity)-[:Relationship]->(b:Entity) "
                f"WHERE a.id IN UNNEST(@nodes) AND b.id IN UNNEST(@nodes) "
                f"AND a.id != b.id "
                f"RETURN a.id AS src, b.id AS tgt",
                params={"nodes": source_nodes},
                ptypes=ptypes,
            )
            one_hop = [[r[0], r[1]] for r in rows]

        # 2-hop
        if max_hops >= 2:
            rows = self._gql(
                f"GRAPH {self._graph_name} "
                f"MATCH (a:Entity)-[:Relationship]->(m:Entity)"
                f"-[:Relationship]->(b:Entity) "
                f"WHERE a.id IN UNNEST(@nodes) AND b.id IN UNNEST(@nodes) "
                f"AND a.id != b.id AND m.id != a.id AND m.id != b.id "
                f"RETURN a.id AS src, m.id AS mid, b.id AS tgt",
                params={"nodes": source_nodes},
                ptypes=ptypes,
            )
            two_hop = [[r[0], r[1], r[2]] for r in rows]

        # 3-hop
        if max_hops >= 3:
            rows = self._gql(
                f"GRAPH {self._graph_name} "
                f"MATCH (a:Entity)-[:Relationship]->(m1:Entity)"
                f"-[:Relationship]->(m2:Entity)-[:Relationship]->(b:Entity) "
                f"WHERE a.id IN UNNEST(@nodes) AND b.id IN UNNEST(@nodes) "
                f"AND a.id != b.id "
                f"AND m1.id != a.id AND m1.id != b.id "
                f"AND m2.id != a.id AND m2.id != b.id AND m2.id != m1.id "
                f"RETURN a.id AS src, m1.id AS mid1, m2.id AS mid2, b.id AS tgt",
                params={"nodes": source_nodes},
                ptypes=ptypes,
            )
            three_hop = [[r[0], r[1], r[2], r[3]] for r in rows]

        logger.info(
            f"find_paths_between: {len(one_hop)} 1-hop, "
            f"{len(two_hop)} 2-hop, {len(three_hop)} 3-hop paths"
        )
        return {
            "one_hop": one_hop,
            "two_hop": two_hop,
            "three_hop": three_hop,
        }

    async def embed_nodes(self, algorithm: str):
        raise NotImplementedError("Node embedding is not used in PathRAG.")
