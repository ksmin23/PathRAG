"""Google Cloud Spanner Vector DB storage backend for PathRAG.

Implements :class:`PathRAG.base.BaseVectorStorage` using Cloud Spanner's
vector search capabilities (COSINE_DISTANCE with KNN) as the underlying
persistent store.

Requirements::

    pip install google-cloud-spanner numpy

Configuration (via ``global_config`` dict or environment variables):

    GOOGLE_CLOUD_PROJECT – GCP project ID (or ``spanner_project_id`` in config)
    SPANNER_INSTANCE     – Spanner instance ID
    SPANNER_DATABASE     – Spanner database ID

Example::

    from PathRAG import PathRAG

    rag = PathRAG(
        working_dir="./data",
        vector_storage="SpannerVectorDBStorage",
        addon_params={
            "spanner_project_id": "my-project",
            "spanner_instance_id": "my-instance",
            "spanner_database_id": "my-database",
        },
    )
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm.asyncio import tqdm as tqdm_async

os.environ.setdefault("SPANNER_DISABLE_BUILTIN_METRICS", "true")

from google.cloud import spanner
from google.cloud.spanner_v1 import param_types

from ...base import BaseVectorStorage
from ...utils import compute_mdhash_id, logger


@dataclass
class SpannerVectorDBStorage(BaseVectorStorage):
    """Spanner Vector DB backend for PathRAG vector storage.

    One table is created per namespace:

    * ``vdb_{namespace}`` – vector table with embedding and metadata columns

    Uses Spanner's ``COSINE_DISTANCE`` for nearest-neighbor search.
    """

    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        cfg = self.global_config

        self._project_id = cfg.get("spanner_project_id") or os.environ.get(
            "GOOGLE_CLOUD_PROJECT"
        )
        self._instance_id = cfg.get("spanner_instance_id") or os.environ.get(
            "SPANNER_INSTANCE"
        )
        self._database_id = cfg.get("spanner_database_id") or os.environ.get(
            "SPANNER_DATABASE"
        )
        self._max_batch_size = cfg.get("embedding_batch_num", 32)
        self._embedding_dim = self.embedding_func.embedding_dim
        self._table_name = f"vdb_{self.namespace}"
        self._vector_index_name = f"{self._table_name}_embedding_idx"

        self.cosine_better_than_threshold = cfg.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

        self._client = spanner.Client(
            project=self._project_id,
            disable_builtin_metrics=True,
        )
        self._instance = self._client.instance(self._instance_id)
        self._database = self._instance.database(self._database_id)

        self._ensure_schema()

        logger.info(
            f"SpannerVectorDBStorage initialised – "
            f"instance={self._instance_id}, database={self._database_id}, "
            f"table={self._table_name}"
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
        if not self._table_exists(self._table_name):
            meta_columns = []
            for field_name in sorted(self.meta_fields):
                meta_columns.append(f"  {field_name} STRING(MAX)")

            columns = [
                "  id STRING(MAX) NOT NULL",
                f"  embedding ARRAY<FLOAT64>(vector_length=>{self._embedding_dim})",
                "  content STRING(MAX)",
            ] + meta_columns

            ddl = (
                f"CREATE TABLE {self._table_name} (\n"
                + ",\n".join(columns)
                + "\n) PRIMARY KEY (id)"
            )

            op = self._database.update_ddl([ddl])
            op.result()
            logger.info(
                f"Spanner schema ensured for vector table '{self._table_name}'"
            )

        self._ensure_vector_index()

    def _index_exists(self, index_name: str) -> bool:
        with self._database.snapshot() as snap:
            rows = list(
                snap.execute_sql(
                    "SELECT COUNT(*) FROM information_schema.indexes "
                    "WHERE table_name = @t AND index_name = @idx",
                    params={"t": self._table_name, "idx": index_name},
                    param_types={
                        "t": param_types.STRING,
                        "idx": param_types.STRING,
                    },
                )
            )
            return rows[0][0] > 0

    def _ensure_vector_index(self):
        """Create a vector index on the embedding column if one does not exist.

        Without this index, every vector query performs a full table scan
        to compute cosine distance for all rows.  The vector index enables
        Spanner's ANN search (APPROX_COSINE_DISTANCE), which uses a
        tree-based structure to narrow the search space significantly.

        Spanner requires the embedding column to have ``vector_length``
        set before a vector index can reference it.  If the table was
        created before this requirement was introduced, we ALTER the
        column first to add the annotation.
        """
        if self._index_exists(self._vector_index_name):
            return

        # Ensure the embedding column has vector_length set; tables
        # created before the vector index support may lack it.
        self._ensure_vector_length()

        cfg = self.global_config
        tree_depth = cfg.get("vector_index_tree_depth", 2)

        ddl = (
            f"CREATE VECTOR INDEX {self._vector_index_name} "
            f"ON {self._table_name}(embedding) "
            f"WHERE embedding IS NOT NULL "
            f"OPTIONS(distance_type='COSINE', tree_depth={tree_depth})"
        )
        op = self._database.update_ddl([ddl])
        op.result()
        logger.info(
            f"Vector index '{self._vector_index_name}' created "
            f"on '{self._table_name}'"
        )

    def _ensure_vector_length(self):
        """ALTER the embedding column to add vector_length if missing.

        Spanner's VECTOR INDEX requires the leading key column to have
        ``vector_length`` set.  Tables created with a plain
        ``ARRAY<FLOAT64>`` column will fail at index creation time
        without this annotation.
        """
        with self._database.snapshot() as snap:
            rows = list(
                snap.execute_sql(
                    "SELECT c.SPANNER_TYPE "
                    "FROM information_schema.columns c "
                    "WHERE c.TABLE_NAME = @t AND c.COLUMN_NAME = 'embedding'",
                    params={"t": self._table_name},
                    param_types={"t": param_types.STRING},
                )
            )

        if not rows:
            return

        spanner_type = rows[0][0]
        if "vector_length" in spanner_type.lower():
            return

        ddl = (
            f"ALTER TABLE {self._table_name} "
            f"ALTER COLUMN embedding "
            f"ARRAY<FLOAT64>(vector_length=>{self._embedding_dim})"
        )
        op = self._database.update_ddl([ddl])
        op.result()
        logger.info(
            f"Altered column embedding on '{self._table_name}' "
            f"to set vector_length={self._embedding_dim}"
        )

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []

        list_data = [
            {
                "id": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)
        embeddings = np.concatenate(embeddings_list)

        if len(embeddings) != len(list_data):
            logger.error(
                f"embedding is not 1-1 with data, "
                f"{len(embeddings)} != {len(list_data)}"
            )
            return []

        # Build columns list based on meta_fields
        base_columns = ["id", "embedding", "content"]
        extra_columns = sorted(self.meta_fields)
        all_columns = base_columns + extra_columns

        # Batch write to Spanner (max 20000 mutations per commit)
        SPANNER_BATCH_SIZE = 500
        for batch_start in range(0, len(list_data), SPANNER_BATCH_SIZE):
            batch_end = min(batch_start + SPANNER_BATCH_SIZE, len(list_data))
            values = []
            for i in range(batch_start, batch_end):
                row = list_data[i]
                embedding_list = embeddings[i].tolist()
                content = contents[i]
                row_values = [row["id"], embedding_list, content]
                for col in extra_columns:
                    row_values.append(row.get(col))
                values.append(row_values)

            with self._database.batch() as batch:
                batch.insert_or_update(
                    table=self._table_name,
                    columns=all_columns,
                    values=values,
                )

        return list_data

    # ------------------------------------------------------------------
    # Query (ANN via APPROX_COSINE_DISTANCE + Vector Index)
    # ------------------------------------------------------------------
    #
    # Why we use APPROX_COSINE_DISTANCE instead of COSINE_DISTANCE:
    #
    # A naive query like:
    #
    #   SELECT id, (1.0 - COSINE_DISTANCE(embedding, @q)) AS similarity
    #   FROM table ORDER BY similarity DESC LIMIT @k
    #
    # has two problems that prevent Spanner from using its KNN optimiser:
    #
    #  1. Wrapping the distance in an expression (1.0 - ...) and sorting
    #     by the derived alias hides the KNN pattern from the query
    #     planner, so every query degrades to a full table scan that
    #     computes cosine distance for every single row.
    #
    #  2. Even with a bare ``ORDER BY COSINE_DISTANCE(...)``, Spanner
    #     still performs an *exact* KNN scan without a vector index —
    #     which is O(n) on table size.
    #
    # By creating a VECTOR INDEX (see _ensure_vector_index) and querying
    # with APPROX_COSINE_DISTANCE + FORCE_INDEX, Spanner performs an
    # approximate nearest-neighbour (ANN) search using its tree-based
    # index structure, reducing query cost from O(n) to O(log n).
    # The ``num_leaves_to_search`` option controls the accuracy/speed
    # trade-off: higher values yield more accurate results at the cost
    # of increased latency.
    #
    # IMPORTANT: Spanner restricts APPROX_COSINE_DISTANCE to the
    # ORDER BY clause only — it cannot appear in SELECT or WHERE.
    # Therefore we use COSINE_DISTANCE in SELECT to obtain the actual
    # distance value, and APPROX_COSINE_DISTANCE solely in ORDER BY
    # to drive the ANN index scan.  Similarity (1.0 - distance) is
    # then computed in Python.
    # ------------------------------------------------------------------

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0].tolist()

        # Build meta field selection
        meta_select = ""
        if self.meta_fields:
            meta_select = ", " + ", ".join(sorted(self.meta_fields))

        num_leaves = self.global_config.get("num_leaves_to_search", 100)

        # APPROX_COSINE_DISTANCE can ONLY appear in the ORDER BY clause.
        # Spanner rejects it in SELECT or WHERE.  We use COSINE_DISTANCE
        # in SELECT to obtain the actual distance value for each result.
        approx_order = (
            f"APPROX_COSINE_DISTANCE(embedding, @query_embedding, "
            f"""options => JSON '{{"num_leaves_to_search": {num_leaves}}}')"""
        )

        sql = (
            f"SELECT id, content{meta_select}, "
            f"COSINE_DISTANCE(embedding, @query_embedding) AS distance "
            f"FROM {self._table_name}"
            f"@{{FORCE_INDEX={self._vector_index_name}}} "
            f"WHERE embedding IS NOT NULL "
            f"ORDER BY {approx_order} ASC "
            f"LIMIT @top_k"
        )

        with self._database.snapshot() as snap:
            rows = list(
                snap.execute_sql(
                    sql,
                    params={
                        "query_embedding": embedding,
                        "top_k": top_k,
                    },
                    param_types={
                        "query_embedding": param_types.Array(param_types.FLOAT64),
                        "top_k": param_types.INT64,
                    },
                )
            )

        results = []
        sorted_meta = sorted(self.meta_fields)
        distance_idx = 2 + len(sorted_meta)
        for row in rows:
            doc_id = row[0]
            content = row[1]
            distance = row[distance_idx]
            similarity = 1.0 - distance

            # Filter by cosine similarity threshold
            if similarity < self.cosine_better_than_threshold:
                continue

            result = {
                "id": doc_id,
                "content": content,
                "distance": similarity,
            }
            for idx, field_name in enumerate(sorted_meta):
                result[field_name] = row[2 + idx]
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    async def delete_entity(self, entity_name: str):
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            with self._database.snapshot() as snap:
                rows = list(
                    snap.execute_sql(
                        f"SELECT 1 FROM {self._table_name} WHERE id = @id",
                        params={"id": entity_id},
                        param_types={"id": param_types.STRING},
                    )
                )
            if rows:
                with self._database.batch() as batch:
                    batch.delete(
                        table=self._table_name,
                        keyset=spanner.KeySet(keys=[[entity_id]]),
                    )
                logger.info(f"Entity {entity_name} has been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_relation(self, entity_name: str):
        try:
            # Find relations where src_id or tgt_id matches entity_name
            with self._database.snapshot() as snap:
                rows = list(
                    snap.execute_sql(
                        f"SELECT id FROM {self._table_name} "
                        f"WHERE src_id = @name OR tgt_id = @name",
                        params={"name": entity_name},
                        param_types={"name": param_types.STRING},
                    )
                )

            if rows:
                ids_to_delete = [[row[0]] for row in rows]
                with self._database.batch() as batch:
                    batch.delete(
                        table=self._table_name,
                        keyset=spanner.KeySet(keys=ids_to_delete),
                    )
                logger.info(
                    f"All relations related to entity {entity_name} have been deleted."
                )
            else:
                logger.info(f"No relations found for entity {entity_name}.")
        except Exception as e:
            logger.error(
                f"Error while deleting relations for entity {entity_name}: {e}"
            )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    async def index_done_callback(self):
        logger.info(
            f"SpannerVectorDBStorage index_done_callback "
            f"(table={self._table_name})"
        )
