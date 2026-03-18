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
        if self._table_exists(self._table_name):
            return

        meta_columns = []
        for field_name in sorted(self.meta_fields):
            meta_columns.append(f"  {field_name} STRING(MAX)")

        columns = [
            "  id STRING(MAX) NOT NULL",
            f"  embedding ARRAY<FLOAT64>",
            "  content STRING(MAX)",
        ] + meta_columns

        ddl = (
            f"CREATE TABLE {self._table_name} (\n"
            + ",\n".join(columns)
            + "\n) PRIMARY KEY (id)"
        )

        op = self._database.update_ddl([ddl])
        op.result()
        logger.info(f"Spanner schema ensured for vector table '{self._table_name}'")

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
    # Query (cosine similarity via COSINE_DISTANCE)
    # ------------------------------------------------------------------

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0].tolist()

        # Build meta field selection
        meta_select = ""
        if self.meta_fields:
            meta_select = ", " + ", ".join(sorted(self.meta_fields))

        sql = (
            f"SELECT id, content, "
            f"(1.0 - COSINE_DISTANCE(embedding, @query_embedding)) AS similarity"
            f"{meta_select} "
            f"FROM {self._table_name} "
            f"ORDER BY similarity DESC "
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
        for row in rows:
            doc_id = row[0]
            content = row[1]
            similarity = row[2]

            # Filter by cosine similarity threshold
            if similarity < self.cosine_better_than_threshold:
                continue

            result = {
                "id": doc_id,
                "content": content,
                "distance": similarity,
            }
            for idx, field_name in enumerate(sorted_meta):
                result[field_name] = row[3 + idx]
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
