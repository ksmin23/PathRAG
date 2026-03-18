"""Google Cloud Spanner KV storage backend for PathRAG.

Implements :class:`PathRAG.base.BaseKVStorage` using Cloud Spanner as the
underlying persistent key-value store.  Values are stored as JSON strings.

Requirements::

    pip install google-cloud-spanner

Configuration (via ``global_config`` dict or environment variables):

    GOOGLE_CLOUD_PROJECT – GCP project ID (or ``spanner_project_id`` in config)
    SPANNER_INSTANCE     – Spanner instance ID
    SPANNER_DATABASE     – Spanner database ID

Example::

    from PathRAG import PathRAG

    rag = PathRAG(
        working_dir="./data",
        kv_storage="SpannerKVStorage",
        addon_params={
            "spanner_project_id": "my-project",
            "spanner_instance_id": "my-instance",
            "spanner_database_id": "my-database",
        },
    )
"""

import json
import os
from dataclasses import dataclass
from typing import Union

os.environ.setdefault("SPANNER_DISABLE_BUILTIN_METRICS", "true")

from google.cloud import spanner
from google.cloud.spanner_v1 import param_types

from .base import BaseKVStorage
from .utils import logger


@dataclass
class SpannerKVStorage(BaseKVStorage):
    """Spanner backend for PathRAG key-value storage.

    A single table ``{namespace}_kv`` is created with columns:

    * ``id`` (STRING) – primary key
    * ``data`` (STRING) – JSON-serialised value dict

    All Spanner SDK calls are synchronous.  The async method signatures
    satisfy ``BaseKVStorage``.
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

        self._table = f"{self.namespace}_kv"

        self._client = spanner.Client(
            project=self._project_id,
            disable_builtin_metrics=True,
        )
        self._instance = self._client.instance(self._instance_id)
        self._database = self._instance.database(self._database_id)

        self._ensure_schema()

        logger.info(
            f"SpannerKVStorage initialised – instance={self._instance_id}, "
            f"database={self._database_id}, table={self._table}"
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
        if self._table_exists(self._table):
            return

        ddl = [
            f"CREATE TABLE {self._table} ("
            f"  id STRING(MAX) NOT NULL,"
            f"  data STRING(MAX)"
            f") PRIMARY KEY (id)"
        ]
        op = self._database.update_ddl(ddl)
        op.result()
        logger.info(f"Spanner schema ensured for KV table '{self._table}'")

    # ------------------------------------------------------------------
    # BaseKVStorage implementation
    # ------------------------------------------------------------------

    async def all_keys(self) -> list[str]:
        with self._database.snapshot() as snap:
            rows = list(
                snap.execute_sql(f"SELECT id FROM {self._table}")
            )
            return [row[0] for row in rows]

    async def get_by_id(self, id: str) -> Union[dict, None]:
        with self._database.snapshot() as snap:
            rows = list(
                snap.execute_sql(
                    f"SELECT data FROM {self._table} WHERE id = @id",
                    params={"id": id},
                    param_types={"id": param_types.STRING},
                )
            )
            if not rows:
                return None
            return json.loads(rows[0][0]) if rows[0][0] else None

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[dict, None]]:
        if not ids:
            return []

        with self._database.snapshot() as snap:
            rows = list(
                snap.execute_sql(
                    f"SELECT id, data FROM {self._table} "
                    f"WHERE id IN UNNEST(@ids)",
                    params={"ids": ids},
                    param_types={"ids": param_types.Array(param_types.STRING)},
                )
            )

        result_map = {}
        for row in rows:
            val = json.loads(row[1]) if row[1] else None
            if val is not None and fields is not None:
                val = {k: v for k, v in val.items() if k in fields}
            result_map[row[0]] = val

        return [result_map.get(id, None) for id in ids]

    async def filter_keys(self, data: list[str]) -> set[str]:
        if not data:
            return set()

        with self._database.snapshot() as snap:
            rows = list(
                snap.execute_sql(
                    f"SELECT id FROM {self._table} WHERE id IN UNNEST(@ids)",
                    params={"ids": data},
                    param_types={"ids": param_types.Array(param_types.STRING)},
                )
            )

        existing = {row[0] for row in rows}
        return set(s for s in data if s not in existing)

    async def upsert(self, data: dict[str, dict]) -> dict[str, dict]:
        if not data:
            return {}

        # Only insert keys that don't already exist (matches JsonKVStorage behaviour)
        existing_keys = set()
        all_keys = list(data.keys())

        with self._database.snapshot() as snap:
            rows = list(
                snap.execute_sql(
                    f"SELECT id FROM {self._table} WHERE id IN UNNEST(@ids)",
                    params={"ids": all_keys},
                    param_types={"ids": param_types.Array(param_types.STRING)},
                )
            )
            existing_keys = {row[0] for row in rows}

        left_data = {k: v for k, v in data.items() if k not in existing_keys}

        if left_data:
            with self._database.batch() as batch:
                batch.insert_or_update(
                    table=self._table,
                    columns=["id", "data"],
                    values=[
                        (k, json.dumps(v, ensure_ascii=False))
                        for k, v in left_data.items()
                    ],
                )

        return left_data

    async def drop(self):
        def _txn(transaction):
            transaction.execute_update(f"DELETE FROM {self._table} WHERE TRUE")

        self._database.run_in_transaction(_txn)
        logger.info(f"SpannerKVStorage table '{self._table}' cleared.")

    async def index_done_callback(self):
        logger.info(
            f"SpannerKVStorage index_done_callback (table={self._table})"
        )
