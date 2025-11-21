from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from matrixone import Client

logger = logging.getLogger(__name__)


class MatrixOneVectorStore(VectorStore):
    """MatrixOne vector store integration compatible with LangChain."""

    def __init__(
        self,
        embedding: Embeddings,
        connection_args: Optional[Dict[str, Any]] = None,
        *,
        client: Optional[Client] = None,
        table_name: str = "langchain_vectors",
        content_column: str = "content",
        metadata_column: str = "metadata",
        vector_column: str = "embedding",
        drop_old: bool = False,
        distance: str = "l2",
    ) -> None:
        if client is None and not connection_args:
            raise ValueError("Either an existing MatrixOne Client or connection_args must be provided.")

        self.embedding = embedding
        self.connection_args = connection_args or {}
        self.table_name = table_name
        self.content_column = content_column
        self.metadata_column = metadata_column
        self.vector_column = vector_column
        self.distance = distance

        self.client = client or Client()
        self._owns_client = client is None

        if self._owns_client:
            self._connect_client()
        elif not self.client.connected():
            raise ValueError("Provided MatrixOne Client is not connected.")

        self._create_table_if_not_exists(drop_old)

    def _connect_client(self) -> None:
        required_keys = {"host", "port", "user", "password", "database"}
        missing = required_keys - self.connection_args.keys()
        if missing:
            raise ValueError(f"connection_args missing required keys: {', '.join(sorted(missing))}")

        self.client.connect(**self.connection_args)

    def _create_table_if_not_exists(self, drop_old: bool = False) -> None:
        dummy_embedding = self.embedding.embed_query("matrixone-init")
        dim = len(dummy_embedding)

        if drop_old:
            self.client.execute(f"DROP TABLE IF EXISTS {self.table_name}")

        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id VARCHAR(36) PRIMARY KEY,
            {self.content_column} TEXT,
            {self.metadata_column} JSON,
            {self.vector_column} VECF32({dim})
        )
        """
        self.client.execute(create_sql)

    def _format_metadata(self, metadata: Optional[dict]) -> str:
        if not metadata:
            return "{}"
        return json.dumps(metadata)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        embeddings = self.embedding.embed_documents(texts)

        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        records: List[Dict[str, Any]] = []
        for idx, text in enumerate(texts):
            records.append(
                {
                    "id": ids[idx],
                    self.content_column: text,
                    self.metadata_column: self._format_metadata(metadatas[idx]),
                    self.vector_column: embeddings[idx],
                }
            )

        self.client.vector_ops.batch_insert(self.table_name, records)
        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        rows = self.client.vector_ops.similarity_search(
            self.table_name,
            vector_column=self.vector_column,
            query_vector=embedding,
            limit=k,
            select_columns=[self.content_column, self.metadata_column],
            distance_type=self.distance,
        )

        docs: List[Document] = []
        for row in rows:
            content = row.get(self.content_column)
            metadata_value = row.get(self.metadata_column)
            metadata: Dict[str, Any]
            if isinstance(metadata_value, dict):
                metadata = metadata_value
            elif isinstance(metadata_value, str) and metadata_value:
                try:
                    metadata = json.loads(metadata_value)
                except json.JSONDecodeError:
                    metadata = {"raw_metadata": metadata_value}
            else:
                metadata = {}

            if content is None:
                continue
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            return True

        placeholders = ", ".join(["?"] * len(ids))
        sql = f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"
        self.client.execute(sql, tuple(ids))
        return True

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        client: Optional[Client] = None,
        **kwargs: Any,
    ) -> "MatrixOneVectorStore":
        store = cls(
            embedding=embedding,
            connection_args=connection_args,
            client=client,
            **kwargs,
        )
        store.add_texts(texts, metadatas=metadatas)
        return store

    def __del__(self) -> None:
        if getattr(self, "_owns_client", False):
            try:
                self.client.disconnect()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
