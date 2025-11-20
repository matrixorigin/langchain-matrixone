from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional

import pymysql
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class MatrixOneVectorStore(VectorStore):
    """MatrixOne vector store integration compatible with LangChain."""

    def __init__(
        self,
        embedding: Embeddings,
        connection_args: Dict[str, Any],
        table_name: str = "langchain_vectors",
        content_column: str = "content",
        metadata_column: str = "metadata",
        vector_column: str = "embedding",
        drop_old: bool = False,
    ) -> None:
        self.embedding = embedding
        self.connection_args = connection_args
        self.table_name = table_name
        self.content_column = content_column
        self.metadata_column = metadata_column
        self.vector_column = vector_column

        self._create_table_if_not_exists(drop_old)

    def _get_connection(self):
        return pymysql.connect(**self.connection_args)

    def _create_table_if_not_exists(self, drop_old: bool = False) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                if drop_old:
                    cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")

                dummy_embedding = self.embedding.embed_query("test")
                dim = len(dummy_embedding)

                create_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id VARCHAR(36) PRIMARY KEY,
                    {self.content_column} TEXT,
                    {self.metadata_column} JSON,
                    {self.vector_column} VECF32({dim})
                )
                """
                cursor.execute(create_sql)
            conn.commit()
        finally:
            conn.close()

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

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                for idx, text in enumerate(texts):
                    metadata = json.dumps(metadatas[idx])
                    vector_str = str(embeddings[idx])

                    sql = f"""
                    INSERT INTO {self.table_name}
                    (id, {self.content_column}, {self.metadata_column}, {self.vector_column})
                    VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(sql, (ids[idx], text, metadata, vector_str))
            conn.commit()
        finally:
            conn.close()

        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        vector_str = str(embedding)

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                sql = f"""
                SELECT {self.content_column}, {self.metadata_column},
                       l2_distance({self.vector_column}, %s) as score
                FROM {self.table_name}
                ORDER BY score ASC
                LIMIT %s
                """
                cursor.execute(sql, (vector_str, k))
                results = cursor.fetchall()
        finally:
            conn.close()

        docs: List[Document] = []
        for content, metadata_json, _score in results:
            metadata = json.loads(metadata_json) if metadata_json else {}
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            return True

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                format_strings = ",".join(["%s"] * len(ids))
                sql = f"DELETE FROM {self.table_name} WHERE id IN ({format_strings})"
                cursor.execute(sql, tuple(ids))
            conn.commit()
        finally:
            conn.close()

        return True

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "MatrixOneVectorStore":
        if not connection_args:
            raise ValueError("connection_args must be provided")

        store = cls(embedding=embedding, connection_args=connection_args, **kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store
