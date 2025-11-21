import unittest
from unittest.mock import MagicMock, patch

from langchain_matrixone import MatrixOneVectorStore
from langchain_core.embeddings import FakeEmbeddings


class TestMatrixOneVectorStore(unittest.TestCase):
    def setUp(self):
        self.embedding = FakeEmbeddings(size=4)
        self.connection_args = {
            "host": "localhost",
            "port": 6001,
            "user": "root",
            "password": "111",
            "database": "test",
        }

    def _mock_client(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.vector_ops = MagicMock()
        mock_client.connected.return_value = True
        mock_client_cls.return_value = mock_client
        return mock_client

    @patch("langchain_matrixone.vectorstores.Client")
    def test_init(self, mock_client_cls):
        mock_client = self._mock_client(mock_client_cls)

        MatrixOneVectorStore(
            embedding=self.embedding,
            connection_args=self.connection_args,
        )

        mock_client.connect.assert_called_once_with(**self.connection_args)
        create_call = mock_client.execute.call_args_list[-1][0][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS langchain_vectors", create_call)
        self.assertIn("VECF32(4)", create_call)

    @patch("langchain_matrixone.vectorstores.Client")
    def test_add_texts(self, mock_client_cls):
        mock_client = self._mock_client(mock_client_cls)
        store = MatrixOneVectorStore(
            embedding=self.embedding,
            connection_args=self.connection_args,
        )

        texts = ["hello", "world"]
        ids = store.add_texts(texts)

        self.assertEqual(len(ids), 2)
        mock_client.vector_ops.batch_insert.assert_called_once()
        table_arg, batch_payload = mock_client.vector_ops.batch_insert.call_args[0]
        self.assertEqual(table_arg, "langchain_vectors")
        self.assertEqual(len(batch_payload), 2)
        self.assertIn("content", batch_payload[0])

    @patch("langchain_matrixone.vectorstores.Client")
    def test_similarity_search(self, mock_client_cls):
        mock_client = self._mock_client(mock_client_cls)
        mock_client.vector_ops.similarity_search.return_value = [
            {"content": "content1", "metadata": '{"key": "val"}'},
            {"content": "content2", "metadata": "{}"},
        ]

        store = MatrixOneVectorStore(
            embedding=self.embedding,
            connection_args=self.connection_args,
        )

        results = store.similarity_search("query", k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].page_content, "content1")
        self.assertEqual(results[0].metadata, {"key": "val"})
        mock_client.vector_ops.similarity_search.assert_called_once()

    @patch("langchain_matrixone.vectorstores.Client")
    def test_delete(self, mock_client_cls):
        mock_client = self._mock_client(mock_client_cls)
        store = MatrixOneVectorStore(
            embedding=self.embedding,
            connection_args=self.connection_args,
        )

        store.delete(ids=["1", "2"])

        delete_call = mock_client.execute.call_args_list[-1]
        sql = delete_call[0][0]
        params = delete_call[0][1]
        self.assertIn("DELETE FROM", sql)
        self.assertEqual(params, ("1", "2"))


if __name__ == "__main__":
    unittest.main()
