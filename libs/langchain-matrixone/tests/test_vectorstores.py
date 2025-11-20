import unittest
from unittest.mock import MagicMock, patch

from langchain_matrixone import MatrixOneVectorStore
from langchain_core.embeddings import FakeEmbeddings


class TestMatrixOneVectorStore(unittest.TestCase):
    def setUp(self):
        self.embedding = FakeEmbeddings(size=4)
        self.connection_args = {"host": "localhost"}

    @patch("pymysql.connect")
    def test_init(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        MatrixOneVectorStore(
            embedding=self.embedding,
            connection_args=self.connection_args,
        )

        mock_cursor.execute.assert_called()
        call_args = mock_cursor.execute.call_args[0][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS langchain_vectors", call_args)
        self.assertIn("VECF32(4)", call_args)

    @patch("pymysql.connect")
    def test_add_texts(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        store = MatrixOneVectorStore(
            embedding=self.embedding,
            connection_args=self.connection_args,
        )

        texts = ["hello", "world"]
        ids = store.add_texts(texts)

        self.assertEqual(len(ids), 2)

        calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
        insert_calls = [c for c in calls if "INSERT INTO" in c]
        self.assertGreaterEqual(len(insert_calls), 2)

    @patch("pymysql.connect")
    def test_similarity_search(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        mock_cursor.fetchall.return_value = [
            ("content1", '{"key": "val"}', 0.1),
            ("content2", '{}', 0.2),
        ]

        store = MatrixOneVectorStore(
            embedding=self.embedding,
            connection_args=self.connection_args,
        )

        results = store.similarity_search("query", k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].page_content, "content1")
        self.assertEqual(results[0].metadata, {"key": "val"})

        calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
        select_calls = [c for c in calls if "SELECT" in c and "l2_distance" in c]
        self.assertGreaterEqual(len(select_calls), 1)

    @patch("pymysql.connect")
    def test_delete(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        store = MatrixOneVectorStore(
            embedding=self.embedding,
            connection_args=self.connection_args,
        )

        store.delete(ids=["1", "2"])

        calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
        delete_calls = [c for c in calls if "DELETE FROM" in c]
        self.assertGreaterEqual(len(delete_calls), 1)


if __name__ == "__main__":
    unittest.main()
