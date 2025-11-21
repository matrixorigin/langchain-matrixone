# LangChain MatrixOne

This package contains the LangChain integration for MatrixOne.

## Installation

You can install this package using pip or uv:

```bash
pip install langchain-matrixone
# or
uv pip install langchain-matrixone
```

## Prerequisites

- A running [MatrixOne](https://matrixorigin.io/) instance that you can reach over MySQL protocol.
- Network credentials (host, port, user, password, database) with permission to create tables.
- An embedding model supported by LangChain (e.g., `OpenAIEmbeddings`, `HuggingFaceEmbeddings`, or a custom `Embeddings` implementation).
- The [MatrixOne Python SDK](https://matrixone.readthedocs.io/) (installed automatically with this package) must be able to connect with vector operations enabled.

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest
```

## Usage

```python
from langchain_matrixone import MatrixOneVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

connection_args = {
    "host": "127.0.0.1",
    "port": 6001,
    "user": "root",
    "password": "111",
    "database": "langchain_demo",
}

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = MatrixOneVectorStore(
    embedding=embedder,
    connection_args=connection_args,
    table_name="langchain_vectors",
)

# Ensure you have a running MatrixOne deployment accessible via MySQL.
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
docs = splitter.split_documents(
    [
        Document(page_content="MatrixOne is a scalable cloud-native database."),
        Document(page_content="LangChain provides common interfaces for LLM apps."),
    ]
)

vector_store.add_texts([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])

query = "What is LangChain?"
similar = vector_store.similarity_search(query, k=2)
for doc in similar:
    print(doc.page_content, doc.metadata)
```

### Using an existing MatrixOne Client

If you already manage a `matrixone.Client` elsewhere in your application, pass it
directly to the vector store to reuse pooled connections:

```python
from matrixone import Client

client = Client()
client.connect(**connection_args)

vector_store = MatrixOneVectorStore(
    embedding=embedder,
    client=client,
    table_name="langchain_vectors",
)
```

## Notes

- The vector table schema is created automatically if it does not exist. Set `drop_old=True` during initialization to recreate it.
- Any additional metadata stored alongside texts must be JSON serializable.
- MatrixOne currently expects embeddings as `VECF32`. Ensure the embedding dimension stays constant for a given table.
- Use `MatrixOneVectorStore.from_texts` when you want a single call that both creates the store and inserts documents.
- For advanced vector indexing (IVF/HNSW) or additional capabilities review the [MatrixOne vector operations guide](https://matrixone.readthedocs.io/en/latest/core_features/vector_search.html).

## Features

- Vector Search with MatrixOne
- Metadata filtering (if supported)
- Add/Delete documents
