# LangChain MatrixOne

This package contains the LangChain integration for MatrixOne.

## Installation

You can install this package using pip or uv:

```bash
pip install langchain-matrixone
# or
uv pip install langchain-matrixone
```

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
from langchain_core.embeddings import FakeEmbeddings

embeddings = FakeEmbeddings(size=4)
vector_store = MatrixOneVectorStore(
    embedding=embeddings,
    connection_args={
        "host": "127.0.0.1",
        "port": 6001,
        "user": "dump",
        "password": "111",
        "database": "test"
    },
    table_name="langchain_test"
)

vector_store.add_texts(["hello", "world"])
results = vector_store.similarity_search("hello", k=1)
print(results)
```

## Features

- Vector Search with MatrixOne
- Metadata filtering (if supported)
- Add/Delete documents
