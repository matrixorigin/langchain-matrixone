.. LangChain MatrixOne documentation master file.

Welcome to LangChain MatrixOne's documentation!
===============================================

This guide explains how to connect LangChain applications to MatrixOne using
the ``MatrixOneVectorStore`` integration included in this package.

Prerequisites
-------------

* A running MatrixOne deployment accessible via the MySQL protocol.
* Credentials with privileges to create tables and insert/query data.
* A LangChain-compatible embedding model (OpenAI, Hugging Face, Azure, etc.).

Installation
------------

Install the package from PyPI with your preferred tool:

.. code-block:: bash

   pip install langchain-matrixone
   # or
   uv pip install langchain-matrixone

Quickstart Example
------------------

.. code-block:: python

   from langchain_matrixone import MatrixOneVectorStore
   from langchain_community.embeddings import HuggingFaceEmbeddings
   from langchain_text_splitters import RecursiveCharacterTextSplitter
   from langchain_core.documents import Document

    # make sure you have a running MatrixOne deployment accessible via the MySQL protocol.
   connection_args = {
       "host": "127.0.0.1",
       "port": 6001,
       "user": "root",
       "password": "111",
       "database": "langchain_demo",
   }

   embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
   vector_store = MatrixOneVectorStore(
       embedding=embeddings,
       connection_args=connection_args,
       table_name="langchain_vectors",
   )

   docs = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40).split_documents(
       [
           Document(page_content="MatrixOne offers fast vector search with SQL semantics."),
           Document(page_content="LangChain unifies LLM tooling in Python and JavaScript."),
       ]
   )

   vector_store.add_texts(
       [doc.page_content for doc in docs],
       metadatas=[doc.metadata for doc in docs],
   )

   results = vector_store.similarity_search("What does LangChain provide?", k=2)
   for doc in results:
       print(doc.page_content, doc.metadata)

Reusing an existing client
--------------------------

You can pass a pre-configured ``matrixone.Client`` if your application already
maintains pooled connections:

.. code-block:: python

   from matrixone import Client

   client = Client()
   client.connect(**connection_args)

   vector_store = MatrixOneVectorStore(
       embedding=embeddings,
       client=client,
       table_name="langchain_vectors",
   )

Notes
-----

* Tables are created automatically on first use; pass ``drop_old=True`` to rebuild.
* Metadata is stored as JSON in MatrixOne—ensure values are serializable.
* Keep the embedding dimension consistent per table to avoid schema mismatches.
* See the `MatrixOne Python SDK documentation <https://matrixone.readthedocs.io/>`__
  for advanced vector index management and connection best practices.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api_reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

