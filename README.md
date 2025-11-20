# LangChain MatrixOne Integrations

This repository follows the [LangChain integration template](https://github.com/langchain-ai/integration-repo-template).

## Packages

- [libs/langchain-matrixone](libs/langchain-matrixone): MatrixOne vector store integration and docs.

## Development

Use [uv](https://github.com/astral-sh/uv) to manage dependencies.

```bash
uv sync
uv run pytest libs/langchain-matrixone/tests
```

## Documentation

The Sphinx docs live in [docs/](docs) and reference the package inside `libs/langchain-matrixone`.
