# Arabic Poetry GraphRAG Quickstart

This folder now contains a thin service and CLI that wrap the [`nano-graphrag`](../nano-graphrag) engine and ingest
`../arabicpoetry_first20.csv` (a 20‑row sample of the full corpus).  The goal is to expose a public
API with the familiar `.build_index(...)`, `.retrieve(...)`, and `.dump_index(...)` surface that you can reuse
inside any orchestrator or agent.

## Files

- `poetry_graph_service.py` – loads the CSV, normalises metadata, and forwards calls to `GraphRAGRunner`.
- `poetry_graph_cli.py` – convenience CLI for building the index and testing retrieval.

Both modules expect the `nano-graphrag` repo to live next to `sample-app/` (which is how this workspace is laid out).

## Installation

1. Install Python 3.10+ (`py -3` works on Windows).
2. Install GraphRAG dependencies from the cloned repo:

   ```powershell
   cd ..\nano-graphrag
   py -3 -m pip install -r requirements.txt
   ```

3. Provide your LLM credentials (e.g. `OPENAI_API_KEY`) and, if you want to use Neo4j/Chroma, the corresponding
   connection variables. You can place them in `sample-app/.env`.

> **Note**  
> The embedded copy of `nano-graphrag` does not ship optional storage plugins such as ChromaDB or Firestore.
> The wrapper now falls back to the built-in NanoVectorDB when these classes are missing. If you do need Chroma,
> install the extras from the upstream project (for example `pip install ".[chroma]"`) before enabling the
> `--chromadb-*` flags or related environment variables.

The CLI and service default to in-process `NetworkX` + `NanoVectorDB`, so you can start locally without Neo4j/Chroma.

### Avoiding API rate limits

Every GraphRAG instance created by this sample now injects conservative throttling defaults:

```
best_model_max_async = 1
cheap_model_max_async = 1
embedding_func_max_async = 1
embedding_batch_num = 8
best_model_max_token_size = 4096
cheap_model_max_token_size = 2048
```

Override them via environment variables whenever you have higher quotas:

| Env var                             | GraphRAG kwarg                |
| ----------------------------------- | ----------------------------- |
| `GRAPH_RAG_BEST_MODEL_MAX_ASYNC`    | `best_model_max_async`        |
| `GRAPH_RAG_CHEAP_MODEL_MAX_ASYNC`   | `cheap_model_max_async`       |
| `GRAPH_RAG_EMBEDDING_MAX_ASYNC`     | `embedding_func_max_async`    |
| `GRAPH_RAG_EMBEDDING_BATCH`         | `embedding_batch_num`         |
| `GRAPH_RAG_BEST_MODEL_MAX_TOKENS`   | `best_model_max_token_size`   |
| `GRAPH_RAG_CHEAP_MODEL_MAX_TOKENS`  | `cheap_model_max_token_size`  |

These settings keep OpenAI from responding with `429 Too Many Requests` when you index larger corpora inside a Codespace.

## CLI usage

Build an index for the demo CSV (filters are optional and can be repeated):

```powershell
cd sample-app
py -3 poetry_graph_cli.py build `
    --csv ..\arabicpoetry_first20.csv `
    --workdir .\poetry_artifacts `
    --run-id poetry-demo `
    --filter meter=rajaz
```

Query the graph with the same run configuration.  You can pass structured constraints so the retriever knows which
attributes to favour (meter, rhyme, era, theme, etc.).

```powershell
py -3 poetry_graph_cli.py query `
    --run-id poetry-demo `
    --workdir .\poetry_artifacts `
    --question "أريد بيتًا في الفخر من بحر الطويل ينتهي بحرف الميم" `
    --constraint meter=الطويل `
    --constraint rhyme=م `
    --top-k 4
```

To inspect the available metadata (meters, eras, rhymes) before querying:

```powershell
py -3 poetry_graph_cli.py describe --csv ..\arabicpoetry_first20.csv
```

If you already have Neo4j/Chroma running (for example via `docker-compose.yml`), add the connection flags:

```powershell
py -3 poetry_graph_cli.py build `
    --neo4j-url bolt://localhost:7687 `
    --neo4j-user neo4j `
    --neo4j-password password `
    --chromadb-host localhost `
    --chromadb-port 8001
```

## Embedding the service

Use `PoetryGraphRAGService` when you need programmatic access (e.g. inside a FastAPI endpoint or an agent):

```python
from pathlib import Path
from poetry_graph_service import PoetryGraphRAGService

service = PoetryGraphRAGService(
    csv_path=Path("../arabicpoetry_first20.csv"),
    workdir=Path("./poetry_artifacts"),
    run_id="poetry-demo",
    mode="global",  # optional: default is "local"
    neo4j_url="bolt://localhost:7687",  # optional
    neo4j_user="neo4j",
    neo4j_password="password",
)

service.build_index(filters={"meter": "الطويل"})
contexts = service.retrieve(
    "أعطني بيتًا حماسيًا يفتخر بالبطولات",
    top_k=4,
    constraints={"meter": "الطويل", "rhyme": "م"}
)
```

The retrieved contexts already include a JSON blob (`StructuredMetadata`) with all poem attributes, so your downstream
LLM can reason about meter, rhyme, era, theme, and the two hemistichs simultaneously.

## Next steps

1. Replace the 20‑row sample with the full CSV and re-run `build`.
2. Add additional normalisation (e.g. map the unnamed Arabic columns to canonical keys such as `theme`, `subject`, etc.).
3. Wire the service into your existing generator so the retrieved contexts can condition the poem synthesis prompt.
