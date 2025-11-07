from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "nano-graphrag"))

from nano_graphrag.graphrag import GraphRAG
from nano_graphrag.base import QueryParam
from nano_graphrag._storage import Neo4jStorage, ChromaDBStorage
from nano_graphrag._utils import logger


DEFAULT_BOOK_PATH = HERE / "book.txt"
DEFAULT_WORKDIR = HERE / "build_artifacts"


def _load_env() -> dict[str, str]:
    """
    Merge process env with the closest `.env` file (current dir first, then scripts/.env).
    Values already present in the process environment win.
    """
    env: dict[str, str] = dict(os.environ)
    candidates = [HERE / ".env", HERE / "scripts" / ".env"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env.setdefault(key.strip(), value.strip())
    return env


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def build() -> None:
    env = _load_env()

    required = ["OPENAI_API_KEY", "NEO4J_URL", "NEO4J_USER", "NEO4J_PASSWORD"]
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

    # Ensure OpenAI picks up the key when GraphRAG initialises.
    os.environ.setdefault("OPENAI_API_KEY", env["OPENAI_API_KEY"])
    if "OPENAI_RESPONSE_MODEL" in env:
        os.environ.setdefault("OPENAI_RESPONSE_MODEL", env["OPENAI_RESPONSE_MODEL"])

    neo4j_url = env["NEO4J_URL"]
    neo4j_auth = (env["NEO4J_USER"], env["NEO4J_PASSWORD"])

    chroma_host = env.get("CHROMADB_HOST", "localhost")
    chroma_port = int(env.get("CHROMADB_PORT", 8001))
    chroma_database = env.get("CHROMADB_DATABASE", "default_database")
    chroma_tenant = env.get("CHROMADB_TENANT", "default_tenant")
    chroma_reset = _as_bool(env.get("CHROMADB_RESET_COLLECTION"), True)
    chroma_collection = env.get("CHROMADB_COLLECTION")

    def _sanitize_collection(name: str) -> str:
        cleaned = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.strip().lower()
        )
        cleaned = cleaned.strip("-_")
        if len(cleaned) < 3:
            cleaned = f"{cleaned or 'collection'}_{os.getpid()}"
        return cleaned[:63]

    book_path = Path(env.get("BOOK_PATH") or DEFAULT_BOOK_PATH)
    if not book_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {book_path}")

    text = book_path.read_text(encoding="utf-8")
    working_dir = Path(env.get("GRAPHRAG_WORKDIR") or DEFAULT_WORKDIR)
    working_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building GraphRAG index")
    rag = GraphRAG(
        working_dir=str(working_dir),
        graph_storage_cls=Neo4jStorage,
        vector_db_storage_cls=ChromaDBStorage,
        vector_db_storage_cls_kwargs={
            "host": chroma_host,
            "port": chroma_port,
            "database": chroma_database,
            "tenant": chroma_tenant,
            "reset_collection": chroma_reset,
            **(
                {"collection_name": _sanitize_collection(chroma_collection)}
                if chroma_collection
                else {}
            ),
        },
        addon_params={
            "neo4j_url": neo4j_url,
            "neo4j_auth": neo4j_auth,
        },
        enable_naive_rag=True,
    )

    logger.info("Indexing %s", book_path)
    rag.insert([text])

    probe_question = env.get("TEST_QUESTION", "What is this book about?")
    contexts = rag.query(
        probe_question,
        QueryParam(mode="global", only_need_context=True, top_k=3),
    )

    print("Index build complete.")
    print(f"- Working directory: {working_dir}")
    print(f"- Neo4j namespace label: {rag.chunk_entity_relation_graph.namespace}")
    print(f"- Probe question: {probe_question}")
    print(f"- Returned {len(contexts) if contexts else 0} context chunk(s).")

    if contexts:
        preview = contexts[0].replace("\n", " ")[:200]
        print(f"  First context snippet: {preview}{'...' if len(contexts[0]) > 200 else ''}")

    print("\nNext steps to verify:")
    print("1. Neo4j nodes count: docker compose exec neo4j cypher-shell -u neo4j -p <password> \"MATCH (n) RETURN count(n);\"")
    print("2. Chroma collections: curl http://localhost:8001/api/v1/collections")


if __name__ == "__main__":
    try:
        build()
    except Exception as exc:  # pragma: no cover
        print(f"Build failed: {exc}", file=sys.stderr)
        sys.exit(1)
