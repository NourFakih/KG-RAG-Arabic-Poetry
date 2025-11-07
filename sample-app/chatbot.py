from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "nano-graphrag"))

from openai import OpenAI
from nano_graphrag.graphrag import GraphRAG  # type: ignore
from nano_graphrag.base import QueryParam  # type: ignore
from nano_graphrag._storage import Neo4jStorage, ChromaDBStorage  # type: ignore
from nano_graphrag._utils import logger  # type: ignore


DEFAULT_WORKDIR = HERE / "build_artifacts"
NEO4J_LOG_PATH = HERE / "neo4jlog.txt"
CHROMA_LOG_PATH = HERE / "chromadblog.txt"


def _load_env() -> dict[str, str]:
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


def _sanitize_collection(name: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.strip().lower()
    )
    cleaned = cleaned.strip("-_")
    if len(cleaned) < 3:
        cleaned = f"{cleaned or 'collection'}_{os.getpid()}"
    return cleaned[:63]


def initialise_rag() -> GraphRAG:
    env = _load_env()

    required = ["OPENAI_API_KEY", "NEO4J_URL", "NEO4J_USER", "NEO4J_PASSWORD"]
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

    os.environ.setdefault("OPENAI_API_KEY", env["OPENAI_API_KEY"])
    if env.get("OPENAI_RESPONSE_MODEL"):
        os.environ.setdefault("OPENAI_RESPONSE_MODEL", env["OPENAI_RESPONSE_MODEL"])

    chat_model = env.get("CHATBOT_MODEL", env.get("OPENAI_RESPONSE_MODEL", "gpt-4o-mini"))

    neo4j_url = env["NEO4J_URL"]
    neo4j_auth = (env["NEO4J_USER"], env["NEO4J_PASSWORD"])

    chroma_host = env.get("CHROMADB_HOST", "localhost")
    chroma_port = int(env.get("CHROMADB_PORT", 8001))
    chroma_database = env.get("CHROMADB_DATABASE", "default_database")
    chroma_tenant = env.get("CHROMADB_TENANT", "default_tenant")
    chroma_reset = _as_bool(env.get("CHROMADB_RESET_COLLECTION"), False)
    chroma_collection = env.get("CHROMADB_COLLECTION")

    working_dir = Path(env.get("GRAPHRAG_WORKDIR") or DEFAULT_WORKDIR)
    working_dir.mkdir(parents=True, exist_ok=True)

    vector_kwargs: dict[str, object] = {
        "host": chroma_host,
        "port": chroma_port,
        "database": chroma_database,
        "tenant": chroma_tenant,
        "reset_collection": chroma_reset,
    }
    if chroma_collection:
        vector_kwargs["collection_name"] = _sanitize_collection(chroma_collection)

    rag = GraphRAG(
        working_dir=str(working_dir),
        graph_storage_cls=Neo4jStorage,
        vector_db_storage_cls=ChromaDBStorage,
        vector_db_storage_cls_kwargs=vector_kwargs,
        addon_params={
            "neo4j_url": neo4j_url,
            "neo4j_auth": neo4j_auth,
        },
        enable_naive_rag=True,
    )
    return rag, chat_model


async def _query_chroma(storage: ChromaDBStorage | None, query: str, top_k: int = 5):
    if storage is None:
        return []
    try:
        return await storage.query(query, top_k=top_k)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to query ChromaDB for logging: %s", exc)
        return []


async def _fetch_neo4j_nodes(storage: Neo4jStorage, limit: int = 10):
    query = f"""
        MATCH (n:`{storage.namespace}`)
        OPTIONAL MATCH (n)-[r]-(m)
        WITH n, size((n)--()) AS degree
        RETURN n.id AS id,
               coalesce(n.entity_name, n.title, n.id) AS label,
               degree,
               coalesce(n.communityIds, []) AS communityIds,
               properties(n) AS properties
        ORDER BY degree DESC
        LIMIT $limit
    """
    records = []
    try:
        async with storage.async_driver.session() as session:
            result = await session.run(query, limit=limit)
            async for record in result:
                records.append(record.data())
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to query Neo4j for logging: %s", exc)
    return records


def _write_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_json_safe(payload), ensure_ascii=False))
        fh.write("\n")


def _json_safe(value):
    """
    Recursively convert values into JSON-serialisable structures.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    # cover numpy scalars, Decimal, etc.
    try:
        if hasattr(value, "item"):
            return _json_safe(value.item())
    except Exception:  # pragma: no cover
        pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover
            return str(value)
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


def format_answer(answer) -> str:
    if answer is None:
        return "No answer returned."
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, dict):
        return json.dumps(answer, ensure_ascii=False, indent=2)
    if isinstance(answer, (list, tuple)):
        return "\n".join(str(item) for item in answer)
    return str(answer)


def main() -> None:
    rag, chat_model = initialise_rag()
    client = OpenAI()
    print("Chatbot ready. Type your question, or 'exit' to quit.")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        timestamp = datetime.utcnow().isoformat() + "Z"
        try:
            contexts = rag.query(
                question,
                QueryParam(mode="global", only_need_context=True, top_k=5),
            )
        except Exception as exc:
            print(f"[Error] Failed to retrieve context: {exc}")
            continue

        contexts_list = (
            contexts
            if isinstance(contexts, list)
            else [contexts] if isinstance(contexts, str) else []
        )

        context_text = "\n\n".join(
            f"Context {idx + 1}:\n{ctx}" for idx, ctx in enumerate(contexts_list)
        ) or "No relevant context was retrieved."

        print(f"[Retrieved {len(contexts_list)} context chunk(s)] : {context_text}\n")

        user_content = (
            "Use the context below to answer the user's question. "
            "If the context is insufficient, say you do not know.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}"
        )

        try:
            completion = client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant that only answers using the supplied context."},
                    {"role": "user", "content": user_content},
                ],
            )
            answer_text = completion.choices[0].message.content if completion.choices else "No answer returned."
        except Exception as exc:
            print(f"[Error] Failed to generate answer via OpenAI: {exc}")
            answer_text = "I wasn't able to generate an answer."

        print(f"Bot: {answer_text}\n")

        chroma_hits = asyncio.run(_query_chroma(rag.chunks_vdb, question, top_k=5))
        neo4j_hits = asyncio.run(_fetch_neo4j_nodes(rag.chunk_entity_relation_graph, limit=10))

        try:
            chroma_payload = {
                "timestamp": timestamp,
                "query": question,
                "contexts": contexts_list,
                "vector_hits": chroma_hits,
                "answer": answer_text,
                "prompt": user_content,
            }
            _write_log(CHROMA_LOG_PATH, chroma_payload)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write chroma log: %s", exc)

        try:
            neo4j_payload = {
                "timestamp": timestamp,
                "query": question,
                "namespace": rag.chunk_entity_relation_graph.namespace,
                "top_nodes": neo4j_hits,
                "answer": answer_text,
                "prompt": user_content,
            }
            _write_log(NEO4J_LOG_PATH, neo4j_payload)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write neo4j log: %s", exc)

        print(f"[Logged] Neo4j -> {NEO4J_LOG_PATH.name}, Chroma -> {CHROMA_LOG_PATH.name}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)
