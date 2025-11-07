from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from poetry_graph_service import PoetryGraphRAGService


HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "arabicpoetry_first20.csv"


def _parse_kv(items: List[str] | None) -> Dict[str, object]:
    if not items:
        return {}
    result: Dict[str, object] = {}
    for raw in items:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(f"Invalid filter '{raw}'. Use key=value.")
        key, value = raw.split("=", 1)
        values = [v.strip() for v in value.split(",") if v.strip()]
        result[key.strip()] = values if len(values) > 1 else (values[0] if values else "")
    return result


def _collect_runner_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    mapping = {
        "neo4j_url": args.neo4j_url,
        "neo4j_user": args.neo4j_user,
        "neo4j_password": args.neo4j_password,
        "chromadb_host": args.chromadb_host,
        "chromadb_port": args.chromadb_port,
        "chromadb_collection": args.chromadb_collection,
        "chromadb_tenant": args.chromadb_tenant,
        "chromadb_database": args.chromadb_database,
        "chromadb_reset": args.chromadb_reset,
    }
    return {k: v for k, v in mapping.items() if v not in {None, ""}}


def _build_service(args: argparse.Namespace) -> PoetryGraphRAGService:
    return PoetryGraphRAGService(
        csv_path=args.csv,
        workdir=args.workdir,
        run_id=args.run_id,
        mode=args.mode,
        **_collect_runner_kwargs(args),
    )


def _cmd_build(args: argparse.Namespace) -> None:
    service = _build_service(args)
    filters = _parse_kv(args.filter)
    docs = service.build_index(filters=filters)
    print(f"Indexed {len(docs)} poem rows from {args.csv}")


def _cmd_query(args: argparse.Namespace) -> None:
    service = _build_service(args)
    if args.auto_build:
        service.build_index(filters=_parse_kv(args.filter))
    contexts = service.retrieve(
        args.question,
        top_k=args.top_k,
        constraints=_parse_kv(args.constraint),
    )
    if not contexts:
        print("No context returned.")
        return
    for idx, ctx in enumerate(contexts, start=1):
        print(f"\n--- Context {idx} ---\n{ctx.strip()}\n")


def _cmd_describe(args: argparse.Namespace) -> None:
    service = _build_service(args)
    stats = service.describe_corpus()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to the Arabic poetry CSV.")
    parent.add_argument("--workdir", default=str(HERE / "poetry_artifacts"), help="Directory for GraphRAG caches.")
    parent.add_argument("--run-id", default=None, help="Custom run identifier. Defaults to timestamp.")
    parent.add_argument("--mode", default="local", help="GraphRAG query mode (local/global).")
    parent.add_argument("--neo4j-url", help="Neo4j bolt URL if using external graph storage.")
    parent.add_argument("--neo4j-user")
    parent.add_argument("--neo4j-password")
    parent.add_argument("--chromadb-host")
    parent.add_argument("--chromadb-port", type=int)
    parent.add_argument("--chromadb-collection")
    parent.add_argument("--chromadb-tenant")
    parent.add_argument("--chromadb-database")
    parent.add_argument("--chromadb-reset", action="store_true")

    parser = argparse.ArgumentParser(description="GraphRAG helper for the Arabic poetry corpus.", parents=[parent])
    sub = parser.add_subparsers(dest="command", required=True)

    build_cmd = sub.add_parser("build", parents=[parent], help="Build or rebuild the poetry index.")
    build_cmd.add_argument("--filter", action="append", help="Restrict ingestion: e.g. meter=rajaz or era=Abbasid.")
    build_cmd.set_defaults(func=_cmd_build)

    query_cmd = sub.add_parser("query", parents=[parent], help="Retrieve contexts for a user query.")
    query_cmd.add_argument("question", help="User instruction or description of the desired poem.")
    query_cmd.add_argument("--top-k", type=int, default=5)
    query_cmd.add_argument("--constraint", action="append", help="Metadata constraints, e.g. meter=الطويل.")
    query_cmd.add_argument("--filter", action="append", help="Filters applied when auto-building.")
    query_cmd.add_argument("--auto-build", action="store_true", help="Ingest before querying.")
    query_cmd.set_defaults(func=_cmd_query)

    describe_cmd = sub.add_parser("describe", parents=[parent], help="Print dataset metadata.")
    describe_cmd.set_defaults(func=_cmd_describe)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
