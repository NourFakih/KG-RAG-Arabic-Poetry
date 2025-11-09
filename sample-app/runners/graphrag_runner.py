from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple

import csv
import json
import os
import re
import sys

from _vendor_paths import NANO_GRAPHRAG_PATH  # noqa: F401

from nano_graphrag.graphrag import GraphRAG
from nano_graphrag.base import QueryParam
from nano_graphrag._storage import (
    Neo4jStorage,
    NetworkXStorage,
)

try:  # optional extras are not always bundled with nano-graphrag
    from nano_graphrag._storage import ChromaDBStorage
except ImportError:  # pragma: no cover - depends on install flavour
    ChromaDBStorage = None  # type: ignore

try:
    from nano_graphrag._storage import FireStoreKVStorage
except ImportError:  # pragma: no cover
    FireStoreKVStorage = None  # type: ignore

from .baserag_runner import BaseRagRunner

try:  # when runners is treated as a package, grab helper from parent
    from ..graphrag_limits import get_graphrag_limits
except ImportError:  # fallback for direct script execution
    from graphrag_limits import get_graphrag_limits  # type: ignore


_CHUNK_SPLIT = re.compile(r"--New Chunk--\\n")


class GraphRAGRunner(BaseRagRunner):
    def __init__(
        self,
        workdir: Path,
        *,
        mode: str = "local",
        run_id: str,
        cache_dir: Path,
        embedding_func,
        neo4j_url: str | None = None,
        neo4j_auth: Tuple[str, str] | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        chromadb_host: str | None = None,
        chromadb_port: int | None = None,
        chromadb_collection: str | None = None,
        chromadb_tenant: str | None = None,
        chromadb_database: str | None = None,
        chromadb_ssl: bool | None = None,
        chromadb_headers: Dict[str, str] | None = None,
        chromadb_reset: bool = False,
        use_firestore_kv: bool = False,
        firestore_project_id: str | None = None,
        firestore_collection_prefix: str | None = None,
        **kwargs
    ):
        super().__init__(workdir, **kwargs)

        self.mode      = mode
        self.run_id    = run_id
        self.cache_dir = cache_dir
        self.workdir   = workdir

        self.debug_log = self.workdir / "debug_graph.log"
        self.debug_log.parent.mkdir(parents=True, exist_ok=True)
        self.debug_log.write_text("", encoding="utf-8")  # clear old log

        graph_kwargs: Dict = {**kwargs}
        for key, value in get_graphrag_limits().items():
            graph_kwargs.setdefault(key, value)
        addon_params = dict(graph_kwargs.pop("addon_params", {}))
        vector_kwargs: Dict = dict(graph_kwargs.pop("vector_db_storage_cls_kwargs", {}))

        if neo4j_url:
            if neo4j_auth is None and neo4j_user is not None and neo4j_password is not None:
                neo4j_auth = (neo4j_user, neo4j_password)
            if neo4j_auth is None:
                raise ValueError("neo4j_url provided without credentials; supply neo4j_auth or user/password")
            addon_params.update({"neo4j_url": neo4j_url, "neo4j_auth": neo4j_auth})
            graph_storage_cls = Neo4jStorage
        else:
            graph_storage_cls = graph_kwargs.pop("graph_storage_cls", NetworkXStorage)

        def _sanitize(text: str) -> str:
            clean = re.sub(r"[^0-9a-zA-Z_\-]+", "_", text.strip())
            clean = re.sub(r"_+", "_", clean)
            return clean.strip("_") or "default"

        def _coerce_bool(val, fallback=None):
            if val is None:
                return fallback
            if isinstance(val, bool):
                return val
            val_str = str(val).strip().lower()
            if val_str in {"1", "true", "yes", "on", "y"}:
                return True
            if val_str in {"0", "false", "no", "off", "n"}:
                return False
            return fallback

        env_chromadb_host = chromadb_host or os.getenv("CHROMADB_HOST")
        if env_chromadb_host:
            if ChromaDBStorage is None:
                raise ImportError(
                    "ChromaDBStorage is unavailable in this nano-graphrag build. "
                    "Remove CHROMADB_* settings or install the Chroma extra."
                )
            graph_kwargs["vector_db_storage_cls"] = ChromaDBStorage
            vector_kwargs["host"] = env_chromadb_host
            port_value = (
                chromadb_port if chromadb_port is not None else os.getenv("CHROMADB_PORT")
            )
            if port_value is not None:
                vector_kwargs["port"] = int(port_value)
            if chromadb_ssl is not None:
                vector_kwargs["ssl"] = chromadb_ssl
            elif "CHROMADB_SSL" in os.environ:
                vector_kwargs["ssl"] = os.getenv("CHROMADB_SSL")
            if chromadb_headers:
                vector_kwargs["headers"] = chromadb_headers
            tenant_value = chromadb_tenant or os.getenv("CHROMADB_TENANT")
            if tenant_value:
                vector_kwargs["tenant"] = tenant_value
            database_value = chromadb_database or os.getenv(
                "CHROMADB_DATABASE", "default_database"
            )
            if database_value:
                vector_kwargs["database"] = database_value
            if chromadb_reset:
                reset_value = True
            elif "CHROMADB_RESET_COLLECTION" in os.environ:
                reset_value = os.getenv("CHROMADB_RESET_COLLECTION")
            else:
                reset_value = None
            if reset_value is not None:
                bool_reset = _coerce_bool(reset_value, fallback=None)
                if bool_reset is not None:
                    vector_kwargs["reset_collection"] = bool_reset
            chosen_collection = (
                chromadb_collection or os.getenv("CHROMADB_COLLECTION")
            )
            if chosen_collection is None:
                chosen_collection = _sanitize(run_id)
            else:
                chosen_collection = _sanitize(chosen_collection)
            vector_kwargs["collection_name"] = chosen_collection

        env_firestore_project = firestore_project_id or os.getenv("FIRESTORE_PROJECT_ID")
        env_firestore_prefix = firestore_collection_prefix or os.getenv("FIRESTORE_COLLECTION_PREFIX")
        if use_firestore_kv or env_firestore_project:
            if FireStoreKVStorage is None:
                raise ImportError(
                    "FireStoreKVStorage is unavailable in this nano-graphrag build. "
                    "Disable Firestore KV support or install the Firestore extra."
                )
            graph_kwargs.setdefault("key_string_value_json_storage_cls", FireStoreKVStorage)
            if env_firestore_project:
                graph_kwargs["firestore_project_id"] = env_firestore_project
            if env_firestore_prefix:
                graph_kwargs["firestore_collection_prefix"] = env_firestore_prefix

        if addon_params:
            graph_kwargs["addon_params"] = addon_params
        graph_kwargs.setdefault("graph_storage_cls", graph_storage_cls)
        graph_kwargs.setdefault("enable_naive_rag", True)
        if vector_kwargs:
            graph_kwargs["vector_db_storage_cls_kwargs"] = vector_kwargs

        self._rag = GraphRAG(
            working_dir=str(workdir / ".graphrag_cache"),
            embedding_func=embedding_func,
            **graph_kwargs,
        )
        kv_file = self.cache_dir / "kv_store_text_chunks.json"
        if kv_file.exists():
            data = json.loads(kv_file.read_text(encoding="utf-8"))
            self._id_by_text_head  = {v["content"][:120]: k for k, v in data.items()}
            self._id_by_full_content = {v["content"]: k for k, v in data.items()}
            self._log(
                "[build_index] head->ID samples:",
                list(self._id_by_text_head.items())[:5]
            )
        else:
            self._id_by_text_head = {}
            self._id_by_full_content = {}
        self._entity_hits_cache: Dict[str, List[str]] = {}
        self._chunk_cache: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ utils
    def _log(self, *msgs):
        with open(self.debug_log, "a", encoding="utf-8") as f:
            f.write(" ".join(str(m) for m in msgs) + "\n")

    def _clean(self, text: str) -> str:
        t = text.strip().strip('"\'')
        return "" if t.lower() == "content" else t

    # ------------------------------------------------------------------ index
    def build_index(self, docs: List[str]):
        self._chunk_cache.clear()
        self._entity_hits_cache.clear()
        self._rag.insert(docs)

        kv_file = self.cache_dir / "kv_store_text_chunks.json"
        if kv_file.exists():
            data = json.loads(kv_file.read_text(encoding="utf-8"))
            self._id_by_text_head  = {v["content"][:120]: k for k, v in data.items()}
            self._id_by_full_content = {v["content"]: k for k, v in data.items()}
            self._log(
                "[build_index] head->ID samples:",
                list(self._id_by_text_head.items())[:5]
            )
        else:
            self._id_by_text_head = {}
            self._id_by_full_content = {}

    # ---------------------------------------------------------------- splitter
    def _split_context(self, raw: str) -> List[str]:
        """
        Extract individual chunk texts from GraphRAG / HiRAG responses.

        Handles three formats:
        1. Explicit `--New Chunk--` delimiters.
        2. The JSON array under the `-----Document Chunks(DC)-----` header.
        3. Legacy CSV under `-----Source Documents-----`.
        """
        # 1) custom delimiter
        if _CHUNK_SPLIT.search(raw):
            return _CHUNK_SPLIT.split(raw)

        # 2) structured "Document Chunks" JSON
        if "-----Document Chunks" in raw:
            try:
                # isolate the block starting at the header
                start = raw.index("-----Document Chunks")
                block = raw[start:]

                # first ``` and the matching closing ```
                first = block.find("```")
                if first != -1:
                    second = block.find("```", first + 3)
                    if second != -1:
                        json_text = block[first + 3 : second]
                        objs = json.loads(json_text)
                        return [obj["content"] for obj in objs if "content" in obj]
            except Exception as exc:
                self._log("[_split_context] failed to parse Document Chunks:", exc)

        # 3) legacy CSV
        if "-----Source Documents-----" in raw:
            part = raw.split("-----Source Documents-----")[-1]
            rows = [l.strip() for l in part.splitlines() if l.strip()]
            return [" ".join(r.split(",")[1:]).strip('"') for r in rows[1:]]

        # fallback
        return [raw]

    def _map_ids(self, chunks_text: List[str]) -> List[str]:
        ids: List[str] = []
        for txt in chunks_text:
            cid = self._id_by_full_content.get(txt)
            if not cid:
                cid = self._id_by_text_head.get(txt[:120], "")
            ids.append(cid or "")
        self._log("[ _map_ids ]", ids)
        return ids

    # ---------------------------------------------------------------- retrieve
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8
    ) -> List[str]:
        self._log(f"[retrieve] query='{query}' top_k={top_k}")

        # 1) run the RAG
        param  = QueryParam(mode=self.mode, only_need_context=True, top_k=top_k)
        result = self._rag.query(query, param)
        if result is None:
            self._log("[retrieve] GraphRAG.query returned None")
            self._entity_hits_cache.pop(query, None)
            self._chunk_cache.pop(query, None)
            return []

        ctx_list = [result] if isinstance(result, str) else list(result)
        self._log("[retrieve] raw ctx_list:", ctx_list[:2])

        chunks = []
        entity_hits = set()

        # 2) split & clean
        for block in ctx_list:
            parts = self._split_context(block)
            self._log("[retrieve] raw parts:", parts[:3])

            cleaned = [c for p in parts if (c := self._clean(p))]
            self._log("[retrieve] cleaned parts:", cleaned[:3])
            chunks.extend(cleaned)

            # 3) extract Entities CSV from the full block
            m = re.search(r'-----Entities-----\s*```csv\s*([\s\S]+?)```', block)
            if not m:
                continue

            csv_text = m.group(1).strip()
            reader = csv.reader(StringIO(csv_text))
            headers = next(reader, None)
            if not headers:
                continue

            # FIXED: normalize header names by stripping quotes & whitespace
            headers_clean = [h.strip().strip('"') for h in headers]
            if "entity" not in headers_clean:
                continue

            ent_idx = headers_clean.index("entity")
            for row in reader:
                if len(row) > ent_idx:
                    name = row[ent_idx].strip().strip('"')
                    if name:
                        # GraphML node IDs are the label in quotes
                        entity_hits.add(f'"{name}"')

        self._log("[retrieve] plain chunks:", chunks)
        entity_list = sorted(entity_hits)
        self._log("[retrieve] entity_hits:", entity_list)

        self._entity_hits_cache[query] = entity_list
        self._chunk_cache[query] = chunks

        return chunks

    # --------------------------------------------------------------- dump index
    def dump_index(self, qid_hits, out_path, run_id):
        payload = {
            "run_id": run_id,
            "queries": [],
        }

        for qid, contexts in qid_hits.items():
            mapped_chunk_ids: List[str] = []
            for ctx in contexts:
                chunk_ids = self._map_ids(self._split_context(ctx))
                mapped_chunk_ids.extend(chunk_ids)
            deduped_ids = [cid for cid in dict.fromkeys(mapped_chunk_ids) if cid]
            entities = self._entity_hits_cache.get(qid)
            if entities is None:
                for query, cached_chunks in self._chunk_cache.items():
                    if cached_chunks == contexts:
                        entities = self._entity_hits_cache.get(query, [])
                        break
            payload["queries"].append(
                {
                    "qid": qid,
                    "contexts": contexts,
                    "chunk_ids": deduped_ids,
                    "entities": entities or [],
                }
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
