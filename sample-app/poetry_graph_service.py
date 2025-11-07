from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "nano-graphrag"))
sys.path.insert(0, str(HERE.parent / "nano-graphrag"))

from nano_graphrag._llm import openai_embedding  # type: ignore

from runners.graphrag_runner import GraphRAGRunner


def _clean(value: str | None) -> str:
    return (value or "").strip()


@dataclass
class PoetryRecord:
    poem_title: str = ""
    first_hemistich: str = ""
    second_hemistich: str = ""
    poet: str = ""
    meter: str = ""
    sub_meter: str = ""
    era: str = ""
    rhyme: str = ""
    type_en: str = ""
    type_ar: str = ""
    link: str = ""
    gender: str = ""
    extras: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "PoetryRecord":
        base_keys = {
            "poem_title",
            "first_hemistich",
            "second_hemistich",
            "poet",
            "meter",
            "sub_meter",
            "era",
            "rhyme",
            "type_en",
            "type_ar",
            "link",
            "gender",
        }
        cleaned = {key: _clean(row.get(key)) for key in base_keys}
        extras = {}
        for key, value in row.items():
            if key not in base_keys:
                extras[key] = _clean(value)
        return cls(extras=extras, **cleaned)

    def to_metadata(self) -> Dict[str, str]:
        payload = {
            "poem_title": self.poem_title,
            "first_hemistich": self.first_hemistich,
            "second_hemistich": self.second_hemistich,
            "poet": self.poet,
            "meter": self.meter,
            "sub_meter": self.sub_meter,
            "era": self.era,
            "rhyme": self.rhyme,
            "type_en": self.type_en,
            "type_ar": self.type_ar,
            "link": self.link,
            "gender": self.gender,
        }
        payload.update(self.extras)
        return {k: v for k, v in payload.items() if v}

    def matches_filters(self, filters: Dict[str, Sequence[str] | str]) -> bool:
        if not filters:
            return True
        metadata = self.to_metadata()
        for key, expected in filters.items():
            current = metadata.get(key)
            if current is None:
                return False
            if isinstance(expected, (list, tuple, set)):
                normalized = {str(v).strip().lower() for v in expected}
                if current.strip().lower() not in normalized:
                    return False
                continue
            if current.strip().lower() != str(expected).strip().lower():
                return False
        return True

    def to_document(self) -> str:
        metadata = self.to_metadata()
        header = self.poem_title or f"{self.poet or 'قصيدة مجهولة'} ({self.meter or 'meter-unknown'})"
        tags = ", ".join(f"{k}: {v}" for k, v in metadata.items() if k not in {"first_hemistich", "second_hemistich"})
        verse = " — ".join(part for part in [self.first_hemistich, self.second_hemistich] if part)
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        english = (
            f"This poem uses the {self.meter or 'unknown'} meter, rhymes on {self.rhyme or 'unknown'}, "
            f"and speaks about the theme hinted by {self.type_en or metadata.get('theme', 'unspecified')}."
        )
        return (
            f"ArabicPoemRecord\n"
            f"Title: {header}\n"
            f"Poet: {self.poet or 'غير معروف'} | Era: {self.era or 'غير محدد'} | Gender: {self.gender or 'غير معين'}\n"
            f"Attributes: {tags}\n"
            f"StructuredMetadata: {metadata_json}\n"
            f"EnglishSummary: {english}\n"
            f"ArabicVerse: {verse}\n"
        )


class PoetryGraphRAGService:
    """
    High-level helper that loads the CSV, builds GraphRAG indices, and exposes build/retrieve methods.
    """

    def __init__(
        self,
        csv_path: Path | str,
        *,
        workdir: Path | str | None = None,
        run_id: str | None = None,
        mode: str = "local",
        cache_subdir: str = "poetry_cache",
        embedding_func=None,
        **runner_kwargs,
    ):
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV corpus not found: {self.csv_path}")

        self.workdir = Path(workdir or HERE / "poetry_artifacts").resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = self.workdir / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or f"poetry-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

        if embedding_func is None:
            embedding_func = openai_embedding

        runner_kwargs.setdefault("mode", mode)
        self._runner = GraphRAGRunner(
            workdir=self.workdir,
            run_id=self.run_id,
            cache_dir=self.cache_dir,
            embedding_func=embedding_func,
            **runner_kwargs,
        )

        self._records: List[PoetryRecord] | None = None

    # ------------------------------------------------------------------ dataset
    def _load_records(self) -> List[PoetryRecord]:
        if self._records is not None:
            return self._records
        with self.csv_path.open(encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = [PoetryRecord.from_row(row) for row in reader if any(row.values())]
        if not rows:
            raise ValueError(f"No rows found in {self.csv_path}")
        self._records = rows
        return rows

    def _filter_records(self, filters: Dict[str, Sequence[str] | str] | None) -> List[PoetryRecord]:
        records = self._load_records()
        if not filters:
            return records
        return [record for record in records if record.matches_filters(filters)]

    def _format_constraints(self, constraints: Dict[str, str | Sequence[str]] | None) -> str:
        if not constraints:
            return ""
        pieces = []
        for key, value in constraints.items():
            if isinstance(value, (list, tuple, set)):
                joined = ", ".join(str(v) for v in value)
            else:
                joined = str(value)
            pieces.append(f"{key}: {joined}")
        return "\n".join(pieces)

    # ------------------------------------------------------------------ API
    def build_index(self, *, filters: Dict[str, Sequence[str] | str] | None = None) -> List[str]:
        docs = [record.to_document() for record in self._filter_records(filters)]
        if not docs:
            raise ValueError("No documents to index; broaden your filters.")
        self._runner.build_index(docs)
        return docs

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        constraints: Dict[str, str | Sequence[str]] | None = None,
    ) -> List[str]:
        constraint_block = self._format_constraints(constraints)
        full_query = query
        if constraint_block:
            full_query = f"{query}\n\nقيود مطلوبة:\n{constraint_block}"
        return self._runner.retrieve(full_query, top_k=top_k)

    def dump_index(self, qid_hits: Dict[str, List[str]], out_path: Path) -> None:
        self._runner.dump_index(qid_hits, out_path, self.run_id)

    # ------------------------------------------------------------------ helpers
    def describe_corpus(self) -> Dict[str, object]:
        records = self._load_records()
        meters = sorted({record.meter for record in records if record.meter})
        rhymes = sorted({record.rhyme for record in records if record.rhyme})
        eras = sorted({record.era for record in records if record.era})
        return {
            "csv_path": str(self.csv_path),
            "records": len(records),
            "meters": meters,
            "rhymes": rhymes,
            "eras": eras,
        }
