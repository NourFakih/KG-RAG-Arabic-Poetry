from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
class BaseRagRunner(ABC):
    """Minimal façade for any RAG variant."""
    def __init__(self, workdir, **kwargs):
    
        self.workdir = workdir

    @abstractmethod
    def build_index(self, docs: List[str]): ...
    
    @abstractmethod
    def retrieve(self, query: str, *, top_k: int = 8) -> List[str]: ...
    @abstractmethod
    def dump_index(
        self,
        qid_hits: dict[str, list[str]],
        out_path: Path,
        run_id: str,
    ) -> None:
        """Write enriched index.json for the visualizer."""