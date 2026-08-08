from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelConfig:
    """Pinned local model configuration.

    The model names are used only during the one-time download step. Runtime
    loading is local-only (`local_files_only=True`).
    """

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "c9745ed"
    generation_model: str = "google/flan-t5-small"
    generation_revision: str = "f6d0c2c"
    device: str = "cpu"


@dataclass(frozen=True)
class AgentConfig:
    data_dir: Path = PROJECT_ROOT
    model_dir: Path = PROJECT_ROOT / "models"
    max_revisions: int = 1
    top_k: int = 5
    use_huggingface: bool = True
    model: ModelConfig = ModelConfig()

    @property
    def knowledge_base_dir(self) -> Path:
        return self.data_dir / "knowledge_base"

    @property
    def schema_path(self) -> Path:
        return self.data_dir / "output_schema.json"

    @property
    def cases_path(self) -> Path:
        return self.data_dir / "resolved_cases.json"