"""One-time model download. Runtime never downloads or calls hosted APIs."""

from pathlib import Path

from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> None:
    embedding_name = "sentence-transformers/all-MiniLM-L6-v2"
    generation_name = "google/flan-t5-small"
    print(f"Downloading {embedding_name} into {MODELS / 'embeddings'}")
    AutoTokenizer.from_pretrained(embedding_name, revision="c9745ed").save_pretrained(MODELS / "embeddings")
    AutoModel.from_pretrained(embedding_name, revision="c9745ed").save_pretrained(MODELS / "embeddings")
    print(f"Downloading {generation_name} into {MODELS / 'generation'}")
    AutoTokenizer.from_pretrained(generation_name, revision="f6d0c2c").save_pretrained(MODELS / "generation")
    AutoModelForSeq2SeqLM.from_pretrained(generation_name, revision="f6d0c2c").save_pretrained(MODELS / "generation")
    print("Done. The agent can now run with network access disabled.")


if __name__ == "__main__":
    main()