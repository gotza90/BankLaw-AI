from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"

_model = SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    vector = _model.encode(
        text,
        normalize_embeddings=True,
    )

    return vector.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = _model.encode(
        texts,
        normalize_embeddings=True,
    )

    return vectors.tolist()