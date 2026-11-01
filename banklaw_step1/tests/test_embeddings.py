from banklaw_step1.embeddings import embed_text, embed_texts
from numpy import dot
from numpy.linalg import norm


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return float(dot(a, b) / (norm(a) * norm(b)))


def test_similar_arabic_legal_texts_are_closer():
    query = embed_text("ما هي حقوق عملاء البنوك؟")
    similar = embed_text("تعليمات حماية حقوق عملاء البنوك")
    unrelated = embed_text("أسعار الذهب في السوق العالمية")

    similar_score = cosine_similarity(query, similar)
    unrelated_score = cosine_similarity(query, unrelated)

    assert similar_score > unrelated_score


def test_embed_texts_returns_one_vector_per_text():
    texts = [
        "حقوق عملاء البنوك",
        "إجراءات تقديم الشكاوى",
        "الحسابات الراكدة",
    ]

    vectors = embed_texts(texts)

    assert isinstance(vectors, list)
    assert len(vectors) == len(texts)
    assert all(len(vector) == 1024 for vector in vectors)
    assert all(
        isinstance(value, float)
        for vector in vectors
        for value in vector
    )