import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

texts = [
    "ما هي إجراءات فتح حساب بنكي؟",
    "إيه الأوراق المطلوبة عشان أعمل حساب في البنك؟",
    "ما هو سعر الدولار اليوم؟"
]

for text in texts:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    embedding = result.embeddings[0].values

    print("\nTEXT:", text)
    print("Embedding dimensions:", len(embedding))
    print("First 10 values:", embedding[:10])