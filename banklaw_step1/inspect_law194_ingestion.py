from pathlib import Path

from banklaw_step1.ingestion import ingest_directory


source = Path("data")
temp_dir = Path("_law194_only")
temp_dir.mkdir(exist_ok=True)

pdf = source / "CBE Law No. 194 of 2020.pdf"
target = temp_dir / pdf.name

if not target.exists():
    target.write_bytes(pdf.read_bytes())

manifest = source / "corpus_manifest.json"

documents = ingest_directory(
    temp_dir,
    manifest_path=manifest,
)

print("documents =", len(documents))

for i, document in enumerate(documents[:10], start=1):
    print(f"\n===== DOCUMENT {i} =====")
    print("section_number =", document["metadata"]["section_number"])
    print("section_level =", document["metadata"]["section_level"])
    print("category =", document["metadata"]["category"])
    print("source =", document["metadata"]["source"])
    print("document_type =", document["metadata"]["document_type"])
    print("chars =", len(document["page_content"]))
    print(document["page_content"][:500])