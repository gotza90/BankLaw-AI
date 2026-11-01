import fitz
import pytesseract
import io
from PIL import Image, ImageOps

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pdf = fitz.open(r"data/CBE Law No. 194 of 2020.pdf")
pages = [1, 10, 20]

for i in pages:
    print(f"\n===== PAGE {i + 1} =====")

    pix = pdf[i].get_pixmap(dpi=300, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    raw = pytesseract.image_to_string(
        img,
        lang="ara",
        config="--psm 3",
    )

    gray = pytesseract.image_to_string(
        ImageOps.grayscale(img),
        lang="ara",
        config="--psm 3",
    )

    print("RAW:")
    print(raw[:1800])

    print("\nGRAY:")
    print(gray[:1800])
