import json
import os
from typing import List, Dict, Any
from app.config.settings import settings
from app.utils.pdf_processor import extract_pages_from_pdf, get_filename_from_path
from app.pipeline.gemini_per_page import process_page_with_gemini


def process_pdf_to_json(pdf_path: str) -> str:
    """
    Processes a PDF with Gemini-only OCR:
    - Renders each page to an image
    - Classifies layout with Gemini
    - Extracts text per column WITHOUT image splitting
    - Emits a JSON file with per-page regions labeled as RIGHT/LEFT/FULL

    Returns path to the JSON output file.
    """
    images: List[bytes] = extract_pages_from_pdf(pdf_path)
    results: List[Dict[str, Any]] = []
    for idx, img in enumerate(images, start=1):
        page_result = process_page_with_gemini(img, page_number=idx)
        results.append(page_result)

    base = get_filename_from_path(pdf_path)
    out_dir = os.path.join(settings.OUTPUT_DIR, "json")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}.gemini_ocr.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"document": base, "pages": results}, f, ensure_ascii=False, indent=2)
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m app.pipeline.run_pdf_to_json <path_to_pdf>")
        raise SystemExit(2)
    input_pdf = sys.argv[1]
    output = process_pdf_to_json(input_pdf)
    print(f"Wrote: {output}")


