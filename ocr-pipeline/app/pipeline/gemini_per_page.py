import json
import re
from typing import Any, Dict
from app.config.settings import settings
from app.config.prompts import LAYOUT_CLASSIFICATION_PROMPT, COLUMN_AWARE_EXTRACTION_PROMPT
from app.services.genai_client import process_image_sync_with_retry


def _safe_json_parse(text: str) -> Dict[str, Any]:
    """
    Try to parse strict JSON. If the model wrapped it in prose accidentally,
    attempt to extract the first {...} block.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # extract first top-level JSON object
    match = re.search(r"\\{[\\s\\S]*\\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    raise ValueError("Failed to parse JSON from Gemini response")


def classify_layout(image_bytes: bytes) -> str:
    """
    Use Gemini to classify page layout.
    Returns one of: SINGLE_COLUMN, TWO_COLUMN, TABLE, LANDSCAPE
    """
    resp = process_image_sync_with_retry(image_bytes, settings.LAYOUT_CLASSIFICATION_PROMPT)
    token = resp.strip().upper()
    # Normalize unexpected wrappers (e.g., code fences)
    token = re.sub(r"[^A-Z_]", "", token)
    if token in {"SINGLE_COLUMN", "TWO_COLUMN", "TABLE", "LANDSCAPE"}:
        return token
    # Fallback to SINGLE_COLUMN if uncertain
    return "SINGLE_COLUMN"


def extract_columns_with_gemini(image_bytes: bytes, page_number: int, layout: str) -> Dict[str, Any]:
    """
    Ask Gemini to extract per-column text directly from the full page image (no cropping),
    labeling sides and including page_number.
    """
    header = f\"\"\"\nYou are given page_number = {page_number} and detected layout = {layout}.\nFollow the instructions precisely.\n\"\"\"\n
    prompt = header + COLUMN_AWARE_EXTRACTION_PROMPT
    resp = process_image_sync_with_retry(image_bytes, prompt)
    data = _safe_json_parse(resp)
    # Ensure required fields and normalize
    data.setdefault("page_number", page_number)
    data.setdefault("layout", layout)
    if "regions" not in data or not isinstance(data["regions"], list) or len(data["regions"]) == 0:
        # Construct a minimal fallback
        data["regions"] = [{"side": "FULL", "text": ""}]
    # Enforce side ordering for TWO_COLUMN: RIGHT then LEFT
    if layout == "TWO_COLUMN":
        sides = {"RIGHT": None, "LEFT": None}
        for r in data["regions"]:
            side = str(r.get("side", "")).upper()
            if side in sides and sides[side] is None:
                sides[side] = r
        ordered = [sides["RIGHT"], sides["LEFT"]]
        data["regions"] = [r for r in ordered if r is not None]
        # If missing one side, keep whatever exists
        if not data["regions"]:
            data["regions"] = [{"side": "RIGHT", "text": ""}, {"side": "LEFT", "text": ""}]
    else:
        # SINGLE_COLUMN/TABLE/LANDSCAPE => single FULL region
        if not any(str(r.get("side", "")).upper() == "FULL" for r in data["regions"]):
            # collapse into one FULL region concatenating text
            merged = "\\n\\n".join(str(r.get("text", "")) for r in data["regions"])
            data["regions"] = [{"side": "FULL", "text": merged}]
    return data


def process_page_with_gemini(image_bytes: bytes, page_number: int) -> Dict[str, Any]:
    """
    Public API: classify page, then extract per-column text using Gemini without image splitting.
    Returns a dict with keys: page_number, layout, regions: [{side, text}, ...]
    """
    layout = classify_layout(image_bytes)
    return extract_columns_with_gemini(image_bytes, page_number, layout)


