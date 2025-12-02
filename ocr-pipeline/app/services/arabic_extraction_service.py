import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple
from app.utils.pdf_processor import extract_pages_from_pdf
from app.services.genai_client import process_image_async
from app.utils.docx_creator import create_docx_from_extracted_pages, create_output_docx_filename
from app.config.settings import settings
from app.utils.text_validators import validate_extracted_text

def _extract_json_from_response(response: str) -> dict:
    """
    Extract JSON from LLM response that may contain extra text before/after the JSON.
    """
    import re
    
    response = response.strip()
    if not response:
        raise ValueError("Empty response")
    
    # First try direct parsing
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in the response
    # Look for the first { and balance braces to find the JSON
    start_idx = response.find('{')
    if start_idx == -1:
        raise ValueError("No JSON object found in response")
    
    brace_count = 0
    end_idx = start_idx
    for i in range(start_idx, len(response)):
        if response[i] == '{':
            brace_count += 1
        elif response[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
    
    if brace_count != 0:
        raise ValueError("Unbalanced braces in JSON")
    
    json_str = response[start_idx:end_idx + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON extracted: {e}")

logger = logging.getLogger(__name__)

async def extract_arabic_text_from_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract Arabic text from PDF pages using Gemini 2.5 Flash with LLM-driven column detection.
    
    The LLM analyzes each page and intelligently detects columns WITHOUT heuristic splitting.
    For two-column pages, the LLM extracts both columns from the full page image.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dictionaries containing extracted text for each page
    """
    extracted_pages: List[Dict[str, Any]] = []
    image_data_map: Dict[Tuple[int, str | None], bytes] = {}

    try:
        # Extract pages as images
        page_images = extract_pages_from_pdf(pdf_path)
        logger.info(f"Extracted {len(page_images)} pages from PDF: {pdf_path}")

        # Process each page using LLM-driven column detection (no heuristic splitting)
        tasks = []
        page_tasks_map = {}  # Map task index to page_num
        
        for page_num, image_bytes in enumerate(page_images, 1):
            # Use LLM to detect layout AND extract columns in one call
            task = process_column_aware_extraction(image_bytes, page_num)
            tasks.append(task)
            page_tasks_map[len(tasks) - 1] = page_num
            image_data_map[(page_num, None)] = image_bytes

        # Wait for all pages to be processed
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results - LLM returns regions with layout info
        for task_idx, result in enumerate(results):
            page_num = page_tasks_map.get(task_idx, 0)
            
            if isinstance(result, Exception):
                logger.error(f"Error processing page {page_num} of {pdf_path}: {str(result)}")
                extracted_pages.append({
                    'page_number': page_num,
                    'column_type': None,
                    'layout': 'SINGLE_COLUMN',
                    'text': '',
                    'structured_format': 'plain',
                    'structured_table': None,
                    'notes': '',
                    'validation_issues': [f"exception: {str(result)}"],
                    'retry_attempted': False,
                    'error': str(result)
                })
            else:
                # Result is from column-aware extraction with regions
                if isinstance(result, dict):
                    layout = result.get('layout', 'SINGLE_COLUMN')
                    regions = result.get('regions', [])
                    extraction_error = result.get('error')
                    
                    # Process regions (RIGHT, LEFT, or FULL)
                    for region in regions:
                        side = region.get('side', 'FULL')
                        text_payload = (region.get('text') or '').strip()
                        
                        # Map side to column_type
                        column_type = None
                        if side == 'RIGHT':
                            column_type = 'right'
                        elif side == 'LEFT':
                            column_type = 'left'
                        
                        validation_source = _compose_validation_source(text_payload, None, '')
                        validation_issues = validate_extracted_text(validation_source) if validation_source else []
                        if extraction_error:
                            validation_issues.append(f"llm_error: {extraction_error}")
                        
                        extracted_pages.append({
                            'page_number': page_num,
                            'column_type': column_type,
                            'layout': layout,
                            'text': text_payload,
                            'structured_format': 'plain',
                            'structured_table': None,
                            'notes': '',
                            'validation_issues': validation_issues,
                            'retry_attempted': False,
                            'error': extraction_error
                        })
                else:
                    # Fallback for unexpected result format
                    extraction_error = None
                    text_payload = str(result).strip()
                    
                    extracted_pages.append({
                        'page_number': page_num,
                        'column_type': None,
                        'layout': 'SINGLE_COLUMN',
                        'text': text_payload,
                        'structured_format': 'plain',
                        'structured_table': None,
                        'notes': '',
                        'validation_issues': [],
                        'retry_attempted': False,
                        'error': extraction_error
                    })

        # Retry duplicate pages with stricter prompt if necessary
        extracted_pages = await reconcile_duplicate_content(extracted_pages, image_data_map)

    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        # Return empty results for all pages if PDF processing fails
        page_images = extract_pages_from_pdf(pdf_path)
        for page_num in range(1, len(page_images) + 1):
            extracted_pages.append({
                'page_number': page_num,
                'text': '',
                'structured_format': 'plain',
                'structured_table': None,
                'notes': '',
                'validation_issues': [f"exception: {str(e)}"],
                'retry_attempted': False,
                'error': str(e)
            })

    return extracted_pages


async def _call_llm_with_retry(
    image_bytes: bytes,
    prompt: str,
    context_label: str,
) -> str:
    attempts = max(settings.LLM_MAX_RETRIES, 1)
    base_delay = max(settings.LLM_RETRY_DELAY_SECONDS, 0.0)
    backoff = max(settings.LLM_RETRY_BACKOFF, 1.0)

    last_exception: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = await process_image_async(image_bytes, prompt)
            if response and response.strip():
                return response
            raise ValueError("empty response from LLM")
        except Exception as exc:  # noqa: PERF203
            last_exception = exc
            if attempt >= attempts:
                logger.error(
                    "LLM call failed after %s attempts for %s: %s",
                    attempts,
                    context_label,
                    exc
                )
                break

            sleep_for = base_delay * (backoff ** (attempt - 1))
            logger.warning(
                "LLM call attempt %s/%s failed for %s: %s. Retrying in %.1fs",
                attempt,
                attempts,
                context_label,
                exc,
                sleep_for
            )
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    raise last_exception or RuntimeError(f"LLM call failed for {context_label}")


async def process_column_aware_extraction(image_bytes: bytes, page_number: int) -> Dict[str, Any]:
    """
    Process a page using LLM-driven column detection and extraction.
    
    The LLM analyzes the full page image and:
    1. Detects if the page has one column, two columns, or is a table
    2. Extracts text from each region without physical image splitting
    
    Returns a dictionary with layout info and regions containing extracted text.
    """
    try:
        # Call LLM with column-aware extraction prompt
        raw_response = await _call_llm_with_retry(
            image_bytes,
            settings.COLUMN_AWARE_EXTRACTION_PROMPT,
            context_label=f"column-aware extraction for page {page_number}"
        )
        
        # Parse JSON response
        parsed = _extract_json_from_response(raw_response)
        
        layout = parsed.get('layout', 'SINGLE_COLUMN')
        regions = parsed.get('regions', [])
        
        logger.info(f"Page {page_number} column-aware extraction: layout={layout}, regions={len(regions)}")
        
        return {
            'page_number': page_number,
            'layout': layout,
            'regions': regions,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Error in column-aware extraction for page {page_number}: {str(e)}")
        # Fallback: return empty regions with error
        return {
            'page_number': page_number,
            'layout': 'SINGLE_COLUMN',
            'regions': [],
            'error': str(e)
        }


async def process_arabic_text_extraction(
    image_bytes: bytes,
    page_number: int,
    column_type: str | None = None,
    layout: str | None = None,
    prompt_override: str | None = None
) -> Dict[str, Any]:
    """
    Process a single page image (or column) to extract Arabic text.

    Args:
        image_bytes: Image bytes of the page or column
        page_number: Page number for logging
        column_type: 'left', 'right', or None for single column
        layout: Layout classification for the page
        prompt_override: Optional override prompt for retries

    Returns:
        Dictionary containing extracted Arabic text and metadata
    """
    column_info = f" ({column_type})" if column_type else ""
    layout_info = f" [{layout}]" if layout else ""

    try:
        prompt = prompt_override or settings.ARABIC_EXTRACTION_PROMPT
        
        # Add instruction for column-specific extraction if applicable
        if column_type:
            column_prompt = (
                f"{prompt}\n\nIMPORTANT: This image contains only the {column_type.upper()} column "
                f"of page {page_number}. Extract ALL text from this {column_type} column only. "
                "Do not miss any details."
            )
        else:
            column_prompt = prompt
            
        extracted_text = await _call_llm_with_retry(
            image_bytes,
            column_prompt,
            context_label=f"page {page_number}{column_info}{layout_info}"
        )

        logger.info(f"Successfully extracted text from page {page_number}{column_info}{layout_info}")
        return {
            'text': (extracted_text or "").strip(),
            'format': 'plain',
            'structured_table': None,
            'notes': '',
            'error': None
        }

    except Exception as e:
        logger.error(f"Error extracting text from page {page_number}{column_info}{layout_info}: {str(e)}")
        return {
            'text': '',
            'format': 'plain',
            'structured_table': None,
            'notes': '',
            'error': str(e)
        }


async def process_table_text_extraction(image_bytes: bytes, page_number: int) -> Dict[str, Any]:
    """
    Process a table page image and return structured table data.
    """
    try:
        raw_response = await _call_llm_with_retry(
            image_bytes,
            settings.TABLE_EXTRACTION_PROMPT,
            context_label=f"table page {page_number}"
        )
        parsed = _extract_json_from_response(raw_response)
        headers = parsed.get('headers') or []
        rows = parsed.get('rows') or []
        notes = parsed.get('notes') or ''

        flattened_text = _flatten_table_to_text(headers, rows, notes)
        logger.info(f"Successfully extracted table from page {page_number}")
        return {
            'text': flattened_text,
            'format': 'table',
            'structured_table': {
                'headers': headers,
                'rows': rows,
                'notes': notes
            },
            'notes': notes,
            'error': None
        }
    except Exception as e:
        logger.warning(f"JSON parsing failed for table page {page_number}, falling back to plain text extraction: {str(e)}")
        # Fall back to plain text extraction for table pages that fail JSON parsing
        try:
            fallback_result = await process_arabic_text_extraction(
                image_bytes, 
                page_number, 
                None, 
                layout="TABLE"
            )
            logger.info(f"Fallback plain text extraction successful for table page {page_number}")
            return fallback_result
        except Exception as fallback_e:
            logger.error(f"Fallback extraction also failed for table page {page_number}: {str(fallback_e)}")
            return {
                'text': '',
                'format': 'plain',
                'structured_table': None,
                'notes': '',
                'error': f"Table extraction failed: {str(e)}; Fallback failed: {str(fallback_e)}"
            }

async def classify_page_layout(image_bytes: bytes) -> str:
    """
    Classify the page layout using the LLM to cross-check heuristic splits.

    Returns one of: SINGLE_COLUMN, TWO_COLUMN, TABLE, LANDSCAPE
    """
    try:
        label = await _call_llm_with_retry(
            image_bytes,
            settings.LAYOUT_CLASSIFICATION_PROMPT,
            context_label="layout classification"
        )
        label = (label or "").strip().upper()
        if "TWO_COLUMN" in label:
            return "TWO_COLUMN"
        if "TABLE" in label:
            return "TABLE"
        if "LANDSCAPE" in label:
            return "LANDSCAPE"
        return "SINGLE_COLUMN"
    except Exception as e:
        logger.warning(f"Layout classification failed, defaulting to SINGLE_COLUMN: {str(e)}")
        return "SINGLE_COLUMN"


async def reconcile_duplicate_content(
    extracted_pages: List[Dict[str, Any]],
    image_data_map: Dict[Tuple[int, str | None], bytes],
) -> List[Dict[str, Any]]:
    """
    Detect and retry pages with identical content that likely indicate OCR loops.
    """
    if not extracted_pages:
        return extracted_pages

    hash_map: Dict[str, List[int]] = {}
    for idx, page in enumerate(extracted_pages):
        text = (page.get('text') or '').strip()
        if not text:
            continue
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        hash_map.setdefault(digest, []).append(idx)

    for indices in hash_map.values():
        if len(indices) <= 1:
            continue

        for page_index in indices:
            page_entry = extracted_pages[page_index]
            if page_entry.get('layout') == 'TABLE':
                page_entry.setdefault('validation_issues', []).append('duplicate_table_content_detected')
                continue

            if page_entry.get('retry_attempted'):
                page_entry.setdefault('validation_issues', []).append('duplicate_content_after_retry')
                continue

            key = (page_entry.get('page_number'), page_entry.get('column_type'))
            image_bytes = image_data_map.get(key)
            if not image_bytes:
                page_entry.setdefault('validation_issues', []).append('duplicate_content_missing_image_bytes')
                continue

            retry_payload = await process_arabic_text_extraction(
                image_bytes,
                page_entry.get('page_number', 0),
                column_type=page_entry.get('column_type'),
                layout=page_entry.get('layout'),
                prompt_override=settings.ARABIC_EXTRACTION_PROMPT_STRICT,
            )

            page_entry['retry_attempted'] = True
            if retry_payload.get('text'):
                page_entry['text'] = retry_payload.get('text', '').strip()
                page_entry['structured_format'] = retry_payload.get('format', 'plain')
                page_entry['structured_table'] = retry_payload.get('structured_table')
                page_entry['notes'] = retry_payload.get('notes', '')
                page_entry['error'] = retry_payload.get('error')
                validation_source = _compose_validation_source(
                    page_entry['text'],
                    page_entry.get('structured_table'),
                    page_entry.get('notes', '')
                )
                page_entry['validation_issues'] = validate_extracted_text(validation_source) if validation_source else []
            else:
                page_entry.setdefault('validation_issues', []).append('duplicate_content_retry_failed')

    _mark_remaining_duplicates(extracted_pages)
    return extracted_pages


def _flatten_table_to_text(headers: List[str], rows: List[List[str]], notes: str) -> str:
    """
    Create a plain-text representation of table data to keep compatibility downstream.
    """
    lines: List[str] = []
    cleaned_headers = [header.strip() for header in headers if header is not None]
    if cleaned_headers:
        lines.append("\t".join(cleaned_headers))

    for row in rows:
        safe_row = [(cell or "").strip() for cell in row]
        lines.append("\t".join(safe_row))

    notes = (notes or "").strip()
    if notes:
        lines.append(notes)

    return "\n".join(line for line in lines if line).strip()


def _compose_validation_source(text: str, structured_table: Dict[str, Any] | None, notes: str) -> str:
    """
    Build a canonical string that validators can inspect.
    """
    segments: List[str] = []
    if structured_table:
        table_text = _flatten_table_to_text(
            structured_table.get('headers') or [],
            structured_table.get('rows') or [],
            structured_table.get('notes') or ''
        )
        if table_text:
            segments.append(table_text)
    elif text:
        segments.append(text)

    notes = (notes or "").strip()
    if notes:
        segments.append(notes)

    return "\n\n".join(segments).strip()


def _mark_remaining_duplicates(extracted_pages: List[Dict[str, Any]]) -> None:
    """
    Add validation warnings for any content that still duplicates after retries.
    """
    hash_map: Dict[str, List[int]] = {}
    for idx, page in enumerate(extracted_pages):
        text = (page.get('text') or '').strip()
        if not text:
            continue
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        hash_map.setdefault(digest, []).append(idx)

    for indices in hash_map.values():
        if len(indices) <= 1:
            continue
        for page_index in indices:
            issues = extracted_pages[page_index].setdefault('validation_issues', [])
            if 'duplicate_content_detected' not in issues:
                issues.append('duplicate_content_detected')
def save_extracted_text_as_docx(extracted_pages: List[Dict[str, Any]], input_pdf_path: str, output_dir: str, timing_info: Dict[str, float] = None) -> str:
    """
    Save extracted text as a DOCX file.

    Args:
        extracted_pages: List of extracted page data
        input_pdf_path: Original PDF path
        output_dir: Directory to save the output DOCX
        timing_info: Optional timing information to include in the document

    Returns:
        Path to the created DOCX file
    """
    # Create output filename
    input_filename = os.path.basename(input_pdf_path)
    output_filename = create_output_docx_filename(input_filename)
    output_path = os.path.join(output_dir, output_filename)

    # Create DOCX from extracted text
    create_docx_from_extracted_pages(extracted_pages, input_pdf_path, output_path, timing_info)

    logger.info(f"Saved extracted text to: {output_path}")
    return output_path

async def process_pdf_for_arabic_extraction(pdf_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Main function to process a PDF for Arabic text extraction.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the output

    Returns:
        Dictionary with processing results
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting Arabic text extraction for: {pdf_path}")

        # Extract text from each page
        extraction_start = time.time()
        extracted_pages = await extract_arabic_text_from_pdf_pages(pdf_path)
        extraction_time = time.time() - extraction_start

        # Save extracted text as DOCX
        docx_start = time.time()
        output_path = save_extracted_text_as_docx(extracted_pages, pdf_path, output_dir)
        docx_time = time.time() - docx_start

        # Count successful extractions
        successful_pages = sum(1 for page in extracted_pages if page.get('error') is None and page.get('text', '').strip())
        
        total_time = time.time() - start_time

        return {
            'input_pdf': pdf_path,
            'output_docx': output_path,
            'total_pages': len(extracted_pages),
            'successful_extractions': successful_pages,
            'failed_pages': len(extracted_pages) - successful_pages,
            'extracted_pages': extracted_pages,
            'timing': {
                'total_time_seconds': round(total_time, 2),
                'extraction_time_seconds': round(extraction_time, 2),
                'docx_creation_time_seconds': round(docx_time, 2),
                'average_time_per_page_seconds': round(extraction_time / len(extracted_pages), 2) if extracted_pages else 0
            }
        }

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        return {
            'input_pdf': pdf_path,
            'error': str(e),
            'total_pages': 0,
            'successful_extractions': 0,
            'failed_pages': 0,
            'extracted_pages': [],
            'timing': {
                'total_time_seconds': round(total_time, 2),
                'extraction_time_seconds': 0,
                'docx_creation_time_seconds': 0,
                'average_time_per_page_seconds': 0
            }
        }
