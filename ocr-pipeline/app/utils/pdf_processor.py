try:
    import pymupdf  # Preferred alias
except Exception:  # pragma: no cover
    import fitz as pymupdf  # Fallback for environments exposing only 'fitz'
from PIL import Image
import numpy as np
import io
import os
from typing import List, Tuple, Optional
from app.config.settings import settings

def extract_pages_from_pdf(pdf_path: str) -> List[bytes]:
    """
    Extract images of each page from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of image bytes for each page
    """
    images = []

    try:
        doc = pymupdf.open(pdf_path)
        try:
            for page in doc:
                pix = page.get_pixmap(dpi=settings.PDF_DPI)
                images.append(pix.tobytes("jpeg"))
        finally:
            doc.close()
    except Exception as e:
        print(f"Error extracting pages from PDF {pdf_path}: {str(e)}")
        raise

    return images

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text directly from PDF if available.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content
    """
    text_content = ""

    try:
        doc = pymupdf.open(pdf_path)
        try:
            for page in doc:
                text_content += page.get_text("text")
        finally:
            doc.close()
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {str(e)}")

    return text_content

def get_filename_from_path(file_path: str) -> str:
    """Extract filename without extension from file path."""
    return os.path.splitext(os.path.basename(file_path))[0]

def create_output_filename(input_path: str) -> str:
    """Create output filename with extracted tag."""
    filename = get_filename_from_path(input_path)
    return f"{filename}{settings.EXTRACTED_TAG}.pdf"

def split_page_into_columns(image_bytes: bytes, aggressive: bool = False) -> Tuple[Optional[bytes], Optional[bytes], bool]:
    """
    Split a page image into left and right columns if it appears to have two columns.
    Uses a simple heuristic: if the page is portrait and wider than it is tall, 
    or if there's a visible vertical gap in the middle, split it.
    
    Args:
        image_bytes: JPEG image bytes of the page
        
    Returns:
        Tuple of (left_column_bytes, right_column_bytes, is_split)
        - If split: returns (left_bytes, right_bytes, True)
        - If single column: returns (image_bytes, None, False)
        - For landscape pages: returns (image_bytes, None, False)
    """
    try:
        # Load image from bytes
        img = Image.open(io.BytesIO(image_bytes)).convert('L')  # grayscale
        width, height = img.size

        # 1) Orientation guard: only consider portrait for splitting
        if height <= width:
            return image_bytes, None, False

        # 2) Compute arrays once
        arr = np.asarray(img, dtype=np.uint8)
        # Invert so text (dark) becomes high values after inversion
        ink = 255 - arr
        # Normalize edge margins to avoid margins skewing detection
        margin_y = max(4, height // 50)
        core = ink[margin_y:height - margin_y, :]
        vproj = core.sum(axis=0)

        # 2a) Prefer explicit vertical separator line detection (white or black line spanning most of page)
        # Build boolean masks for near-white and near-black pixels on original grayscale
        gray_core = arr[margin_y:height - margin_y, :]
        white_mask = (gray_core >= 245)
        black_mask = (gray_core <= 10)

        white_ratio_by_x = white_mask.mean(axis=0)
        black_ratio_by_x = black_mask.mean(axis=0)

        # A separator band is a contiguous region where either white or black ratio is very high across rows
        min_band_width = max(1, int(width * (0.003 if not aggressive else 0.001)))
        max_band_width = max(5, int(width * (0.10 if not aggressive else 0.20)))
        min_ratio = (0.90 if not aggressive else 0.70)  # at least X% of rows are pure line color

        def find_best_separator_index():
            candidates = []
            for ratio_by_x, tone in ((white_ratio_by_x, 'white'), (black_ratio_by_x, 'black')):
                x = 0
                while x < width:
                    if ratio_by_x[x] >= min_ratio:
                        start = x
                        while x < width and ratio_by_x[x] >= min_ratio:
                            x += 1
                        end = x  # exclusive
                        band_w = end - start
                        if min_band_width <= band_w <= max_band_width:
                            mid = (start + end) // 2
                            # score by closeness to center
                            center_dist = abs(mid - (width // 2))
                            candidates.append((center_dist, mid, band_w, tone))
                    else:
                        x += 1
            if not candidates:
                return None
            candidates.sort(key=lambda t: t[0])
            return candidates[0][1]

        separator_x = find_best_separator_index()
        if separator_x is not None:
            # Ensure both sides have meaningful content before splitting
            left_region = vproj[:separator_x]
            right_region = vproj[separator_x:]
            plateau = max(float(np.max(vproj)), 1.0)
            left_content = float(np.mean(left_region)) if left_region.size else 0.0
            right_content = float(np.mean(right_region)) if right_region.size else 0.0
            content_frac = settings.SPLIT_BOTH_SIDES_CONTENT_FRAC * (0.5 if aggressive else 1.0)
            both_sides_content = (left_content >= plateau * content_frac) and (right_content >= plateau * content_frac)
            if both_sides_content:
                color_img = Image.open(io.BytesIO(image_bytes))  # original color
                left_img = color_img.crop((0, 0, separator_x, height))
                right_img = color_img.crop((separator_x, 0, width, height))
                if left_img.size[0] >= int(width * settings.SPLIT_MIN_SIDE_FRAC) and right_img.size[0] >= int(width * settings.SPLIT_MIN_SIDE_FRAC):
                    left_buffer = io.BytesIO()
                    left_img.save(left_buffer, format='JPEG', quality=95)
                    right_buffer = io.BytesIO()
                    right_img.save(right_buffer, format='JPEG', quality=95)
                    return left_buffer.getvalue(), right_buffer.getvalue(), True

        # 3) Look for a central valley (gap) near the middle indicating two columns
        mid = width // 2
        window = max(20, width // 12)  # central window ±window
        left_bound = max(0, mid - window)
        right_bound = min(width - 1, mid + window)
        central_slice = vproj[left_bound:right_bound]
        if central_slice.size == 0:
            return image_bytes, None, False

        # Valley detection: find index of minimum ink in central zone
        local_min_idx = int(np.argmin(central_slice)) + left_bound
        valley_value = float(vproj[local_min_idx])

        # Compare valley depth to neighboring plateaus (left/right maxima)
        left_plateau = float(np.max(vproj[max(0, left_bound - window):left_bound])) if left_bound > 0 else float(np.max(vproj[:left_bound + 1]))
        right_plateau = float(np.max(vproj[right_bound:min(width, right_bound + window)])) if right_bound < width - 1 else float(np.max(vproj[right_bound - 1:]))
        plateau = max(left_plateau, right_plateau, 1.0)

        valley_ratio = valley_value / plateau  # lower is better gap

        # 4) Determine continuous low-ink region around the valley (gap width)
        threshold = plateau * 0.20  # 20% of plateau considered "gap"
        # expand left
        l = local_min_idx
        while l > 0 and vproj[l] <= threshold:
            l -= 1
        # expand right
        r = local_min_idx
        while r < width - 1 and vproj[r] <= threshold:
            r += 1
        gap_width = r - l

        # 5) Ensure both halves have sufficient content (avoid splitting tables/single-column)
        left_region = vproj[:local_min_idx]
        right_region = vproj[local_min_idx:]
        left_content = float(np.mean(left_region)) if left_region.size else 0.0
        right_content = float(np.mean(right_region)) if right_region.size else 0.0

        # Horizontal projection: if strong grid-like content (tables), avoid splitting
        hproj = core.sum(axis=1)
        # Count horizontal peaks (rows) by thresholding
        h_thresh = float(np.percentile(hproj, 80))
        horizontal_bands = int(np.sum(hproj > h_thresh))

        # Heuristic rules (configurable via settings):
        # - valley must be significant (ratio <= settings.SPLIT_VALLEY_RATIO)
        # - gap must be reasonably wide (>= settings.SPLIT_GAP_MIN_FRAC of width)
        # - both sides must have meaningful content (>= settings.SPLIT_BOTH_SIDES_CONTENT_FRAC of plateau)
        # - too many horizontal bands imply table-like structure; avoid splitting
        gap_wide_enough = gap_width >= max(10, int(width * settings.SPLIT_GAP_MIN_FRAC))
        both_sides_content = (left_content >= plateau * settings.SPLIT_BOTH_SIDES_CONTENT_FRAC) and (right_content >= plateau * settings.SPLIT_BOTH_SIDES_CONTENT_FRAC)
        valley_significant = valley_ratio <= settings.SPLIT_VALLEY_RATIO
        table_like = horizontal_bands >= max(settings.SPLIT_TABLE_BANDS_MIN, height // 20)

        should_split = valley_significant and gap_wide_enough and both_sides_content and not table_like
        if not should_split:
            # If aggressive mode is on (LLM said TWO_COLUMN), force a safe split at the lowest-ink column near center
            if aggressive:
                split_x = max(1, min(width - 2, local_min_idx))
                color_img = Image.open(io.BytesIO(image_bytes))  # original color
                left_img = color_img.crop((0, 0, split_x, height))
                right_img = color_img.crop((split_x, 0, width, height))
                if left_img.size[0] >= int(width * settings.SPLIT_MIN_SIDE_FRAC) and right_img.size[0] >= int(width * settings.SPLIT_MIN_SIDE_FRAC):
                    left_buffer = io.BytesIO()
                    left_img.save(left_buffer, format='JPEG', quality=95)
                    right_buffer = io.BytesIO()
                    right_img.save(right_buffer, format='JPEG', quality=95)
                    return left_buffer.getvalue(), right_buffer.getvalue(), True
            return image_bytes, None, False

        # 6) Perform split at valley position (local_min_idx)
        split_x = max(1, min(width - 2, local_min_idx))
        color_img = Image.open(io.BytesIO(image_bytes))  # original color
        left_img = color_img.crop((0, 0, split_x, height))
        right_img = color_img.crop((split_x, 0, width, height))

        # Ensure both halves are not too narrow (>= 30% width)
        if left_img.size[0] < int(width * settings.SPLIT_MIN_SIDE_FRAC) or right_img.size[0] < int(width * settings.SPLIT_MIN_SIDE_FRAC):
            return image_bytes, None, False

        # Convert back to bytes
        left_buffer = io.BytesIO()
        left_img.save(left_buffer, format='JPEG', quality=95)
        right_buffer = io.BytesIO()
        right_img.save(right_buffer, format='JPEG', quality=95)

        return left_buffer.getvalue(), right_buffer.getvalue(), True

    except Exception as e:
        # If splitting fails, return original image as single page
        print(f"Error splitting page into columns: {str(e)}")
        return image_bytes, None, False
