import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

ARABIC_DIGIT_MAP: Dict[str, str] = {
    '٠': '0',
    '١': '1',
    '٢': '2',
    '٣': '3',
    '٤': '4',
    '٥': '5',
    '٦': '6',
    '٧': '7',
    '٨': '8',
    '٩': '9',
}

# Known high-risk tokens that frequently suffer from mis-recognition.
CRITICAL_TOKEN_MAPPINGS: Tuple[Dict[str, Iterable[str]], ...] = (
    {'expected': 'يعهد', 'variants': ('يمهد', 'يكلف', 'يمهد الى')},
    {'expected': 'أغسطس', 'variants': ('اغسطس',)},
)

CRITICAL_NAME_MAPPINGS: Tuple[Dict[str, Iterable[str]], ...] = (
    {'expected': 'بدر جاسم الدعيج', 'variants': ('بدر جاسم اليعقوب',)},
    {'expected': 'حبيب جوهر حياة', 'variants': ('حبيب جوهر حيات',)},
    {'expected': 'ضاري عبدالله العثمان', 'variants': ('شاري عبدالله العثمان', 'داري عبدالله العثمان')},
)


def validate_extracted_text(text_blob: str) -> List[str]:
    """
    Run heuristic validations to flag likely OCR issues requiring human review.
    """
    issues: List[str] = []
    if not text_blob:
        return issues

    issues.extend(_check_known_token_variants(text_blob))
    issues.extend(_check_name_variants(text_blob))
    issues.extend(_check_repetition(text_blob))
    issues.extend(_check_dates(text_blob))
    issues.extend(_check_numeric_ranges(text_blob))
    issues.extend(_check_arabic_digits(text_blob))
    issues.extend(_check_hallucination_indicators(text_blob))

    return issues


def _check_known_token_variants(text_blob: str) -> List[str]:
    issues: List[str] = []
    for mapping in CRITICAL_TOKEN_MAPPINGS:
        expected = mapping['expected']
        for variant in mapping['variants']:
            if variant in text_blob and expected not in text_blob:
                issues.append(f"suspect token '{variant}' detected; expected '{expected}'")
    return issues


def _check_name_variants(text_blob: str) -> List[str]:
    issues: List[str] = []
    for mapping in CRITICAL_NAME_MAPPINGS:
        expected = mapping['expected']
        for variant in mapping['variants']:
            if variant in text_blob and expected not in text_blob:
                issues.append(f"suspect name '{variant}' detected; expected '{expected}'")
    return issues


def _check_repetition(text_blob: str, threshold: int = 5) -> List[str]:
    issues: List[str] = []
    lines = [line.strip() for line in text_blob.splitlines() if line.strip()]
    if not lines:
        return issues

    counts = Counter(lines)
    for line, count in counts.items():
        if count > threshold:
            issues.append(f"repetition anomaly: '{line}' repeated {count} times")
    return issues


def _check_dates(text_blob: str) -> List[str]:
    issues: List[str] = []
    # Support both Arabic-Indic and Western digits
    date_pattern = re.compile(r'[\d٠-٩]{4}\s*/\s*[\d٠-٩]{1,2}\s*/\s*[\d٠-٩]{1,2}')

    for match in date_pattern.finditer(text_blob):
        parts = re.split(r'\s*/\s*', match.group())
        if len(parts) != 3:
            continue
        year, month, day = (_normalize_numeric_token(token) for token in parts)
        if not (1900 <= year <= 2100):
            issues.append(f"date anomaly: year {year} out of range in '{match.group()}'")
        if not (1 <= month <= 12):
            issues.append(f"date anomaly: month {month} invalid in '{match.group()}'")
        if not (1 <= day <= 31):
            issues.append(f"date anomaly: day {day} invalid in '{match.group()}'")

    return issues


def _check_numeric_ranges(text_blob: str) -> List[str]:
    issues: List[str] = []
    numeric_pattern = re.compile(r'[\d٠-٩][\d٠-٩,\.\s]*')
    numbers: List[int] = []
    for match in numeric_pattern.finditer(text_blob):
        normalised = _normalize_numeric_token(match.group())
        if normalised is not None:
            numbers.append(normalised)

    if len(numbers) < 2:
        return issues

    sorted_numbers = sorted(numbers)
    smallest = sorted_numbers[0]
    largest = sorted_numbers[-1]
    if smallest > 0 and largest >= smallest * 10:
        issues.append(f"numeric range anomaly: {smallest} vs {largest}")

    return issues


def _normalize_numeric_token(token: str) -> int | None:
    stripped = token.replace(',', '').replace(' ', '')
    translated = ''.join(ARABIC_DIGIT_MAP.get(ch, ch) for ch in stripped)
    if not translated or not translated.isdigit():
        return None
    try:
        return int(translated)
    except ValueError:
        return None


def _check_arabic_digits(text_blob: str) -> List[str]:
    issues: List[str] = []
    # Look for sequences of 3+ Arabic digits that might be OCR errors
    arabic_digit_pattern = re.compile(r'[٠-٩]{3,}')

    for match in arabic_digit_pattern.finditer(text_blob):
        digit_sequence = match.group()
        # Check for suspicious patterns: too many repeated digits, unrealistic sequences
        if _has_suspicious_digit_pattern(digit_sequence):
            issues.append(f"suspicious arabic digits: '{digit_sequence}' may be OCR error")

    return issues


def _has_suspicious_digit_pattern(digits: str) -> bool:
    """
    Check if a sequence of Arabic digits looks suspicious (likely OCR error).
    """
    # Convert to Western digits for easier analysis
    western = ''.join(ARABIC_DIGIT_MAP.get(ch, ch) for ch in digits)

    # Check for too many repeated digits (e.g., 000, 888)
    if len(set(western)) <= 2:  # Mostly the same digit repeated
        return True

    # Check for unrealistic large numbers (like 2040405)
    try:
        num = int(western)
        if num > 10000000:  # Unreasonably large for typical documents
            return True
    except ValueError:
        pass

    # Check for sequences that look like date fragments but malformed
    # (e.g., 0405, 1088 which are common OCR errors)
    if len(western) >= 4 and western.startswith(('0', '1')) and '88' in western:
        return True

    return False


def _check_hallucination_indicators(text_blob: str) -> List[str]:
    """
    Detect signs that the LLM hallucinated content instead of extracting from the image.
    """
    issues: List[str] = []
    
    # Check for excessive formatting/structure that suggests generated content
    # Real scanned documents typically have simpler structure
    lines = text_blob.splitlines()
    if not lines:
        return issues
    
    # Count lines with typical gazette/decree markers (signs of hallucination)
    gazette_markers = ['مرسوم', 'قرار', 'جريدة', 'رسمية', 'نشرة', 'إعلان', 'تعميم']
    marker_count = sum(1 for line in lines if any(marker in line for marker in gazette_markers))
    
    # If more than 30% of lines contain formal gazette markers, likely hallucination
    if len(lines) > 5 and marker_count / len(lines) > 0.3:
        issues.append("hallucination_risk: excessive formal gazette markers detected - content may be generated")
    
    # Check for suspiciously perfect formatting (multiple perfectly aligned columns)
    # Real OCR usually has alignment issues
    if len(lines) > 10:
        tab_lines = sum(1 for line in lines if '\t' in line or '  ' * 3 in line)
        if tab_lines / len(lines) > 0.7:
            issues.append("hallucination_risk: suspiciously perfect tabular formatting - content may be generated")
    
    # Check for excessive length relative to typical page content
    # A single page typically has 100-500 words, not thousands
    word_count = len(text_blob.split())
    if word_count > 2000:
        issues.append(f"hallucination_risk: excessive content ({word_count} words) for single page - may be multiple pages or generated")
    
    return issues
