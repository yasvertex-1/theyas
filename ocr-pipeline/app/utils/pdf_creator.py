from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import os
from typing import List, Dict
from app.config.settings import settings

def create_pdf_with_arabic_text(text_pages: List[str], output_path: str) -> None:
    """
    Create a PDF file with Arabic text content.

    Args:
        text_pages: List of text content for each page
        output_path: Path where to save the PDF
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()

    # Create custom style for Arabic text
    arabic_style = ParagraphStyle(
        'ArabicStyle',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        alignment=2,  # Right alignment for Arabic
        rightIndent=20,
        leftIndent=20,
    )

    # Create story (content) for the PDF
    story = []

    for i, page_text in enumerate(text_pages):
        # Add page text
        if page_text.strip():  # Only add non-empty text
            para = Paragraph(page_text.strip(), arabic_style)
            story.append(para)

        # Add page break except for the last page
        if i < len(text_pages) - 1:
            story.append(Spacer(1, 20))

    # Build PDF
    doc.build(story)

def create_pdf_from_extracted_pages(extracted_pages: List[Dict[str, str]], output_path: str) -> None:
    """
    Create a PDF from extracted text pages.

    Args:
        extracted_pages: List of dictionaries with 'page_number' and 'text' keys
        output_path: Path where to save the PDF
    """
    # Sort pages by page number
    sorted_pages = sorted(extracted_pages, key=lambda x: x.get('page_number', 0))

    # Extract text content
    text_pages = []
    for page in sorted_pages:
        text = page.get('text', '')
        text_pages.append(text)

    # Create PDF
    create_pdf_with_arabic_text(text_pages, output_path)
