import os
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document

class FileTextExtractor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.extension = os.path.splitext(file_path)[1].lower()

    def extract_text(self) -> str:
        with open(self.file_path, "rb") as f:
            content = f.read()

        if self.extension == ".pdf":
            return self._extract_pdf_text(content)
        elif self.extension == ".docx":
            return self._extract_docx_text(content)
        else:
            raise ValueError(f"Unsupported file type: {self.extension}")

    def _extract_pdf_text(self, content: bytes) -> str:
        try:
            with BytesIO(content) as pdf_stream:
                reader = PdfReader(pdf_stream)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise ValueError(f"PDF processing failed: {str(e)}")

    def _extract_docx_text(self, content: bytes) -> str:
        try:
            with BytesIO(content) as doc_stream:
                doc = Document(doc_stream)
                return "\n".join(para.text for para in doc.paragraphs if para.text)
        except Exception as e:
            raise ValueError(f"DOCX processing failed: {str(e)}")
