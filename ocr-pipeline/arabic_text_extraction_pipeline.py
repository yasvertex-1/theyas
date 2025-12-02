import asyncio
import os
import logging
from typing import List, Dict, Any, Optional
from app.services.arabic_extraction_service import process_pdf_for_arabic_extraction
from app.config.settings import settings

logger = logging.getLogger(__name__)

class ArabicTextExtractionPipeline:
    """Main pipeline for extracting Arabic text from PDF documents."""

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the pipeline.

        Args:
            output_dir: Directory to save extracted PDFs. Defaults to settings.OUTPUT_DIR
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Arabic text extraction pipeline initialized. Output directory: {self.output_dir}")

    async def process_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF file for Arabic text extraction.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Processing results dictionary
        """
        if not os.path.exists(pdf_path):
            error_msg = f"PDF file not found: {pdf_path}"
            logger.error(error_msg)
            return {
                'input_pdf': pdf_path,
                'error': error_msg,
                'total_pages': 0,
                'successful_extractions': 0,
                'failed_pages': 0
            }

        logger.info(f"Processing PDF: {pdf_path}")
        result = await process_pdf_for_arabic_extraction(pdf_path, self.output_dir)

        if 'error' not in result:
            logger.info(f"Successfully processed PDF: {pdf_path}")
            output_path = result.get('output_docx') or result.get('output_pdf')
            if output_path:
                logger.info(f"Output saved to: {output_path}")
        else:
            logger.error(f"Failed to process PDF: {pdf_path} - {result['error']}")

        return result

    async def process_multiple_pdfs(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple PDF files for Arabic text extraction.

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            Dictionary with processing results for all PDFs
        """
        logger.info(f"Processing {len(pdf_paths)} PDF files")

        # Limit document-level concurrency using a semaphore
        semaphore = asyncio.Semaphore(settings.DOC_CONCURRENCY)

        async def _bounded_process(path: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.process_single_pdf(path)

        tasks = [asyncio.create_task(_bounded_process(pdf_path)) for pdf_path in pdf_paths]

        # Keep a steady concurrency and collect results as tasks finish
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        processed_results = {}
        summary = {
            'total_pdfs': len(pdf_paths),
            'successful_pdfs': 0,
            'failed_pdfs': 0,
            'total_pages_processed': 0,
            'total_successful_extractions': 0,
            'total_failed_extractions': 0
        }

        for i, result in enumerate(results):
            pdf_path = pdf_paths[i]

            if isinstance(result, Exception):
                logger.error(f"Exception processing {pdf_path}: {str(result)}")
                processed_results[pdf_path] = {
                    'error': str(result),
                    'total_pages': 0,
                    'successful_extractions': 0,
                    'failed_pages': 0
                }
                summary['failed_pdfs'] += 1
            else:
                processed_results[pdf_path] = result

                if 'error' not in result:
                    summary['successful_pdfs'] += 1
                    summary['total_pages_processed'] += result.get('total_pages', 0)
                    summary['total_successful_extractions'] += result.get('successful_extractions', 0)
                    summary['total_failed_extractions'] += result.get('failed_pages', 0)
                else:
                    summary['failed_pdfs'] += 1

        logger.info(f"Pipeline completed. Summary: {summary}")

        return {
            'results': processed_results,
            'summary': summary
        }

    def process_pdfs_sync(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """
        Synchronous wrapper for processing multiple PDFs.

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            Dictionary with processing results
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.process_multiple_pdfs(pdf_paths))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup resources
        from app.services.genai_client import shutdown_executor
        shutdown_executor()

# Convenience function for quick processing
async def extract_arabic_text_from_pdfs(pdf_paths: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract Arabic text from multiple PDF files.

    Args:
        pdf_paths: List of PDF file paths
        output_dir: Output directory for extracted PDFs

    Returns:
        Processing results
    """
    pipeline = ArabicTextExtractionPipeline(output_dir)
    return await pipeline.process_multiple_pdfs(pdf_paths)

def extract_arabic_text_from_pdfs_sync(pdf_paths: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronous version of extract_arabic_text_from_pdfs.

    Args:
        pdf_paths: List of PDF file paths
        output_dir: Output directory for extracted PDFs

    Returns:
        Processing results
    """
    pipeline = ArabicTextExtractionPipeline(output_dir)
    return pipeline.process_pdfs_sync(pdf_paths)
