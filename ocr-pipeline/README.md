# Arabic Text Extraction Pipeline

This pipeline uses Google's Gemini 2.5 Flash model to extract Arabic text from PDF documents containing scanned images of Arabic text. The pipeline processes each page of the PDF individually to ensure accurate text extraction and saves the extracted text as a new PDF with the same name as the original but with "_extracted" suffix.

## Features

- **Page-by-page processing**: Extracts text from each page individually for better accuracy
- **Gemini 2.5 Flash integration**: Uses Google's latest vision model for high-quality Arabic text recognition
- **Batch processing**: Can process multiple PDF files concurrently
- **Error handling**: Robust error handling with retry mechanisms
- **PDF generation**: Creates new PDF files with extracted Arabic text
- **Configurable settings**: Easy to configure through environment variables

## Requirements

- Python 3.8+
- Google Generative AI API key
- Required packages (see requirements.txt)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Google API key:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

## Configuration

You can configure the pipeline using environment variables:

- `GOOGLE_API_KEY`: Your Google Generative AI API key (required)
- `GEMINI_MODEL`: Gemini model to use (default: "gemini-2.0-flash-exp")
- `PDF_DPI`: DPI for PDF page rendering (default: 300)
- `MAX_WORKERS`: Maximum number of worker threads (default: 4)
- `LLM_CONCURRENCY`: Maximum concurrent LLM requests (default: 2)
- `OUTPUT_DIR`: Directory to save extracted PDFs (default: "./output_pdfs")

## Usage

### Quick Start (Recommended)

1. **Add your PDF files** to the `input_pdfs/` folder
2. **Run the extraction script**:
   ```bash
   python extract_arabic_text.py
   ```
3. **Find your extracted PDFs** in the `output_pdfs/` folder

### Basic Usage

```python
from arabic_text_extraction_pipeline import extract_arabic_text_from_pdfs_sync

# Process a single PDF
results = extract_arabic_text_from_pdfs_sync(["path/to/arabic_document.pdf"])
```

### Advanced Usage

```python
import asyncio
from arabic_text_extraction_pipeline import ArabicTextExtractionPipeline

async def main():
    # Create pipeline instance
    pipeline = ArabicTextExtractionPipeline(output_dir="./my_output")

    # Process single PDF
    result = await pipeline.process_single_pdf("path/to/document.pdf")
    print(f"Processed: {result}")

    # Process multiple PDFs
    pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    results = await pipeline.process_multiple_pdfs(pdf_files)

    # Display summary
    summary = results['summary']
    print(f"Total PDFs: {summary['total_pdfs']}")
    print(f"Successful: {summary['successful_pdfs']}")
    print(f"Total pages processed: {summary['total_pages_processed']}")

# Run the pipeline
asyncio.run(main())
```

## Output

The pipeline creates new PDF files with the extracted Arabic text:

- Input: `document.pdf`
- Output: `document_extracted.pdf`

Each output PDF contains the extracted Arabic text from each page of the original document, preserving the page structure.

## Error Handling

The pipeline includes comprehensive error handling:

- **Network errors**: Automatic retry with exponential backoff
- **API rate limits**: Concurrency control to respect API limits
- **Invalid PDFs**: Graceful handling of corrupted or unsupported PDF files
- **Text extraction failures**: Continues processing other pages if one page fails

## Performance Considerations

- **Concurrent processing**: Multiple PDFs are processed simultaneously
- **Memory usage**: Large PDFs are processed page-by-page to manage memory
- **API quotas**: Respects Google's API rate limits through concurrency control

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Ensure your Google API key is set correctly
   - Check that your API key has access to Gemini models

2. **Memory Issues**:
   - For very large PDFs, consider reducing `PDF_DPI` or processing files individually

3. **Rate Limiting**:
   - If you hit rate limits, reduce `LLM_CONCURRENCY` or increase retry delays

4. **Text Quality**:
   - The quality of extracted text depends on the image quality of the PDF
   - Scanned documents with higher DPI generally produce better results

## License

This project is part of the legal-pipeline OCR system.
