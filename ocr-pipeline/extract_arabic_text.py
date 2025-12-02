#!/usr/bin/env python3
"""
Simple Arabic Text Extraction Script

This script automatically processes all PDF files in the input_pdfs folder
and saves the extracted Arabic text as DOCX files to the output_docx folder.
"""

import os
import sys
import logging
import time
from arabic_text_extraction_pipeline import extract_arabic_text_from_pdfs_sync

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main function to process all PDFs in input_pdfs folder."""
    
    # Define directories relative to this script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input_pdfs")
    output_dir = os.path.join(base_dir, "output_docx")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory '{input_dir}' not found.")
        print("Please create the 'input_pdfs' folder and add your PDF files there.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files from input directory
    pdf_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(input_dir, file))
    
    if not pdf_files:
        print(f"ERROR: No PDF files found in '{input_dir}'.")
        print("Please add PDF files to the input_pdfs folder.")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process:")
    for pdf_file in pdf_files:
        print(f"   - {os.path.basename(pdf_file)}")
    
    print(f"\nStarting Arabic text extraction...")
    print(f"Input folder: {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Output format: DOCX files")
    
    # Start timing the entire process
    process_start_time = time.time()
    
    try:
        # Run the extraction pipeline
        results = extract_arabic_text_from_pdfs_sync(pdf_files, output_dir)
        
        # Calculate total process time
        total_process_time = time.time() - process_start_time
        
        # Display results
        summary = results['summary']
        print(f"\nExtraction completed!")
        print(f"Total processing time: {total_process_time:.2f} seconds")
        print(f"Summary:")
        print(f"   - Total PDFs processed: {summary['total_pdfs']}")
        print(f"   - Successful extractions: {summary['successful_pdfs']}")
        print(f"   - Failed extractions: {summary['failed_pdfs']}")
        print(f"   - Total pages processed: {summary['total_pages_processed']}")
        print(f"   - Successful text extractions: {summary['total_successful_extractions']}")
        print(f"   - Failed text extractions: {summary['total_failed_extractions']}")
        
        # Display individual results with timing
        print(f"\nIndividual Results:")
        total_extraction_time = 0
        total_docx_time = 0
        for pdf_path, result in results['results'].items():
            filename = os.path.basename(pdf_path)
            if 'error' in result:
                print(f"   ERROR: {filename}: {result['error']}")
            else:
                output_filename = os.path.basename(result['output_docx'])
                timing = result.get('timing', {})
                total_extraction_time += timing.get('extraction_time_seconds', 0)
                total_docx_time += timing.get('docx_creation_time_seconds', 0)
                print(f"   SUCCESS: {filename} -> {output_filename}")
                print(f"      Pages: {result['successful_extractions']}/{result['total_pages']} extracted")
                print(f"      Time: {timing.get('total_time_seconds', 0):.2f}s total ({timing.get('extraction_time_seconds', 0):.2f}s extraction, {timing.get('docx_creation_time_seconds', 0):.2f}s DOCX)")
                print(f"      Avg per page: {timing.get('average_time_per_page_seconds', 0):.2f}s")
        
        # Display overall timing summary
        print(f"\nTiming Summary:")
        print(f"   - Total process time: {total_process_time:.2f} seconds")
        print(f"   - Total extraction time: {total_extraction_time:.2f} seconds")
        print(f"   - Total DOCX creation time: {total_docx_time:.2f} seconds")
        if summary['total_pages_processed'] > 0:
            print(f"   - Average time per page: {total_extraction_time / summary['total_pages_processed']:.2f} seconds")
        
        print(f"\nExtracted DOCX files saved to: {output_dir}")
        
    except Exception as e:
        print(f"ERROR: Error during extraction: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
