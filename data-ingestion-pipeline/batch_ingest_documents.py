import os
import hashlib
import uuid
import datetime
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from src.file_reader import FileTextExtractor

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = "legal-ai-chatbot"
PINECONE_NAMESPACE = "legal-ai-docs"
SUPPORTED_FILE_TYPES = [".pdf", ".docx"]
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
MAX_CONCURRENT_PROCESSES = 20
DOCS_FOLDER = "data-ingestion-pipeline/docs-ingested"
LOG_FILE = "document_processing.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        if not PINECONE_API_KEY:
            raise RuntimeError("Missing PINECONE_API_KEY in environment variables")

        logger.info("Initializing embedding model...")
        self.embeddings_model = SentenceTransformer("BAAI/bge-m3")

        logger.info("Initializing Pinecone...")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        logger.info("Initialization complete")

    def generate_document_metadata(
        self, filename: str, content: bytes, document_id: str
    ) -> dict:
        return {
            "document_id": document_id,
            "source": filename,
            "file_type": Path(filename).suffix.lower(),
            "file_size": len(content),
            "ingestion_timestamp": datetime.datetime.now().isoformat(),
            "document_hash": hashlib.sha256(content).hexdigest(),
            "processing_id": str(uuid.uuid4()),
        }

    def process_single_document(self, file_path: Path) -> Dict[str, Any]:
        start_time = datetime.datetime.now()
        filename = file_path.name

        logger.info(f"Starting processing: {filename}")

        try:
            file_extension = file_path.suffix.lower()

            if file_extension not in SUPPORTED_FILE_TYPES:
                raise ValueError(f"Unsupported file type: {file_extension}")

            with open(file_path, "rb") as f:
                content = f.read()

            if not content:
                raise ValueError("Empty file")

            document_id = f"doc_{hashlib.sha256(content).hexdigest()[:16]}"
            metadata = self.generate_document_metadata(filename, content, document_id)

            extractor = FileTextExtractor(file_path=str(file_path))
            extracted_text = extractor.extract_text()

            if not extracted_text.strip():
                raise ValueError("No text content extracted from document")

            logger.info(f"Extracted {len(extracted_text)} characters from {filename}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True,
                length_function=len,
                is_separator_regex=False,
            )

            parent_doc = LangchainDocument(
                page_content=extracted_text, metadata=metadata
            )
            split_docs = text_splitter.split_documents([parent_doc])

            for i, doc in enumerate(split_docs):
                doc.metadata.update(
                    {
                        "chunk_id": f"{document_id}_{i}",
                        "chunk_index": i,
                        "total_chunks": len(split_docs),
                        "content_hash": hashlib.sha256(
                            doc.page_content.encode()
                        ).hexdigest(),
                    }
                )

            logger.info(f"Split {filename} into {len(split_docs)} chunks")

            embeddings = self.embeddings_model.encode(
                [doc.page_content for doc in split_docs]
            )

            vectors = []
            for doc, embedding in zip(split_docs, embeddings):
                metadata_with_text = doc.metadata.copy()
                metadata_with_text["text"] = doc.page_content
                metadata_with_text["text_length"] = len(doc.page_content)

                vector = {
                    "id": doc.metadata["chunk_id"],
                    "values": embedding.tolist(),
                    "metadata": metadata_with_text,
                }
                vectors.append(vector)

            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self.index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)

            processing_time = (datetime.datetime.now() - start_time).total_seconds()

            result = {
                "status": "success",
                "filename": filename,
                "document_id": document_id,
                "file_type": file_extension,
                "file_size": len(content),
                "chunks_ingested": len(split_docs),
                "processing_time": processing_time,
                "vector_count": len(split_docs),
            }

            logger.info(
                f"✓ SUCCESS: {filename} - {len(split_docs)} chunks in {processing_time:.2f}s"
            )
            return result

        except Exception as e:
            processing_time = (datetime.datetime.now() - start_time).total_seconds()
            error_msg = str(e)

            result = {
                "status": "failed",
                "filename": filename,
                "error": error_msg,
                "processing_time": processing_time,
            }

            logger.error(f"✗ FAILED: {filename} - {error_msg}")
            return result

    def find_documents(self) -> List[Path]:
        docs_path = Path(DOCS_FOLDER)

        if not docs_path.exists():
            logger.warning(f"Folder '{DOCS_FOLDER}' does not exist")
            return []

        documents = []
        for ext in SUPPORTED_FILE_TYPES:
            documents.extend(docs_path.glob(f"*{ext}"))

        logger.info(f"Found {len(documents)} documents to process")
        return documents

    def process_batch(self):
        logger.info("=" * 80)
        logger.info("STARTING BATCH DOCUMENT PROCESSING")
        logger.info("=" * 80)

        documents = self.find_documents()

        if not documents:
            logger.warning("No documents found to process")
            return

        results = []
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROCESSES) as executor:
            future_to_doc = {
                executor.submit(self.process_single_document, doc): doc
                for doc in documents
            }

            for future in as_completed(future_to_doc):
                result = future.result()
                results.append(result)

        self.print_summary(results)

    def print_summary(self, results: List[Dict[str, Any]]):
        logger.info("=" * 80)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 80)

        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]

        logger.info(f"Total Documents: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")

        if successful:
            total_chunks = sum(r["chunks_ingested"] for r in successful)
            total_time = sum(r["processing_time"] for r in successful)
            logger.info(f"Total Chunks Ingested: {total_chunks}")
            logger.info(f"Total Processing Time: {total_time:.2f}s")

        if failed:
            logger.info("\nFailed Documents:")
            for r in failed:
                logger.error(f"  - {r['filename']}: {r['error']}")

        summary_file = f"processing_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nDetailed results saved to: {summary_file}")
        logger.info("=" * 80)


def main():
    try:
        processor = DocumentProcessor()
        processor.process_batch()
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
