import os
import hashlib
import uuid
import datetime
import tempfile
import asyncio
import functools
from typing import List, Union

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from src.file_reader import FileTextExtractor

# --- Environment Variable Loading ---
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --- Configuration Constants ---
PINECONE_INDEX_NAME = "legal-support-ai"
PINECONE_NAMESPACE = "legal-support-ai-docs"
SUPPORTED_FILE_TYPES = [".pdf", ".docx"]
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
MAX_CONCURRENT_PROCESSES = 10  # throttle parallel ingestion tasks

# --- Global Initializations ---
if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Missing required API keys in environment variables")

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

embeddings_model1 = SentenceTransformer('BAAI/bge-m3')

# --- Response Models ---
class IngestionResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size: int
    chunks_ingested: int
    processing_time: float
    vector_count: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
    document: str


# --- FastAPI Application ---
app = FastAPI(
    title="Enhanced Document Ingestion API",
    description="API for ingesting documents with RAG-optimized metadata",
    version="2.1.0"
)


def generate_document_metadata(
    filename: str,
    content: bytes,
    content_type: str,
    document_id: str
) -> dict:
    """Generate comprehensive metadata for RAG retrieval."""
    return {
        "document_id": document_id,
        "source": filename,
        "file_type": content_type,
        "file_size": len(content),
        "ingestion_timestamp": datetime.datetime.now().isoformat(),
        "document_hash": hashlib.sha256(content).hexdigest(),
        "processing_id": str(uuid.uuid4()),
    }


def sync_process_file(
    filename: str,
    content: bytes,
    content_type: str
) -> IngestionResponse:
    """
    Synchronous helper to process a single file:
      - Validate the extension
      - Write to a temp file
      - Extract text
      - Split into chunks + metadata
      - Ingest into Pinecone
      - Return IngestionResponse
    Raises HTTPException on validation or processing failures.
    """
    start_time = datetime.datetime.now()
    file_extension = os.path.splitext(filename)[1].lower()

    # Validate file type
    if file_extension not in SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file_extension}'. Supported types: {SUPPORTED_FILE_TYPES}"
        )

    # Generate a deterministic document ID based on content
    document_id = f"doc_{hashlib.sha256(content).hexdigest()[:16]}"
    metadata = generate_document_metadata(filename, content, content_type, document_id)

    # Write content to a temporary file for extraction
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract text from the temp file
        extractor = FileTextExtractor(file_path=tmp_path)
        extracted_text = extractor.extract_text()

        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text content extracted from document"
            )
    finally:
        # Always clean up the temp file
        os.unlink(tmp_path)

    # Configure text splitter with enhanced metadata handling
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        length_function=len,
        is_separator_regex=False,
    )

    # Create a LangchainDocument and split it
    parent_doc = LangchainDocument(
        page_content=extracted_text,
        metadata=metadata
    )
    split_docs = text_splitter.split_documents([parent_doc])

    # Add chunk-level metadata to each split document
    for i, doc in enumerate(split_docs):
        doc.metadata.update({
            "chunk_id": f"{document_id}_{i}",
            "chunk_index": i,
            "total_chunks": len(split_docs),
            "content_hash": hashlib.sha256(doc.page_content.encode()).hexdigest()
        })

    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Get or create the index
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1024,  # Dimension of Google's embedding-001 model
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region="us-east-1")
        )
    
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Generate embeddings for each chunk
    embeddings = embeddings_model1.encode([doc.page_content for doc in split_docs])
    
    # Prepare vectors for upsert
    vectors = []
    for doc, embedding in zip(split_docs, embeddings):
        vector = {
            'id': doc.metadata['chunk_id'],
            'values': embedding,
            'metadata': doc.metadata
        }
        vectors.append(vector)
    
    # Upsert vectors in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)

    # Compute processing metrics
    processing_time = (datetime.datetime.now() - start_time).total_seconds()

    return IngestionResponse(
        document_id=document_id,
        filename=filename,
        file_type=file_extension,
        file_size=len(content),
        chunks_ingested=len(split_docs),
        processing_time=processing_time,
        vector_count=len(split_docs)
    )


async def process_file(file: UploadFile, sem: asyncio.Semaphore) -> Union[IngestionResponse, ErrorResponse]:
    """
    Async wrapper for processing a single UploadFile with a concurrency limit:
      - Acquires semaphore before proceeding
      - Reads bytes asynchronously
      - Delegates the synchronous work to a threadpool
      - Catches and wraps exceptions into ErrorResponse for batch return
    """
    async with sem:
        try:
            # Asynchronously read file content
            content = await file.read()
            if not content:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Empty file received"
                )

            # Run the synchronous ingestion logic in a threadpool
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                functools.partial(
                    sync_process_file,
                    file.filename,
                    content,
                    file.content_type
                )
            )
            return result

        except HTTPException as he:
            return ErrorResponse(
                error="Processing failure",
                detail=str(he.detail),
                document=file.filename
            )

        except Exception as e:
            return ErrorResponse(
                error="Processing failure",
                detail=str(e),
                document=file.filename
            )


@app.post(
    "/ingest-documents/",
    response_model=List[Union[IngestionResponse, ErrorResponse]],
    status_code=status.HTTP_207_MULTI_STATUS
)
async def ingest_documents_endpoint(files: List[UploadFile] = File(...)):
    """
    Endpoint to ingest multiple documents in parallel, but with a concurrency limit.
    Returns a list of either IngestionResponse (success) or ErrorResponse (failure) for each file.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESSES)

    tasks = [process_file(file, semaphore) for file in files]
    results = await asyncio.gather(*tasks)
    return results


# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "index": PINECONE_INDEX_NAME,
        "namespace": PINECONE_NAMESPACE,
        "embedding_model": "models/embedding-001",
        "max_concurrent_ingestions": MAX_CONCURRENT_PROCESSES
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
