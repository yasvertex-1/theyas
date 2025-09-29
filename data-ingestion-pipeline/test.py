import os
import hashlib
import uuid
import datetime
import tempfile
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document as LangchainDocument
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
MAX_CONCURRENT_PROCESSES = 5  # Limit concurrent processing

# --- Global Initializations ---
if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Missing required API keys in environment variables")

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=GOOGLE_API_KEY
)

# --- Response Models ---
class BatchIngestionItem(BaseModel):
    filename: str
    status: str  # "success" or "error"
    document_id: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    chunks_ingested: Optional[int] = None
    processing_time: Optional[float] = None
    vector_count: Optional[int] = None
    error_type: Optional[str] = None
    error_detail: Optional[str] = None

# --- FastAPI Application ---
app = FastAPI(
    title="Enhanced Batch Document Ingestion API",
    description="API for ingesting multiple documents asynchronously with parallel processing",
    version="2.1.0"
)

def generate_document_metadata(
    filename: str,
    content_type: str,
    content: bytes,
    document_id: str
) -> dict:
    """Generate comprehensive metadata for RAG retrieval"""
    return {
        "document_id": document_id,
        "source": filename,
        "file_type": content_type,
        "file_size": len(content),
        "ingestion_timestamp": datetime.datetime.now().isoformat(),
        "document_hash": hashlib.sha256(content).hexdigest(),
        "processing_id": str(uuid.uuid4()),
    }

async def process_single_file(file: UploadFile) -> BatchIngestionItem:
    """Process a single file and return ingestion result"""
    start_time = datetime.datetime.now()
    filename = file.filename
    file_extension = os.path.splitext(filename)[1].lower()
    
    # Validate file type
    if file_extension not in SUPPORTED_FILE_TYPES:
        return BatchIngestionItem(
            filename=filename,
            status="error",
            error_type="UnsupportedFileType",
            error_detail=f"Unsupported file type. Supported types: {SUPPORTED_FILE_TYPES}"
        )
    
    try:
        # Read and validate file content
        content = await file.read()
        if not content:
            return BatchIngestionItem(
                filename=filename,
                status="error",
                error_type="EmptyFile",
                error_detail="Empty file received"
            )
        
        # Generate document identifiers
        document_id = f"doc_{hashlib.sha256(content).hexdigest()[:16]}"
        metadata = generate_document_metadata(
            filename,
            file.content_type,
            content,
            document_id
        )
        
        # Process file using temporary file
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Extract text
            extractor = FileTextExtractor(file_path=tmp_path)
            extracted_text = extractor.extract_text()
            
            if not extracted_text.strip():
                return BatchIngestionItem(
                    filename=filename,
                    status="error",
                    error_type="NoTextContent",
                    error_detail="No text content extracted from document"
                )
        finally:
            os.unlink(tmp_path)  # Ensure temp file cleanup

        # Configure text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Create and split documents
        parent_doc = LangchainDocument(
            page_content=extracted_text, 
            metadata=metadata
        )
        split_docs = text_splitter.split_documents([parent_doc])
        
        # Add chunk-level metadata
        for i, doc in enumerate(split_docs):
            doc.metadata.update({
                "chunk_id": f"{document_id}_{i}",
                "chunk_index": i,
                "total_chunks": len(split_docs),
                "content_hash": hashlib.sha256(doc.page_content.encode()).hexdigest()
            })

        # Ingest into Pinecone
        PineconeVectorStore.from_documents(
            documents=split_docs,
            embedding=embeddings_model,
            index_name=PINECONE_INDEX_NAME,
            namespace=PINECONE_NAMESPACE,
            batch_size=100,
            text_key="text"
        )

        # Calculate processing metrics
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        
        return BatchIngestionItem(
            filename=filename,
            status="success",
            document_id=document_id,
            file_type=file_extension,
            file_size=len(content),
            chunks_ingested=len(split_docs),
            processing_time=processing_time,
            vector_count=len(split_docs))
            
    except Exception as e:
        return BatchIngestionItem(
            filename=filename,
            status="error",
            error_type="ProcessingError",
            error_detail=str(e)
        )

@app.post(
    "/ingest-documents/", 
    response_model=List[BatchIngestionItem],
    status_code=status.HTTP_207_MULTI_STATUS,
    summary="Ingest multiple documents in parallel",
    description="Process multiple files asynchronously with parallel execution. Returns individual status for each file."
)
async def ingest_documents_endpoint(files: List[UploadFile] = File(...)):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESSES)
    
    async def process_with_concurrency(file):
        async with semaphore:
            return await process_single_file(file)
    
    tasks = [process_with_concurrency(file) for file in files]
    return await asyncio.gather(*tasks)

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "index": PINECONE_INDEX_NAME,
        "namespace": PINECONE_NAMESPACE,
        "embedding_model": "models/embedding-001",
        "max_concurrent": MAX_CONCURRENT_PROCESSES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)