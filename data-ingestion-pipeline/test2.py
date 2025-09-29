import os
import hashlib
import uuid
import datetime
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from dotenv import load_dotenv
from pydantic import BaseModel
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

# --- Global Initializations ---
if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Missing required API keys in environment variables")

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=GOOGLE_API_KEY
)


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
    version="2.0.0"
)

def generate_document_metadata(
    file: UploadFile, 
    content: bytes,
    document_id: str
) -> dict:
    """Generate comprehensive metadata for RAG retrieval"""
    return {
        "document_id": document_id,
        "source": file.filename,
        "file_type": file.content_type,
        "file_size": len(content),
        "ingestion_timestamp": datetime.datetime.now().isoformat(),
        "document_hash": hashlib.sha256(content).hexdigest(),
        "processing_id": str(uuid.uuid4()),
    }

@app.post(
    "/ingest-document/", 
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def ingest_document_endpoint(file: UploadFile = File(...)):
    start_time = datetime.datetime.now()
    file_extension = os.path.splitext(file.filename)[1].lower()

    # Validate file type
    if file_extension not in SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Supported types: {SUPPORTED_FILE_TYPES}"
        )

    try:
        # Read and validate file content
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Empty file received"
            )
        
        # Generate document identifiers
        document_id = f"doc_{hashlib.sha256(content).hexdigest()[:16]}"
        metadata = generate_document_metadata(file, content, document_id)
        
        # Process file using temporary file
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Extract text
            extractor = FileTextExtractor(file_path=tmp_path)
            extracted_text = extractor.extract_text()
            
            if not extracted_text.strip():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="No text content extracted from document"
                )
        finally:
            os.unlink(tmp_path)  # Ensure temp file cleanup

        # Configure text splitter with enhanced metadata handling
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

        # Ingest into Pinecone with RAG-optimized settings
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
        
        return IngestionResponse(
            document_id=document_id,
            filename=file.filename,
            file_type=file_extension,
            file_size=len(content),
            chunks_ingested=len(split_docs),
            processing_time=processing_time,
            vector_count=len(split_docs)
        )

    except HTTPException:
        raise  # Re-raise handled exceptions
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Processing failure",
                "detail": str(e),
                "document": file.filename
            }
        )

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "index": PINECONE_INDEX_NAME,
        "namespace": PINECONE_NAMESPACE,
        "embedding_model": "models/embedding-001"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)