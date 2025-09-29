import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

def create_pinecone_index():
    PC = Pinecone(
         api_key=os.environ.get("PINECONE_API_KEY"))
    
    index_name = "legal-support-ai"
    dimension = 1024  # embedding dimension
    
    if index_name not in PC.list_indexes().names():
        PC.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Created index {index_name}")
    else:
        print(f"Index {index_name} already exists")

if __name__ == "__main__":
    create_pinecone_index()