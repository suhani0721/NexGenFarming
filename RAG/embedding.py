from sentence_transformers import SentenceTransformer
#from .config import EMBEDDING_MODEL
from langsmith import traceable
VECTOR_DB_DIR = "vectorstores"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384
TOP_K = 3

model = SentenceTransformer(EMBEDDING_MODEL)

@traceable(name="Embedding Step")
def embed_text(text):
    return model.encode(text)

@traceable(name="Chunking Step")
def chunk_text(text, chunk_size=500, overlap=75):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks