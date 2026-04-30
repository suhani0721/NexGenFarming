import faiss
import os
import numpy as np
#from .embeddings import VECTOR_DB_DIR, VECTOR_DIMENSION
from langsmith import traceable
VECTOR_DIMENSION = 384
VECTOR_DB_DIR = "vectorstores"


def get_user_vector_path(user_id):
    user_dir = os.path.join(VECTOR_DB_DIR, f"user_{user_id}")
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, "index.faiss")

@traceable(name="FAISS Index")
def load_or_create_index(user_id):
    path = get_user_vector_path(user_id)

    if os.path.exists(path):
        return faiss.read_index(path)
    else:
        return faiss.IndexFlatL2(VECTOR_DIMENSION)

@traceable(name="Save FAISS Index")
def save_index(index, user_id):
    path = get_user_vector_path(user_id)
    faiss.write_index(index, path)

@traceable(name="Add Embeddings to FAISS")
def add_embeddings(index, embeddings):
    vectors = np.array(embeddings).astype("float32")
    index.add(vectors)