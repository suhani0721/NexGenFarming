import os
import pickle
#from .config import VECTOR_DB_DIR
VECTOR_DB_DIR = "vectorstores"
from langsmith import traceable

def get_metadata_path(user_id):
    user_dir = os.path.join(VECTOR_DB_DIR, f"user_{user_id}")
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, "metadata.pkl")

@traceable(name="Load Metadata")
def load_metadata(user_id):
    path = get_metadata_path(user_id)

    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


@traceable(name="Save Metadata")
def save_metadata(user_id, metadata):
    path = get_metadata_path(user_id)

    with open(path, "wb") as f:
        pickle.dump(metadata, f)