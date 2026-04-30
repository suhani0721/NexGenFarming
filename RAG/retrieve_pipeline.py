import numpy as np
from .embedding import embed_text
from .vectordb import load_or_create_index
from .metadata_store import load_metadata
#from .config import TOP_K
from langsmith import traceable
TOP_K = 3

@traceable(name="Retrieve Memory")
def retrieve_memory(user_id, question):
    # 1️⃣ Load index
    index = load_or_create_index(user_id)

    if index.ntotal == 0:
        return []

    # 2️⃣ Embed query
    query_embedding = embed_text(question)
    query_vector = np.array([query_embedding]).astype("float32")

    # 3️⃣ Search
    D, I = index.search(query_vector, TOP_K)

    # 4️⃣ Load metadata
    metadata = load_metadata(user_id)

    retrieved_chunks = []
    for idx in I[0]:
        if idx < len(metadata):
            retrieved_chunks.append(metadata[idx]["chunk"])

    return retrieved_chunks