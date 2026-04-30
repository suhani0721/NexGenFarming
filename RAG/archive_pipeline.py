from .embedding import chunk_text
from .embedding import embed_text
from .vectordb import load_or_create_index, save_index, add_embeddings
from .metadata_store import load_metadata, save_metadata
from langsmith import traceable
from datetime import datetime

@traceable(name="Archive Session")
def archive_session(user_id, messages):
    """
    messages = [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """

    # 1️⃣ Convert messages to text block
    full_text = ""
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        full_text += f"{role}: {content}\n"

    # 2️⃣ Chunk conversation
    chunks = chunk_text(full_text)

    # 3️⃣ Embed chunks
    import numpy as np

    embeddings = np.array(
        [embed_text(chunk) for chunk in chunks]
    ).astype("float32")
    # 4️⃣ Load or create index
    index = load_or_create_index(user_id)

    # 5️⃣ Add embeddings
    add_embeddings(index, embeddings)

    # 6️⃣ Save index
    save_index(index, user_id)

    # 7️⃣ Store metadata
    metadata = load_metadata(user_id)

    for chunk in chunks:
        metadata.append({
            "user_id": user_id,
            "chunk": chunk,
            "timestamp": str(datetime.now())
        })

    save_metadata(user_id, metadata)

    return True