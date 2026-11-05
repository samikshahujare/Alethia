import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.store import get_vector_store

store = get_vector_store()
res = store.collection.get()
print(f"Total chunks: {len(res['ids'])}")
for i in range(len(res['ids'])):
    cid = res['ids'][i]
    src = res['metadatas'][i]['source']
    doc = res['documents'][i].strip().replace('\n', ' ')
    print(f"[{cid}] ({src}): {doc}")
