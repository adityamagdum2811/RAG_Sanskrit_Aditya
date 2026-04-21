from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load file
with open("data/sanskrit.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

# Load embedding model (CPU friendly)
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

# Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save files
faiss.write_index(index, "vector.index")

# Save chunks
import pickle
with open("chunks.pkl", "wb") as f:
    
    pickle.dump(chunks, f)

print("✅ Ingestion complete")
