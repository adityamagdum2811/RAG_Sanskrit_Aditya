import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load vector DB
index = faiss.read_index("vector.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# CPU LLM
generator = pipeline(
    "text-generation",
    model="google/flan-t5-base",   # CPU friendly
    max_length=200
)

def retrieve(query):
    q_embedding = embed_model.encode([query])
    D, I = index.search(np.array(q_embedding), k=3)
    
    results = [chunks[i] for i in I[0]]
    return " ".join(results)

def generate_answer(query):
    context = retrieve(query)

    prompt = f"""
    संदर्भ:
    {context}

    प्रश्न:
    {query}

    उत्तर:
    """

    response = generator(prompt)[0]['generated_text']
    return response


# Test
query = input("Enter Sanskrit question: ")
print(generate_answer(query))
