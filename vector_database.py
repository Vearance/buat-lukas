import faiss
import numpy as np
import re
import json
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# Task 2.2 - Vector Storage and Retrieval

# use miniLM
model = SentenceTransformer('all-MiniLM-L6-v2')

with open('output/embeddings_all-MiniLM-L6-v2.json', 'r', encoding='utf-8') as f:
    data_minilm = json.load(f)

embeddings = data_minilm['embeddings']
metadata = data_minilm['metadata']

# normalize embeddings to unit vectors
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
# divide each embedding by its norm to get unit vectors
embeddings = embeddings / norms


dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity

index.add(embeddings)

# the same function as in clean.py, but modified to have an input of text instead of path
def clean_text(text):
    text = re.sub(r"[^\w\s:@\-\.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text):
    cleaned_text = clean_text(text)
    sentences = sent_tokenize(cleaned_text)
    stop_words = set(stopwords.words('english'))

    cleaned = []
    for i in sentences:
        words = i.split()
        filtered = [
            word for word in words if word.lower() not in stop_words
        ]
        sen = ' '.join(filtered)
        cleaned.append(sen)

    return cleaned


def retrieve_chunks(query, top_k=3):

    query_cleaned = preprocess_text(query)
    print(query_cleaned)
    query_embed = model.encode(query_cleaned)  # encode query
    query_embed = query_embed / np.linalg.norm(query_embed)  # normalize, as the index contains normalized embeddings

    svec = np.array(query_embed).reshape(1, -1)  # expects 2D numpy array
    # or svec = np.array([query_embed])

    distances, pos = index.search(svec, k=top_k)

    # retrieve data
    results = []
    for i, idx in enumerate(pos[0]):
        results.append({
            "chunk": metadata[idx]["chunk"],
            "document": metadata[idx]["document"],
            "index": metadata[idx]["index"],
            "distance": distances[0][i]
        })
    return results

# example
query = "William Shakespeare plays"
top_chunks = retrieve_chunks(query)
for i, chunk in enumerate(top_chunks):
    print(f"Top-{i+1} Chunk:")
    print(f"Document: {chunk['document']} in chunk index {chunk['index']}")
    print(f"Text: {chunk['chunk']}")
    print(f"Distance: {chunk['distance']:.4f}")
    print("-" * 30)
