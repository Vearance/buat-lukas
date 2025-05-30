import faiss
import numpy as np
import re
import json
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# Task 3.1 - Contextual Query Handling

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


def weighted_query(query, query_history, top_k=3, current_weight=0.7):

    query_cleaned = preprocess_text(query)
    print(query_cleaned)
    query_embed = model.encode(query_cleaned)  # encode query
    query_embed = query_embed / np.linalg.norm(query_embed)  # normalize, as the index contains normalized embeddings

    if query_history:
        history_embed = model.encode(" ".join(query_history))
        history_embed = history_embed / np.linalg.norm(history_embed)
    else:
        history_embed = np.zeros_like(query_embed)  # no history

    # weight
    combined_query = current_weight * query_embed + (1 - current_weight) * history_embed

    svec = np.array(combined_query).reshape(1, -1)
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
query_history = ['Ducati Panigale V4']
current_query = "engine power"

top_chunks = weighted_query(current_query, query_history)
for i, chunk in enumerate(top_chunks):
    print(f"Top-{i+1} Chunk:")
    print(f"Document: {chunk['document']} in chunk index {chunk['index']}")
    print(f"Text: {chunk['chunk']}")
    print(f"Distance: {chunk['distance']:.4f}")
    print("-" * 30)
