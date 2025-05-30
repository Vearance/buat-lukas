import faiss
import time
import numpy as np
import re
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# Task 3.2 - Pipeline Simulation
# ----------

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('output/embeddings_all-MiniLM-L6-v2.json', 'r', encoding='utf-8') as f:
    data_minilm = json.load(f)

embeddings = data_minilm['embeddings']
metadata = data_minilm['metadata']

norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

timing_data = {
    "retrieval_times": [],
    "generation_times": []
}

# wrapper function to measure execution time
def timed_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time

    if func.__name__ == "weighted_query":
        timing_data["retrieval_times"].append(elapsed_time)
    elif func.__name__ == "response_with_api":
        timing_data["generation_times"].append(elapsed_time)

    print(f"{func.__name__} took {elapsed_time:.4f} seconds.")
    return result

def clean_text(text):
    text = re.sub(r"[^\w\s:@\-\.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text):
    cleaned_text = clean_text(text)
    sentences = sent_tokenize(cleaned_text)
    stop_words = set(stopwords.words('english'))

    cleaned = []
    for sentence in sentences:
        words = sentence.split()
        filtered = [word for word in words if word.lower() not in stop_words]
        cleaned.append(' '.join(filtered))

    return cleaned

def weighted_query(query, query_history, top_k=3, current_weight=0.7):
    query_cleaned = preprocess_text(query)
    query_embed = model.encode(query_cleaned)
    query_embed = query_embed / np.linalg.norm(query_embed)

    if query_history:
        history_embed = model.encode(" ".join(query_history))
        history_embed = history_embed / np.linalg.norm(history_embed)
    else:
        history_embed = np.zeros_like(query_embed)

    combined_query = current_weight * query_embed + (1 - current_weight) * history_embed
    svec = np.array(combined_query).reshape(1, -1)
    distances, pos = index.search(svec, k=top_k)

    results = []
    for i, idx in enumerate(pos[0]):
        results.append({
            "chunk": metadata[idx]["chunk"],
            "document": metadata[idx]["document"],
            "index": metadata[idx]["index"],
            "distance": distances[0][i]
        })
    return results

def overlap_handling(top_chunks):
    context = top_chunks[0]['chunk']
    for chunk in top_chunks[1:]:
        context += " " + " ".join(chunk['chunk'].split()[50:])
    return context

# OpenAI API setup
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def response_with_api(query, query_history, top_k=3):
    top_chunks = weighted_query(query, query_history, top_k=top_k)

    context = overlap_handling(top_chunks)

    input_prompt = f"""
    You are an intelligent assistant. Use the context below to answer the query in a concise and accurate manner:

    Context:
    {context}

    Query:
    {query}

    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": input_prompt
            }
        ],
        max_tokens=500,
        temperature=0.7,
    )

    generated_text = response.choices[0].message.content
    return generated_text, top_chunks

# example
query_history = ["Duke Engine", "liquid cooled displacement"]
current_query = "What is the engine power?"

response, chunks = timed_function(response_with_api, current_query, query_history)

print("Generated Response:")
print(response)
print("\nRetrieved Chunks:")
for i, chunk in enumerate(chunks):
    print(f"Top-{i+1} Chunk:")
    print(f"Document: {chunk['document']} in chunk index {chunk['index']}")
    print(f"Text: {chunk['chunk']}")
    print(f"Distance: {chunk['distance']:.4f}")
    print("-" * 30)


print("\nTime:")
if timing_data['retrieval_times']:
    avg_retrieval_time = sum(timing_data['retrieval_times']) / len(timing_data['retrieval_times'])
    print(f"Average retrieval time: {avg_retrieval_time:.4f} seconds.")
else:
    print("No retrieval time.")

if timing_data['generation_times']:
    avg_generation_time = sum(timing_data['generation_times']) / len(timing_data['generation_times'])
    print(f"Average generation time: {avg_generation_time:.4f} seconds.")
else:
    print("No generation time.")
