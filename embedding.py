from sentence_transformers import SentenceTransformer
import json

# TASK 2.1 - Embedding Creation
# ----------

# pre-trained models for comparison
models = {
    "all-MiniLM-L6-v2": SentenceTransformer('all-MiniLM-L6-v2'),
    "distilbert-base-nli-stsb-mean-tokens": SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
}

def create_embedding(data, model_name):
    model = models[model_name]
    embeddings = []
    metadata = []

    for file_name, chunks in data.items():
        for idx, chunk in enumerate(chunks):
            embedding = model.encode(chunk)  # embedding a chunk
            embeddings.append(embedding.tolist())  # add to list
            metadata.append({
                'document': file_name,
                'chunk': chunk,
                'index': idx
            })

    return embeddings, metadata

# load task 1' output file
with open('output/cleaned.json', 'r', encoding='utf-8') as f:
    cleaned_data = json.load(f)

# embedding for each model
for model_name in models:
    embeddings, metadata = create_embedding(cleaned_data, model_name)

    output = {"embeddings": embeddings, "metadata": metadata}
    with open(f'output/embeddings_{model_name}.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)
