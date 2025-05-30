# Mini RAG System

This repository implements a **Retrieval-Augmented Generation (RAG)** system that combines document retrieval using **FAISS** and response generation using **GPT-4o-mini**.

## Installation Instructions

### 1. Clone the repository

```bash
git clone <repository_url>
cd <repository_directory>
```


### 2. Install dependencies

You can install all required dependencies using **pip**. This project relies on libraries such as `faiss`, `sentence-transformers`, and `openai` for vector embeddings and similarity search.

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory of the project and add your OpenAI API key:

```
OPENAI_API_KEY="your_openai_api_key"
```


---

## Usage

### 1. Preprocessing and Chunking Documents

To preprocess the documents, clean the text, and split them into chunks:

```bash
python clean.py
```

This will generate a JSON file containing document chunks and metadata.

### 2. Create Embeddings and Index with FAISS

To generate embeddings for the document chunks and build the FAISS index:

```bash
python embedding.py
```

This script uses pre-trained models (like **MiniLM-L6-v2**) to create vector embeddings and stores them in the FAISS index for efficient retrieval.

### 3. Query Handling and Response Generation

To handle a query and generate a response based on the retrieved document chunks:

```bash
python pipeline.py
```
or 
```bash
python pipeline_optimize.py
```

This script processes the query, retrieves the top-k relevant chunks using FAISS, and generates a response with GPT-4o-mini using the retrieved context. **To change the query and the history, change the example part in each script.**

---
