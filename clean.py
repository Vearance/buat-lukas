import os
import re
import json
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# un-comment this if haven't download these
# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

# TASK 1.1 - Data Ingestion and Cleaning
# ----------

def clean_text(text):
    # remove characters except for meaningful ones
    text = re.sub(r"[^\w\s:@\-\.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()  # normalize newlines

    return text


def preprocess_text(path):
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()

    # cleans the text
    cleaned_text = clean_text(text)

    # tokenize into sentences
    sentences = sent_tokenize(cleaned_text)

    # set stop words
    stop_words = set(stopwords.words('english'))

    cleaned = []
    for i in sentences:
        words = i.split()
        filtered = [
            word for word in words if word.lower() not in stop_words
        ]  # remove stop words
        sen = ' '.join(filtered)
        cleaned.append(sen)

    return cleaned


# TASK 1.2 - Chunking Text
# ----------

def chunking_text(sentences):
    max_tokens = 300
    overlap = 50
    chunks = []
    curr_chunk = []
    curr_length = 0

    for i in sentences:
        sentence_tokens = i.split()
        sentence_length = len(sentence_tokens)

        # if sentence added > max_tokens, save chunk
        if curr_length + sentence_length > max_tokens:
            chunks.append(curr_chunk)

            # start a new chunk with overlap
            curr_chunk = curr_chunk[- overlap:]
            curr_length = len(' '.join(curr_chunk).split())  # check how many words in chunk

        # add sentence to chunk
        curr_chunk.extend(sentence_tokens)
        curr_length += sentence_length

    # add the last chunk
    if curr_chunk:
        chunks.append(curr_chunk)

    # change chunks into strings
    return [' '.join(chunk) for chunk in chunks]


def process(path):
    results = {}
    for file_name in os.listdir(path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(path, file_name)
            cleaned_sentences = preprocess_text(file_path)

            chunks = chunking_text(cleaned_sentences)
            results[file_name] = chunks

    return results

directory = 'files'
data = process(directory)

# save as JSON
with open('output/cleaned.json', 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=4)
