import warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import os
import re
import numpy as np
# Load model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to encode text to embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    # Use the mean of the last hidden state for sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

# Function to parse and split PDF into sentences
def parse_document(file_path):
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + " "
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [sentence for sentence in sentences if sentence.strip()]  # Remove empty sentences

# Load and embed sentences from PDF
file_path = "C:/Users/Santhosh.M/Downloads/Redacted.pdf"
sentences = parse_document(file_path)
sentence_embeddings = [get_embedding(sentence) for sentence in sentences]

# Encode query
query = "Has the patient experienced any lower extremity pain post-surgery?"
query_embedding = get_embedding(query)

# Calculate cosine similarities
sentence_embeddings = np.vstack(sentence_embeddings)
similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]

# Retrieve sentences with similarity above the threshold
threshold = 0.7
responses = [sentences[i] for i, sim in enumerate(similarities) if sim > threshold]

# Display results
if responses:
    print("Relevant Sentences:")
    for response in responses:
        print(response)
else:
    print("No similar sentences found.")