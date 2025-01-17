#!pip install sentence-transformers
#!pip install pypdf
import warnings
warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("allenai/biomed_roberta_base")

from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import os
import re

os.environ["HF_TOKEN"] = "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"

file_path = "C:/Users/Santhosh.M/Downloads/Redacted.pdf"

def parse_document(file_path):
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + " "
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences

sentences = parse_document(file_path)


embeddings = model.encode(sentences)


query = "What was the GCS of patient on 18/10/2022 ?"

query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, embeddings)[0]


# print("Shape of similarity matrix:", similarities.shape)
# print("Similarity matrix:\n", similarities)
threshold=0.5
response = None
for i, sim in enumerate(similarities):
    if sim > threshold:
        response = f"{sentences[i]}"
        break
if response:
    print(response)
else:
    print("No similar sentences found.")