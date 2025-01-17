#pip install sentence-transformers
#pip install pypdf

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")

from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import os
import re

os.environ["HF_TOKEN"] = "hjl526KwinBvySEJWE7au8bC4llCaFt7G7THsYra"

file_path = "C:/Users/Santhosh.M/Downloads/Medical_Records.pdf"


def parse_document(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    sentences = re.split(r"(?<=[.!?]) +", text)
    return sentences


sentences = parse_document(file_path)


embeddings = model.encode(sentences)


query = "What are the benefits of digitizing medical records?"

query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, embeddings)[0]


# print("Shape of similarity matrix:", similarities.shape)
# print("Similarity matrix:\n", similarities)
threshold = 0.7
response = None
for i, sim in enumerate(similarities):
    if sim > threshold:
        response = f"{sentences[i]}"
        break
if response:
    print(response)
else:
    print("No similar sentences found.")
