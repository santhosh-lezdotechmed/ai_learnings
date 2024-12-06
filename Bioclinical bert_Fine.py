import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load MedEmbed model (ensure model path is correct)
model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")

def extract_text_from_pdf(pdf_path):
    """
    Extract text from each page of a PDF.
    """
    doc = fitz.open(pdf_path)
    text_data = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")
        text_data.append(text)
    doc.close()
    return text_data

def embed_text(text_data, model):
    """
    Generate embeddings for the extracted text using the provided model.
    """
    embeddings = model.encode(text_data, convert_to_tensor=True)
    return embeddings

# Extract text from PDF
pdf_text = extract_text_from_pdf("C:/Users/Santhosh.M/Documents/Test_pdf.pdf")

# Generate embeddings for the extracted text
embeddings = embed_text(pdf_text, model)

# Convert embeddings to a NumPy array for FAISS compatibility
embeddings_np = embeddings.cpu().detach().numpy()  # Ensure embeddings are on CPU
d = embeddings_np.shape[1]  # Dimension of embeddings

# Initialize FAISS index
index = faiss.IndexFlatL2(d)  # FAISS index with L2 (Euclidean) distance
index.add(embeddings_np)  # Add embeddings to the index


def query_pdf(query_text, model, index, original_texts, top_k=3):
    """
    Query the FAISS index with a text query, and retrieve the top_k most relevant sentences.
    """
    # Embed the query
    query_embedding = model.encode([query_text], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().detach().numpy()
    
    # Perform the search in FAISS index
    _, indices = index.search(query_embedding_np, top_k)
    
    # Retrieve and return the top_k results
    results = [original_texts[i] for i in indices[0]]
    return results

# Example query
query = "What is the history of present illness?"

# Retrieve top results for the query
top_results = query_pdf(query, model, index, pdf_text)

# Display top results
for i, result in enumerate(top_results):
    print(f"Result {i + 1}:\n{result}\n")
