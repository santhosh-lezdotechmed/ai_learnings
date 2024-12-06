from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
# Load model directly
from transformers import AutoModel
embedding_model = AutoModel.from_pretrained("ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1", trust_remote_code=True)
# Step 1: Load the embedding model


# Example medical documents to index
documents = [
    "The patient was diagnosed with hypertension and prescribed medication.",
    "Symptoms include fever, chills, and persistent cough.",
    "CT scans showed a mass in the lower lobe of the right lung.",
    "Patient history includes type 2 diabetes and chronic hypertension.",
    "The biopsy results indicated a malignant tumor."
]

# Step 2: Generate embeddings for each document
document_embeddings = embedding_model.encode(documents)

# Step 3: Initialize FAISS index and add embeddings
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # Using L2 (Euclidean) distance for similarity search
index.add(document_embeddings)

# Step 4: Define the retrieval function
def retrieve_documents(query, k=2):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[idx] for idx in indices[0]]

# Step 5: Initialize the language model pipeline for generating responses
generator = pipeline("text-generation", model="gpt2")  # You can replace with a more suitable medical model

# Step 6: Define RAG pipeline function
def rag_pipeline(query):
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query)
    # Concatenate retrieved documents into a prompt for generation
    context = " ".join(retrieved_docs)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    
    # Generate answer based on context and query
    answer = generator(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
    return answer

# Example query
query = "What are the potential treatments for hypertension?"
response = rag_pipeline(query)

print("Response:", response)
