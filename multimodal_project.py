import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import T5ForConditionalGeneration, T5Tokenizer
from PyPDF2 import PdfReader

# Function to load the model based on user input

def load_embedding_model(model_choice):
    model_dict = {
        "abhinand/MedEmbed-large-v0.1": "abhinand/MedEmbed-large-v0.1",
        "abhinand/MedEmbed-small-v0.1": "abhinand/MedEmbed-small-v0.1",
        "abhinand/MedEmbed-base-v0.1": "abhinand/MedEmbed-base-v0.1",
        "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
        "nomic-ai/nomic-embed-text-v1.5": "nomic-ai/nomic-embed-text-v1.5",
        "mixedbread-ai/mxbai-embed-large-v1":"mixedbread-ai/mxbai-embed-large-v1"
    }

    if model_choice in model_dict:
        if model_choice == "nomic-ai/nomic-embed-text-v1.5":
            return SentenceTransformer(model_dict[model_choice], trust_remote_code=True)
        return SentenceTransformer(model_dict[model_choice])
    else:
        raise ValueError("Unsupported model choice. Please select a valid model.")


# Ask user for model choice
model_choice = input("Select embedding model ('abhinand/MedEmbed-large-v0.1', 'abhinand/MedEmbed-small-v0.1', 'abhinand/MedEmbed-base-v0.1','all-MiniLM-L6-v2', 'nomic-ai/nomic-embed-text-v1.5' or 'mixedbread-ai/mxbai-embed-large-v1'): ")
model = load_embedding_model(model_choice)

# Initialize T5 summarization model and tokenizer
summarization_model = T5ForConditionalGeneration.from_pretrained("t5-large")
summarization_tokenizer = T5Tokenizer.from_pretrained("t5-large")

# Define Mytryoshka Loss
class MytryoshkaLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MytryoshkaLoss, self).__init__()
        self.margin = margin

    def forward(self, query_embeddings, target_embeddings):
        distances = torch.norm(query_embeddings - target_embeddings, dim=1)
        return torch.mean(torch.clamp(self.margin - distances, min=0))

# Function to summarize retrieved results
def summarize_results(results):
    if not results:
        return "No results to summarize."
    text_to_summarize = " ".join([sentence[0] for sentence in results])
    input_ids = summarization_tokenizer.encode("summarize: " + text_to_summarize, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarization_model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Load data from file (CSV, PDF, or TXT)
def load_data(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
        print(f"Available columns in CSV: {df.columns.tolist()}")
        print("Please select a column to use for text extraction:")
        for idx, column in enumerate(df.columns):
            print(f"{idx}: {column}")
        col_idx = int(input("Enter the column index to use: "))
        if col_idx < 0 or col_idx >= len(df.columns):
            raise ValueError("Invalid column index selected!")
        selected_column = df.columns[col_idx]
        return df[selected_column].dropna().tolist()
    elif ext == ".pdf":
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text.split("\n")  # Split the text into lines or sentences
    elif ext == ".txt":
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().splitlines()  # Split the content by lines
    else:
        raise ValueError("Unsupported file format. Only CSV, PDF, and TXT are supported.")

# Adjusting the similarity threshold to increase recall and ensure top-k retrieval
def derive_relevant_sentences(query, dataset, top_k=5, similarity_threshold=0.5):
    query_embedding = model.encode([query], convert_to_tensor=True)
    dataset_embeddings = model.encode(dataset, convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding.cpu(), dataset_embeddings.cpu())[0]
    
    # Filter by threshold
    sorted_sentences = [
        (dataset[idx], similarities[idx])
        for idx, score in enumerate(similarities)
        if score >= similarity_threshold
    ]
    sorted_sentences = sorted(sorted_sentences, key=lambda x: x[1], reverse=True)
    
    # Handle case where no sentences meet the threshold
    if not sorted_sentences:
        print("No sentences meet the similarity threshold.")
        return []
    
    return sorted_sentences[:top_k]

# Evaluate retrieval performance
def evaluate_retrieval(query, relevant_sentences, k=5):
    retrieved = [sent[0] for sent in relevant_sentences[:k]]
    relevant = [sent[0] for sent in relevant_sentences]

    def mrr_at_k(retrieved, relevant):
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                return 1 / rank
        return 0

    def recall_at_k(retrieved, relevant, k=5):
        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        return len(retrieved_at_k.intersection(relevant_set)) / len(relevant_set) if len(relevant_set) > 0 else 1.0

    def ndcg_at_k(retrieved, relevant, k=5):
        dcg = sum(1 / np.log2(i + 2) for i, doc in enumerate(retrieved[:k]) if doc in relevant)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        return dcg / idcg if idcg > 0 else 0

    mrr = mrr_at_k(retrieved, relevant)
    recall = recall_at_k(retrieved, relevant, k)
    ndcg = ndcg_at_k(retrieved, relevant, k)
    return mrr, recall, ndcg

# Plot a similarity heatmap
def plot_similarity_heatmap(relevant_sentences, query):
    labels = [f"Sentence {i+1}" for i in range(len(relevant_sentences))]
    similarities = [score for _, score in relevant_sentences]
    
    # Create a distinct color palette for each sentence
    cmap = sns.color_palette("hsv", len(similarities))  # Different colors for each sentence
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        np.array([similarities]).T,
        annot=True,
        cmap=cmap,
        cbar=True,
        xticklabels=[query],
        yticklabels=labels,
    )
    plt.title("Similarity Heatmap")
    plt.xlabel("Query")
    plt.ylabel("Top Sentences")
    plt.show()

# Fine-tune and evaluate
def fine_tune_until_zero(query, dataset, k, threshold=1e-5):
    sorted_sentences = derive_relevant_sentences(query, dataset, top_k=k)
    if not sorted_sentences:
        print("The query is out of context! No relevant information found in the dataset.")
        return

    # Derive and summarize results
    formatted_sentences = [sentence for sentence, _ in sorted_sentences]
    mrr, recall, ndcg = evaluate_retrieval(query, sorted_sentences, k)
    summary = summarize_results(sorted_sentences)

    print(f"MRR: {mrr:.4f}, Recall: {recall:.4f}, NDCG: {ndcg:.4f}")
    print(f"Abstractive Summary:\n{summary}")
    print(f"Top {k} relevant sentences retrieved based on the query:")
    for idx, (sentence, _) in enumerate(sorted_sentences):
        print(f"{idx + 1}: {sentence}")

    # Plot heatmap
    plot_similarity_heatmap(sorted_sentences, query)

    # Fine-tuning with MytryoshkaLoss
    print("Fine-tuning with MytryoshkaLoss...")
    try:
        query_embeddings = model.encode([query], convert_to_tensor=True)
        target_embeddings = model.encode(formatted_sentences, convert_to_tensor=True)

        query_embeddings.requires_grad_()
        target_embeddings.requires_grad_()

        optimizer = torch.optim.Adam([query_embeddings, target_embeddings], lr=1e-5)
        loss_fn = MytryoshkaLoss()

        epoch = 0
        while True:
            epoch += 1
            optimizer.zero_grad()
            loss = loss_fn(query_embeddings, target_embeddings)
            loss.backward()
            optimizer.step()

            # Stop if loss is smaller than the threshold
            if loss.item() <= threshold:
                print(f"Converged at Epoch {epoch}, Loss: {loss.item():.6f}")
                break

    except RuntimeError as e:
        print(f"RuntimeError encountered during fine-tuning: {str(e)}")

# Main Execution
file_path = input("Enter the path to your dataset (CSV, PDF, or TXT): ")
dataset_string = load_data(file_path)
query = input("Enter a query: ")
k = int(input("Enter the number of top relevant sentences to retrieve (e.g., 5): "))
fine_tune_until_zero(query, dataset_string, k)
