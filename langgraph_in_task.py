import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
import fitz  # PyMuPDF
import pandas as pd


# Custom Mytryoshka Loss
class MytryoshkaLoss(nn.Module):
    def __init__(self):
        super(MytryoshkaLoss, self).__init__()

    def forward(self, anchor, positive, negative):
        positive_distance = torch.norm(anchor - positive, p=2, dim=1)
        negative_distance = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.max(positive_distance - negative_distance + 1, torch.tensor(0.0, device=anchor.device))
        return loss.mean()


# Load Pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.split('\n')


# Function to extract text from a TXT file
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.readlines()


# Function to extract data from a CSV file
def extract_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Available columns in CSV: {df.columns.tolist()}")
    if "text" in df.columns:
        print("Using 'text' column by default.")
        return df["text"].dropna().tolist()
    else:
        print("Default 'text' column not found.")
        print("Please select a column to use:")
        for idx, column in enumerate(df.columns):
            print(f"{idx}: {column}")
        col_idx = int(input("Enter the column index to use: "))
        if col_idx < 0 or col_idx >= len(df.columns):
            raise ValueError("Invalid column index selected!")
        selected_column = df.columns[col_idx]
        return df[selected_column].dropna().tolist()


# Load data from various formats
def load_data(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext == ".csv":
        return extract_data_from_csv(file_path)
    else:
        raise ValueError("Unsupported file format! Only PDF, TXT, and CSV are supported.")


# Retrieve relevant sentences and sort by similarity score
def derive_relevant_sentences(query, dataset, similarity_threshold=0.2):  # Lowered threshold for recall improvement
    query_embedding = model.encode([query], convert_to_tensor=True)
    dataset_embeddings = model.encode(dataset, convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding.cpu(), dataset_embeddings.cpu())[0]
    relevant_sentences = [(dataset[i], sim) for i, sim in enumerate(similarities) if sim >= similarity_threshold]
    return sorted(relevant_sentences, key=lambda x: x[1], reverse=True)


# Format results into readable sentences
def format_results(relevant_sentences):
    formatted = []
    for text, score in relevant_sentences:
        if isinstance(text, dict):  # If text is a dictionary, format it
            formatted.append(" and ".join(f"{key}: {value}" for key, value in text.items()))
        else:  # Otherwise, keep the text as is
            formatted.append(text)
    return formatted


# Metrics: MRR, Recall, and NDCG
def evaluate_retrieval(query, relevant_sentences, k=5):
    retrieved = relevant_sentences[:k]
    relevant = [sent[0] for sent in relevant_sentences]  # Ground truth sentences

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

    mrr = mrr_at_k([s[0] for s in retrieved], relevant)
    recall = recall_at_k([s[0] for s in retrieved], relevant, k=k)
    ndcg = ndcg_at_k([s[0] for s in retrieved], relevant, k=k)
    return mrr, recall, ndcg


# Fine-tuning and evaluation
def fine_tune_and_evaluate(query, dataset):
    sorted_sentences = derive_relevant_sentences(query, dataset)
    if not sorted_sentences:
        print("The query is out of context! No relevant information found in the dataset.")
        return

    # Prepare InputExamples with query and relevant sentence pairs
    train_examples = [InputExample(texts=[query, sentence]) for sentence, _ in sorted_sentences]

    # DataLoader with custom collate function
    def collate_fn(batch):
        queries = [ex.texts[0] for ex in batch]
        positives = [ex.texts[1] for ex in batch]
        queries_tensor = torch.tensor(model.encode(queries), requires_grad=True)
        positives_tensor = torch.tensor(model.encode(positives), requires_grad=True)
        negatives_tensor = queries_tensor.clone() + 0.1  # Simulating negative samples
        return queries_tensor, positives_tensor, negatives_tensor

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1, collate_fn=collate_fn)

    # Fine-tune the model
    loss_fn = MytryoshkaLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            anchor, positive, negative = batch
            loss = loss_fn(anchor, positive, negative)  # Calculate the loss using embeddings
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Mytryoshka Loss: {total_loss:.4f}")

    # Save and reload fine-tuned model
    torch.save(model.state_dict(), "finetuned_model.pth")
    model.load_state_dict(torch.load("finetuned_model.pth"))
    model.eval()

    # Evaluation after fine-tuning
    formatted_sentences = format_results(sorted_sentences)
    mrr, recall, ndcg = evaluate_retrieval(query, sorted_sentences)
    print(f"MRR: {mrr:.4f}, Recall@5: {recall:.4f}, NDCG@5: {ndcg:.4f}")
    print("Top relevant sentences retrieved based on the query:")
    for idx, sentence in enumerate(formatted_sentences[:5]):  # Limit output to top 5
        print(f"{idx + 1}: {sentence}")


# Main Execution
file_path = input("Enter the path to your dataset (PDF, TXT, or CSV): ")
dataset_string = load_data(file_path)
query = input("Enter a query: ")
fine_tune_and_evaluate(query, dataset_string)
