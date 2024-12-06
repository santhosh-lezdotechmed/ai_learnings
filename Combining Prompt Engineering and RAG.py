from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import pipeline
import faiss

# 1. Load and process the uploaded document
def load_document(file_path):
    """Reads a text file and splits it into a list of `Document` objects."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    # Create one Document object per paragraph
    return [Document(page_content=paragraph.strip()) for paragraph in content.split("\n") if paragraph.strip()]

# Example: Replace with the path to your uploaded file
file_path = "C:/Users/Santhosh.M/Documents/the-verdict.txt"  # Change this to the path of your uploaded file
docs = load_document(file_path)

# 2. Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Create a FAISS vector store
faiss_index = FAISS.from_documents(docs, embedding_model)

# 4. Define a prompt template for QA
prompt_template = PromptTemplate.from_template(
    "Answer the following question based on the documents: {question}"
)

# 5. Initialize the Hugging Face pipeline
hf_pipeline = pipeline(
    "text-generation",
    model="gpt2",  # Use a valid model from Hugging Face's library
    max_new_tokens=50  # Adjust based on desired output length
)

hf_llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 6. Setup the RetrievalQA system
qa_chain = RetrievalQA.from_chain_type(
    llm=hf_llm,
    chain_type="stuff",
    retriever=faiss_index.as_retriever(),
    return_source_documents=True,  # Return source documents for debugging
)

# 7. Ask a question
question = "What is the main topic of the document?"
output = qa_chain({"query": question})  # Use `call`-like behavior for multiple outputs

# 8. Extract the answer and source documents
answer = output["result"]
source_documents = output["source_documents"]

# Print the answer
print("Answer:", answer)

# # Optionally print the source documents
# print("\nSource Documents:")
# for doc in source_documents:
#     print("-", doc.page_content)
