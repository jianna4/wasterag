""" now here we are going to embedd and vector our loaded chunked documents to later rertieve from it"""

from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import json
import os

with open("chunks.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
# Convert JSON data to LangChain Documents
documents = [
    Document(page_content=item["content"], metadata=item["metadata"])
    for item in json_data
]

embeddings  = OllamaEmbeddings(model="tinyllama")
print("Creating embeddings for the documents...")

# Create a vector store from the documents
vectorstore = FAISS.from_documents(documents, embeddings)
print("Vector store created with FAISS.")

# Save vector store locally
vectorstore.save_local("faiss_index")
print("Vector store saved to 'faiss_index/'")