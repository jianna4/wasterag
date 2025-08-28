from langchain.vectorstores import FAISS
import json
import os
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.schema import Document



with open(r"F:\projects\wasteRAG\fol\waster\chunkkk.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert back to Document objects
documents = [Document(page_content=d["content"], metadata=d["metadata"]) for d in data]

model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#now the vector db
vectorstore = FAISS.from_documents(documents, model)
print("Vector store created with FAISS.")
vectorstore.save_local("faiss_indexxx")
print("Vector store saved to 'faiss_indexxx/'")