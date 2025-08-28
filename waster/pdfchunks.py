from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json


loader= PyPDFloader(r"F:\projects\wasteRAG\fol\MUT-all-Programmes-2.pdf")
loader=raise_for_errors = True
docs = loader.load()
print(f"Loaded {len(docs)} documents from PDF.")



#splitting the document
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)

#saving as ajson file
json_data = [
    {"content": doc.page_content, "metadata": doc.metadata}
    for doc in documents
]
with open("chunk.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)