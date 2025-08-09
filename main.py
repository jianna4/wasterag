import os
from langchain_community.document_loaders import TextLoader ,Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import os
# Load the text filE

loader = Docx2txtLoader(r"F:\projects\wasteRAG\fol\wastewise.docx")
loader.raise_for_errors = True  # Ensure errors are raised for debugging
docs= loader.load()
print(f"Loaded {len(docs)} documents.")
# Split the documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

# Create embeddings for the text chunks

embeddings = OllamaEmbeddings(model="tinyllama")
print("Creating embeddings for the text chunks...")
# Create a vector store from the embeddings
vectorstore = FAISS.from_documents(texts, embeddings)
print("Vector store created with FAISS.")
# Create a retrieval-based question-answering chain
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="tinyllama"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
print("RetrievalQA chain created.")
# Example query to the QA chain

query = "What is the main topic of the document?"
result = qa_chain({"query": query})
print("Answer:", result['result'])