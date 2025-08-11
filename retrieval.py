"""
now here we are goingt o try doingthe retrieval of info from the mbedded vector database
"""
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
import os

embeddings = OllamaEmbeddings(model="tinyllama")
print("Loading vector store from 'faiss_index/'...")
vectorstore = FAISS.load_local(r"F:\projects\wasteRAG\fol\faiss_index", embeddings , allow_dangerous_deserialization=True)
print("Vector store loaded.")

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
query = "What are the previous schools?"
result = qa_chain({"query": query})
print("Answer:", result['result'])

"""
now this works perfectly coz theanswer it gave is:
Answer: School ofAgriculture, which was previously named School of AGRICULTURE.
School ofComputing and Information Technology Technology (SCITT), formerly known as Department of Computer Science.
School ofPure, Applied, and Health Sciences (SPAHS), formerly known as Department ofPhysical and Biochemistry.
School ofHumanities and Social Sciences (SHSS), formerly known as Department of Humanities and Communication Studies.
not the best but not the worst either
"""