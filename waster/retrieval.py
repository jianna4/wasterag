from langchain.chains import RetrievalQA

from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.llms import Ollama

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(r"F:\projects\wasteRAG\fol\waster\faiss_indexxx", embeddings,allow_dangerous_deserialization=True)


#now the retriever

retriever = vectorstore.as_retriever()

#create the qa chain
qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="tinyllama"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Example query to the QA chain
query = "are you located in Nakuru?"
result = qa_chain({"query": query})
print("Answer:", result['result'])