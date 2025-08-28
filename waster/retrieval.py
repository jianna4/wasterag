from langchain.chains import RetrievalQA

from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(r"F:\projects\wasteRAG\fol\waster\faiss_indexxx", embeddings,allow_dangerous_deserialization=True)


#now the retriever

retriever = vectorstore.as_retriever()

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",   # or "gpt-4o", "gpt-3.5-turbo"
    temperature=0,
    api_key=api_key     # from your .env
)

#create the qa chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Example query to the QA chain
query = "are you located in Nakuru?"
result = qa_chain({"query": query})
print("Answer:", result['result'])