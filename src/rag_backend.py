import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ====================== CONFIG ======================
DATA_FOLDER = "data"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"

# ====================== STEP 1: LOAD & CHUNK DOCS ======================
def load_documents():
    pdf_loader = PyPDFDirectoryLoader(DATA_FOLDER)
    docs = pdf_loader.load()
    print(f"Loaded {len(docs)} documents")
    return docs

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    return chunks

# ====================== STEP 2: CREATE VECTOR STORE ======================
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("✅ Vector store created!")
    return vectorstore

# ====================== STEP 3: RAG CHAIN ======================
def create_rag_chain(vectorstore):
    llm = Ollama(model=LLM_MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    template = """You are GaruanCDX Knowledge Assistant.
    Answer the question using ONLY the context below.
    If you don't know, say "I don't have that information in my knowledge base."
    
    Context: {context}
    
    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ====================== MAIN ======================
if __name__ == "__main__":
    print("🚀 Starting GaruanCDX RAG Setup...")
    
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = create_vector_store(chunks)
    
    print("✅ Setup complete! Ready for testing.")

# ====================== QUICK TEST ======================
if __name__ == "__main__":
    print("🚀 Starting GaruanCDX RAG Setup...")
    
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = create_vector_store(chunks)
    
    print("✅ Vector store created!")
    
    # === TEST THE CHATBOT ===
    print("\n🔥 Testing RAG Chatbot...")
    chain = create_rag_chain(vectorstore)
    
    while True:
        question = input("\n🤖 Ask a question (or type 'exit' to stop): ")
        if question.lower() == 'exit':
            break
        answer = chain.invoke(question)
        print("\n📝 Answer:", answer)