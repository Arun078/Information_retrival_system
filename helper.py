import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
GROQ_APY_KEY = os.getenv("GROQ_APY_KEY")
os.environ["GROQ_APY_KEY"] = GROQ_APY_KEY

HUGGING_FACE_APY_KEY = os.getenv("HUGGING_FACE_APY_KEY")
os.environ["HUGGING_FACE_APY_KEY"] = HUGGING_FACE_APY_KEY

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks): 
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en"
    )
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=GROQ_APY_KEY,
        temperature=0.7
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain