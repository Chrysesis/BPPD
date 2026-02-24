from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # IMPORT BARU
from langchain_community.vectorstores import Chroma
import os

# Set API Key Google Gemini (tetap dibutuhkan untuk LLM-nya nanti)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAOGrfkVf9-soLBqcjHtu8SP1rJ8YMhEUY"

def proses_dokumen_bppd(file_name):
    print(f"1. Membaca dokumen: {file_name}...")
    loader = PyPDFLoader(file_name)
    dokumen = loader.load()

    print("2. Memecah teks menjadi potongan kecil (chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(dokumen)

    print("3. Membuat Vector Database dengan HuggingFace (Lokal)...")
    # MENGGUNAKAN EMBEDDING LOKAL YANG SUPER STABIL
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./bppd_db"
    )
    
    print("Selesai! Knowledge Base BPPD berhasil dibuat di folder './bppd_db'")

# Pastikan baris di bawah ini tidak ada tanda '#' agar fungsinya berjalan
proses_dokumen_bppd("Info_BPPD.pdf")