from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
# Tambahkan dotenv untuk keamanan API Key
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Memuat variabel environment dari file .env
load_dotenv()

def get_waktu_sekarang():
    tz = ZoneInfo("Asia/Jakarta")
    return datetime.now(tz).strftime("%A, %d %B %Y | Pukul %H:%M WIB")

app = FastAPI(title="API Virtual Assistant BPPD Kalbar")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    pertanyaan: str

print("Memuat Knowledge Base BPPD...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./bppd_db", embedding_function=embeddings)
# Menambahkan pencarian berdasarkan similarity
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Catatan: Gunakan gemini-1.5-flash jika gemini-3-flash-preview belum stabil
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1) 

# PERBAIKAN PROMPT: Disesuaikan dengan tujuan proposal (Sopan & Formal)
prompt_template = """
Anda adalah Virtual Assistant resmi untuk Badan Pengelola Perbatasan Daerah (BPPD) Provinsi Kalimantan Barat.
Gunakan gaya bahasa yang sopan, formal, informatif, tidak kaku, dan sesuai dengan standar komunikasi instansi pemerintahan.

Tugas Anda HANYA menjawab pertanyaan terkait 4 kategori ini berdasarkan konteks dokumen resmi yang diberikan:
1. Informasi Lintas Batas (syarat dokumen, jam operasional PLBN Entikong, Aruk, Badau, aturan kendaraan).
2. Profil Instansi (visi misi, struktur organisasi, lokasi kantor BPPD Prov. Kalbar).
3. Potensi Perbatasan (destinasi wisata, potensi ekonomi).
4. Layanan Pengaduan (kontak resmi, alur pengaduan).

Aturan Penting:
- Jika informasi yang ditanyakan TIDAK ADA di dalam konteks di bawah ini, katakan dengan sopan bahwa Anda tidak memiliki informasi tersebut dan arahkan pengguna untuk menghubungi kontak resmi pengaduan BPPD Kalbar. 
- Jangan pernah mengarang informasi (halusinasi).
- Jawab secara terstruktur (gunakan bullet points jika perlu agar mudah dibaca).

Konteks Dokumen:
{context}

Pertanyaan Pengguna: {question}

Jawaban Virtual Assistant BPPD Kalbar:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"], partial_variables={"waktu_sekarang": get_waktu_sekarang}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.pertanyaan
    
    try:
        jawaban_ai = qa_chain.invoke(user_query)
        return {"jawaban": jawaban_ai}
    
    except Exception as e:
        return {"jawaban": f"Mohon maaf, saat ini layanan sedang mengalami gangguan. Silakan coba beberapa saat lagi. (Error: {str(e)})"}
