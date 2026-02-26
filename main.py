from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter # TAMBAHAN: Untuk mengatur alur riwayat

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

# TAMBAHAN: Menerima 'riwayat' dari frontend
class ChatRequest(BaseModel):
    pertanyaan: str
    riwayat: str = "" 

print("Memuat Knowledge Base BPPD...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./bppd_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1) 

# TAMBAHAN: Memasukkan konteks {riwayat} ke dalam prompt
prompt_template = """
Anda adalah Virtual Assistant resmi untuk Badan Pengelola Perbatasan Daerah (BPPD) Provinsi Kalimantan Barat.
Gunakan gaya bahasa yang sopan, formal, informatif, dan sesuai dengan standar komunikasi instansi pemerintahan.

[INFO SISTEM]
Waktu saat ini: {waktu_sekarang}
(Jika pengguna menyapa, balaslah dengan salam yang sesuai dengan waktu di atas).

Tugas Anda HANYA menjawab pertanyaan terkait 4 kategori ini berdasarkan konteks dokumen resmi yang diberikan:
1. Informasi Lintas Batas (syarat dokumen, jam operasional PLBN Entikong, Aruk, Badau, aturan kendaraan).
2. Profil Instansi (visi misi, struktur organisasi, lokasi kantor BPPD Prov. Kalbar).
3. Potensi Perbatasan (destinasi wisata, potensi ekonomi).
4. Layanan Pengaduan (kontak resmi, alur pengaduan).

Aturan Penting:
- Jika informasi TIDAK ADA di dalam konteks, katakan bahwa Anda tidak memiliki informasi tersebut.
- Jangan mengarang informasi (halusinasi).
- Jawab secara terstruktur.

ATURAN FORMAT KONTAK:
- Untuk Email, gunakan format: [email@domain.com](mailto:email@domain.com)
- Untuk WhatsApp, ubah angka 0 di depan menjadi 62, hilangkan tanda strip (-), gunakan format: [0812-XXXX-XXXX](https://wa.me/62812XXXXXXXX)

Riwayat Percakapan Sebelumnya:
{riwayat}

Konteks Dokumen:
{context}

Pertanyaan Pengguna Saat Ini: {question}

Jawaban Virtual Assistant BPPD Kalbar:
"""
PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question", "riwayat"], 
    partial_variables={"waktu_sekarang": get_waktu_sekarang}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# TAMBAHAN: Menyusun chain dengan itemgetter agar menerima riwayat
qa_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs, 
        "question": itemgetter("question"),
        "riwayat": itemgetter("riwayat")
    }
    | PROMPT
    | llm
    | StrOutputParser()
)

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        jawaban_ai = qa_chain.invoke({
            "question": request.pertanyaan,
            "riwayat": request.riwayat
        })
        return {"jawaban": jawaban_ai}
    
    except Exception as e:
        return {"jawaban": f"Mohon maaf, saat ini layanan sedang mengalami gangguan. (Error: {str(e)})"}
