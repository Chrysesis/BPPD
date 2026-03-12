from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter 

# 1. KEMBALI MENGGUNAKAN OLLAMA (Local LLM Sesuai Proposal Bab 6.a)
from langchain_community.chat_models import ChatOllama

def get_waktu_sekarang():
    tz = ZoneInfo("Asia/Jakarta")
    return datetime.now(tz).strftime("%A, %d %B %Y | Pukul %H:%M WIB")

app = FastAPI(title="API Virtual Assistant Lintas Batas BPPD Kalbar")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    pertanyaan: str
    riwayat: str = "" 

print("Memuat Knowledge Base Lintas Batas BPPD...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./bppd_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# =====================================================================
# 2. MENGHUBUNGKAN KE OLLAMA DI COLAB VIA NGROK (Sesuai Bab 7.d.1)
# =====================================================================
llm = ChatOllama(
    base_url="https://eloy-falconnoid-sulkily.ngrok-free.dev/", # <--- GANTI DENGAN URL NGROK DARI COLAB NANTI
    model="llama3", 
    temperature=0.1
)

prompt_template = """
Anda adalah Asisten Virtual resmi dari Badan Pengelola Perbatasan Daerah (BPPD) Provinsi Kalimantan Barat.
Gunakan gaya bahasa yang ramah, sopan, formal, dan mudah dipahami layaknya manusia sungguhan.

[INFO SISTEM]
Waktu saat ini: {waktu_sekarang}

PANDUAN MENJAWAB (WAJIB DIIKUTI):
1. **Basa-basi & Identitas:** Jika pengguna menyapa (contoh: "Halo", "Selamat pagi"), mengucapkan terima kasih, atau bertanya "Kamu siapa?", balaslah dengan hangat dan natural. Perkenalkan diri Anda sebagai Asisten Virtual BPPD Kalbar. Anda BISA menjawab ini tanpa perlu mencari di database.
2. **Informasi Instansi:** Untuk pertanyaan mengenai profil instansi, tugas dan fungsi (Tupoksi), layanan informasi (PPID), atau operasional PLBN, HANYA gunakan informasi dari [Konteks Dokumen]. Jawablah dengan rapi dan terstruktur (gunakan poin-poin/bullet jika perlu).
3. **Luar Topik:** Jika pengguna bertanya hal di luar BPPD atau pemerintahan (seperti koding, cuaca, dll), tolak dengan sopan dan ingatkan bahwa Anda khusus melayani informasi seputar BPPD Provinsi Kalimantan Barat.
4. **Aturan Salam:** JANGAN mengulangi salam pembuka yang panjang pada setiap jawaban. Langsung to the point untuk menjawab pertanyaan lanjutan agar percakapan terasa mengalir.

ATURAN FORMAT KONTAK (Jika Anda memberikan informasi kontak):
- Untuk Email, gunakan format: [email@domain.com](mailto:email@domain.com)
- Untuk WhatsApp, ubah angka 0 di depan menjadi 62, hilangkan tanda strip (-), gunakan format: [0812-XXXX-XXXX](https://wa.me/62812XXXXXXXX)
- Untuk sosial media, tambahkan link agar saat diklik langsung terbuka ke akun sosial media.

Riwayat Percakapan Sebelumnya:
{riwayat}

Konteks Dokumen:
{context}

Pertanyaan Pengguna Saat Ini: {question}
Jawaban Asisten BPPD:
"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question", "riwayat"], 
    partial_variables={"waktu_sekarang": get_waktu_sekarang}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

@app.get("/")
async def tampilkan_web():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        jawaban_ai = qa_chain.invoke({
            "question": request.pertanyaan,
            "riwayat": request.riwayat
        })
        return {"jawaban": jawaban_ai}
    
    except Exception as e:
        return {"jawaban": f"Mohon maaf, server sedang sibuk. Mohon coba lagi nanti."}
