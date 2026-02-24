from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set API Key Google Gemini kamu di sini
os.environ["GOOGLE_API_KEY"] = "AIzaSyAOGrfkVf9-soLBqcjHtu8SP1rJ8YMhEUY"

app = FastAPI()

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
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.3) 

prompt_template = """
Anda adalah Virtual Assistant resmi untuk Badan Pengelola Perbatasan Daerah (BPPD) Provinsi Kalimantan Barat.
Gunakan gaya bahasa gaul anak muda gen z yang mudah dipahami oleh masyarakat luas.

Tugas Anda HANYA menjawab pertanyaan terkait 4 kategori ini berdasarkan konteks yang diberikan:
1. Informasi Lintas Batas (syarat dokumen, jam operasional PLBN Entikong, Aruk, Badau, aturan kendaraan).
2. Profil Instansi (visi misi, struktur organisasi, lokasi kantor BPPD Prov. Kalbar).
3. Potensi Perbatasan (destinasi wisata, potensi ekonomi).
4. Layanan Pengaduan (kontak resmi, alur pengaduan).

Jika informasi yang ditanyakan TIDAK ADA di dalam konteks di bawah ini, katakan dengan sopan bahwa Anda tidak memiliki informasi tersebut dan arahkan pengguna untuk menghubungi kontak resmi BPPD Kalbar. Jangan mengarang informasi (halusinasi).

Konteks Dokumen:
{context}

Pertanyaan Pengguna: {question}

Jawaban Virtual Assistant BPPD Kalbar:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# --- PENDEKATAN MODERN (LCEL) MENGGANTIKAN RetrievalQA ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)
# ---------------------------------------------------------

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_query = request.pertanyaan
    
    try:
        # Panggil AI untuk menjawab pertanyaan menggunakan RAG LCEL
        jawaban_ai = qa_chain.invoke(user_query)
        return {"jawaban": jawaban_ai}
    
    except Exception as e:
        return {"jawaban": f"Mohon maaf, terjadi kesalahan pada sistem kami. Detail: {str(e)}"}