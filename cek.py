import google.generativeai as genai
import os

# 1. Masukkan API Key kamu di sini
API_KEY = "AIzaSyAOGrfkVf9-soLBqcjHtu8SP1rJ8YMhEUY"

# 2. Konfigurasi SDK Google
genai.configure(api_key=API_KEY)

print("Mencari daftar model yang tersedia...")
print("-" * 50)

# 3. Panggil fungsi ListModels dan tampilkan hasilnya
try:
    for model in genai.list_models():
        # Kita filter agar hanya menampilkan model yang mendukung pembuatan teks (generateContent)
        if 'generateContent' in model.supported_generation_methods:
            print(f"Nama Model : {model.name}")
            print(f"Versi      : {model.version}")
            print("-" * 50)
except Exception as e:
    print(f"Terjadi kesalahan: {e}")