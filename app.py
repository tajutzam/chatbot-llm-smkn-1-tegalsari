import os
import shutil
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "https://smkn1tegalsari.sch.id",
    "http://localhost:8000",
    "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
DB_DIR = "./db"
VECTOR_STORE_PATH = os.path.join(DB_DIR, "combined_vectorstore")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

class AskRequest(BaseModel):
    query: str


class DeleteRequest(BaseModel):
    filename: str

# Fungsi: rebuild ulang FAISS setelah penghapusan
def rebuild_faiss_index():
    embeddings = OpenAIEmbeddings()
    all_chunks = []

    for fname in os.listdir(UPLOAD_DIR):
        if fname.endswith(".pdf"):
            pdf_path = os.path.join(UPLOAD_DIR, fname)
            try:
                pdf_reader = PdfReader(pdf_path)
                text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    # tambahkan metadata filename
                    all_chunks.append((chunk, {"source": fname}))
            except Exception as e:
                print(f"Gagal memproses {fname}: {e}")

    if all_chunks:
        texts, metadatas = zip(*all_chunks)
        vectordb = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vectordb.save_local(VECTOR_STORE_PATH)
    else:
        # Hapus FAISS index jika semua dokumen dihapus
        for file in ["index.faiss", "index.pkl"]:
            fpath = os.path.join(VECTOR_STORE_PATH, file)
            if os.path.exists(fpath):
                os.remove(fpath)

@app.post("/upload")
async def upload_pdf(file: UploadFile):
    filename = file.filename
    pdf_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except PermissionError:
        return JSONResponse(status_code=500, content={"error": "Permission denied saat menyimpan file. Pastikan folder './uploads' punya izin tulis."})

    pdf_reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()

    if os.path.exists(VECTOR_STORE_PATH):
        vectordb = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        vectordb.add_texts(chunks)
    else:
        vectordb = FAISS.from_texts(chunks, embeddings)

    vectordb.save_local(VECTOR_STORE_PATH)

    return JSONResponse(content={"message": f"Berhasil diunggah & diproses: {filename}"})


@app.post("/ask")
async def ask_question(request: AskRequest):
    try:
        if not os.path.exists(VECTOR_STORE_PATH):
            return JSONResponse(status_code=400, content={"error": "Belum ada knowledge base diunggah."})

        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

        docs = vectordb.similarity_search(request.query)
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=request.query)

        # Fallback ke WhatsApp jika jawaban terlalu pendek atau tidak informatif
        cleaned_response = response.strip().lower()
        fallback_responses = [
            "saya tidak tahu", 
            "maaf, saya tidak tahu", 
            "tidak ditemukan", 
            "saya tidak bisa menjawab itu"
        ]

        if cleaned_response in fallback_responses or len(cleaned_response) < 15:
            return JSONResponse(content={
                "answer": "Maaf, saya tidak menemukan informasi tersebut. Silakan hubungi admin melalui Email atau WhatsApp untuk bantuan lebih lanjut. Email : smkn1tegalsaribwi@gmail.com atau WhatsApp : 0838-5339-9200",
            })

        return JSONResponse(content={"answer": response})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Terjadi kesalahan: {str(e)}"})

@app.get("/list-files")
async def list_uploaded_files():
    try:
        files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
        return JSONResponse(content={"files": files})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- Hapus file PDF dari uploads ---
@app.post("/delete-file")
async def delete_file(request: DeleteRequest):
    filename = request.filename
    file_path = os.path.join(UPLOAD_DIR, filename)

    if os.path.exists(file_path):
        os.remove(file_path)

        # Rebuild FAISS tanpa dokumen yang dihapus
        rebuild_faiss_index()

        return JSONResponse(content={"success": True, "message": f"{filename} berhasil dihapus."})
    else:
        return JSONResponse(status_code=404, content={"success": False, "message": "File tidak ditemukan."})
