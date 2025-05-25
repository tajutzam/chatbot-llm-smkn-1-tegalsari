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
    allow_origins=["https://smkn1tegalsari.sch.id"], 
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

@app.post("/upload")
async def upload_pdf(file: UploadFile):
    filename = file.filename
    pdf_path = os.path.join(UPLOAD_DIR, filename)

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pdf_reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()

    if os.path.exists(VECTOR_STORE_PATH):
        vectordb = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
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

        return JSONResponse(content={"answer": response})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Terjadi kesalahan: {str(e)}"})