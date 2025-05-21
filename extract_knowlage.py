from PyPDF2 import PdfReader
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_community.llms import OpenAI

# Load API key dari .env
load_dotenv()

# Argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path ke file PDF")
ap.add_argument("-o", "--output", required=True, help="Nama folder untuk menyimpan FAISS vectorstore")
args = vars(ap.parse_args())

# Baca file PDF
pdfreader = PdfReader(args['input'])
text = ""
for page in pdfreader.pages:
    if page.extract_text():
        text += page.extract_text()

# Split teks
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n"],
    chunk_size=1000,
    chunk_overlap=202,
    length_function=len
)
chunks = text_splitter.split_text(text)

# Generate embeddings
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_texts(chunks, embeddings)

# Simpan FAISS vectorstore
output_dir = os.path.join("db", args["output"])
vectordb.save_local(output_dir)

print(f"[SUKSES] Knowledge base disimpan di: {output_dir}")
