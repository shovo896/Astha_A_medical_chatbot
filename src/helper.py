from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

def load_pdf_files(data):
    loader=DirectoryLoader(data, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")