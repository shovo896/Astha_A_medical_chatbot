from dotenv import load_dotenv
import hashlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm

from src.helper import (
    download_embeddings,
    filter_to_minimal_docs,
    load_pdf_files,
)


def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )
    return text_splitter.create_documents(
        [doc.page_content for doc in minimal_docs],
        metadatas=[doc.metadata for doc in minimal_docs],
    )


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from environment/.env")

if OPEN_AI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPEN_AI_API_KEY

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_docs = load_pdf_files(data="data/")
filter_docs = filter_to_minimal_docs(extracted_docs)
text_chunks = text_split(filter_docs)
embeddings = download_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)
docsearch = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

batch_size = 100
ids = [
    f"doc-{i}-{hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()[:12]}"
    for i, doc in enumerate(text_chunks)
]

for start in tqdm(range(0, len(text_chunks), batch_size), desc="Uploading to Pinecone"):
    end = start + batch_size
    docsearch.add_documents(
        documents=text_chunks[start:end],
        ids=ids[start:end],
        batch_size=batch_size,
        embedding_chunk_size=batch_size,
    )

print(f"Uploaded/updated {len(text_chunks)} chunks in Pinecone index '{index_name}'.")
