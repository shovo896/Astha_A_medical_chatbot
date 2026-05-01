from dotenv import load_dotenv
import os 
from src.helper import load_pdf_files, filter_to_minimal_docs, download_hiuggingface_embeddings,text_split 
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPEN_AI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_docs=load_pdf_files(data="data/")
filter_docs=filter_to_minimal_docs(extracted_docs)
text_chunks=text_split(filter_docs)
embeddings=download_hiuggingface_embeddings()

pinecone_api_key=os.getenv("PINECONE_API_KEY")
pc=Pinecone(api_key=pinecone_api_key, environment="us-west1-gcp")

index_name="medical-chatbot-index"