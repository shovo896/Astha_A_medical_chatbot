from flask import Flask, render_template, request, jsonify
from src.healper import dowmnload_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import prompt
from dotenv import load_dotenv
import os
from src.prompt import * 

app = Flask(__name__)


app.route("/")
def index():
    return render_template("chat.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    
    
    
