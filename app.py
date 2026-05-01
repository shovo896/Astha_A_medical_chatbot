import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import download_embeddings
from src.prompt import prompt


load_dotenv()

app = Flask(__name__)

INDEX_NAME = "medical-chatbot-index"
_rag_chain = None


def get_rag_chain():
    global _rag_chain

    if _rag_chain is None:
        if not os.getenv("PINECONE_API_KEY"):
            raise RuntimeError("PINECONE_API_KEY is missing from environment/.env")

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing from environment/.env")

        embeddings = download_embeddings()
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
        )
        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
        _rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return _rag_chain


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_message = (
        request.form.get("msg")
        or request.form.get("message")
        or (request.get_json(silent=True) or {}).get("message")
        or (request.get_json(silent=True) or {}).get("input")
    )

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        response = get_rag_chain().invoke({"input": user_message})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"answer": response.get("answer", "")})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
