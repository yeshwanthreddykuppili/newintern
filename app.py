import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from flask import Flask, render_template, request, jsonify
from index import RAGEngine
from intent_router import is_claim_query

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Changed here: initialize rag as None
print("Initializing RAG engine...")
rag = RAGEngine("DATA/Claimsss.pdf")
print("RAG engine ready.")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    global rag
    try:
        user_message = request.json.get("message", "").strip()

        if not user_message:
            return jsonify({"response": "Please enter a valid question."})

        if is_claim_query(user_message):
            answer = rag.answer(user_message)
        else:
            answer = rag.answer(user_message)

        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})


# This API connects the frontend and backend
@app.route("/api/chat", methods=["POST"])
def api_chat_proxy():
    global rag
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"reply": "Please enter a valid question."})

        answer = rag.answer(user_message)

        return jsonify({"reply": answer})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":

    app.run(debug=True)