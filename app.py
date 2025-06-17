import streamlit as st
import re
import requests
from langchain_community.vectorstores import FAISS  # type:ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type:ignore
from langchain_core.prompts import PromptTemplate   #type:ignore
import sys

# Prevent Streamlit from watching torch.classes (causes crash)
sys.modules['torch.classes'] = None
# ==== CONFIG ====
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get GROQ API Key from env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DB_FAISS_PATH = "vectorstore/db_faiss"

# ==== Embeddings ====
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# ==== Prompt ====
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't try to make up an answer.
Don't provide anything out of the given context. 
Context: {context}
Question: {question}
Start the answer directly, no small talk please.
"""
def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

# ==== Helper Functions ====
def is_greeting(query):
    return re.match(r'^\s*(hi|hello|hey|howdy|namaste|hola)\s*$', query.strip(), re.IGNORECASE)

def call_groq_llm(prompt_text: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",  # or use the correct model from GROQ
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.5,
        "max_tokens": 512
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"GROQ API Error {response.status_code}: {response.text}")

def generate_response(context: str, question: str) -> dict:
    prompt_template = set_custom_prompt()
    prompt = prompt_template.format(context=context, question=question)
    answer = call_groq_llm(prompt)
    return {"text": answer}

def custom_qa_chain(user_query: str) -> dict:
    if is_greeting(user_query):
        return {
            "result": "👋 Hello! How can I help you today?",
            "source_documents": []
        }
    
    docs = db.similarity_search(user_query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    if not context.strip():
        return {
            "result": "Sorry, I couldn't find relevant information in the documents.",
            "source_documents": []
        }

    llm_response = generate_response(context=context, question=user_query)
    return {
        "result": llm_response["text"],
        "source_documents": docs
    }

# ==== Streamlit UI ====
st.set_page_config(page_title="Medical Chatbot 💬", page_icon="🩺")

st.title("🧠 Medical AI Chatbot")
st.markdown("Ask any question related to your medical documents.")

# --- Manage state ---
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

def submit():
    st.session_state.submitted = True
    st.session_state.query = st.session_state.user_input
    st.session_state.user_input = ""  # Clear input after submit

# --- Input box ---
st.text_input(
    "Ask your question:",
    key="user_input",
    placeholder="e.g., What are the symptoms of hypertension?",
    on_change=submit
)

# --- Response Display ---
if st.session_state.get("submitted", False):
    with st.spinner("Searching and generating answer..."):
        response = custom_qa_chain(st.session_state.query)

    st.markdown("### 💡 Response:")
    st.success(response["result"])

    if response["source_documents"]:
        st.markdown("### 📚 Source Documents:")
        for i, doc in enumerate(response["source_documents"], 1):
            snippet = doc.page_content[:1000]
            st.markdown(f"**Document {i}:**")
            st.code(snippet + ("..." if len(doc.page_content) > 1000 else ""))

    st.session_state.submitted = False
