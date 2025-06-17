
import re
import requests
from langchain_community.vectorstores import FAISS  # type:ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type:ignore
from langchain_core.prompts import PromptTemplate   #type:ignore

# ==== CONFIG ====
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Adjust if different

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

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

def is_greeting(query):
    return re.match(r'^\s*(hi|hello|hey|howdy|namaste|hola)\s*$', query.strip(), re.IGNORECASE)

def call_groq_llm(prompt_text: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.3-70b-versatile",  # Replace with your desired GROQ model
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
    print('-------------',context)

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

if __name__ == "__main__":
    try:
        while True:
            user_query = input("Write query here (or type 'exit' to quit): ")
            if user_query.lower() == "exit":
                print("Goodbye!")
                break
            response = custom_qa_chain(user_query)

            print("\nRESULT:\n", response["result"])

            if response["source_documents"]:
                print("\nSOURCE DOCUMENTS:")
                for i, doc in enumerate(response["source_documents"], 1):
                    snippet = doc.page_content[:500].replace('\n', ' ') + "..."
                    print(f"\nDocument {i} snippet:\n{snippet}")

    except Exception as e:
        print(f"An error occurred: {e}")