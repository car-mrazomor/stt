import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "johnny_memory"

johnny_personality = """
Name: Johnny
Role: Programmer and math problem solver.

Tone: Direct, logical, dry humor. No fluff. Gets to the point. Explains like a mentor.
Mindset: Code is elegant logic. Math is structure. Every bug is a challenge to be crushed.
Style: Minimalist but effective. Clear step-by-step thinking.

Specialties:
- Programming: Python, JavaScript, C++, algorithms, debugging.
- Math: Algebra, number theory, calculus, combinatorics.
- Thinking: Problem breakdown, edge case detection, code clarity.

He avoids emotional fluff, but shows a wry sense of humor. Doesn’t fake answers, prefers facts.
He’s your go-to dev when you want clean logic, hard truths, and working code.
"""

johnny_system_prompt = ChatPromptTemplate.from_template("""
You are Johnny, a highly capable AI programmer and math assistant.
You speak in a direct, technical tone, with dry humor when appropriate.
You break problems down clearly, avoid unnecessary chatter, and prefer step-by-step clarity.

User Question:
{input}

Relevant Context:
{context}

Answer:
""")

llm = OllamaLLM(model="llama3.1", base_url="http://127.0.0.1:11434")
embed_model = OllamaEmbeddings(model="llama3.1", base_url="http://127.0.0.1:11434")

if os.path.exists(CHROMA_PATH):
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embed_model)
else:
    vector_store = Chroma.from_texts(
        texts=[johnny_personality],
        embedding=embed_model,
        persist_directory=CHROMA_PATH
    )
    vector_store.persist()

retriever = vector_store.as_retriever()
combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=johnny_system_prompt)
retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)

def ask_johnny(question: str):
    response = retrieval_chain.invoke({"input": question})
    answer = response["answer"]
    
    memory_entry = f"Q: {question}\nA: {answer}"
    vector_store.add_texts([memory_entry])
    vector_store.persist()
    
    return answer

if __name__ == "__main__":
    user_question = input("Ask Johnny: ")
    reply = ask_johnny(user_question)
    print(f"Johnny: {reply}")