"""
Step 1 — LangSmith-instrumented RAG Pipeline
=============================================
"""

import os
import time
from pathlib import Path

from config import get_llm, get_embeddings, LANGSMITH_PROJECT

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from qa_pairs import SAMPLE_QUESTIONS


def build_vectorstore():
    embeddings = get_embeddings()
    text = Path("data/knowledge_base.txt").read_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")

    print("Building FAISS index (embedding chunks)...")
    t0 = time.time()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print(f"FAISS index built in {time.time()-t0:.1f}s")
    return vectorstore


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer the question. "
               "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
               "Context:\n{context}"),
    ("human", "{question}"),
])


def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | get_llm()
        | StrOutputParser()
    )
    return chain, retriever


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    return chain.invoke(question)


def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    vectorstore = build_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    print(f"\nRunning {len(SAMPLE_QUESTIONS)} questions...\n")

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        t0 = time.time()
        try:
            answer = ask(chain, question)
            elapsed = time.time() - t0
            print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] ({elapsed:.1f}s) Q: {question[:60]}")
            print(f"       A: {answer[:100]}\n")
        except Exception as e:
            print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] ERROR: {e}\n")

    print(f"Done! {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{LANGSMITH_PROJECT}'")
    print("   Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()
