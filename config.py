import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "day22-langsmith-lab")
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

LANGSMITH_API_KEY = os.environ["LANGCHAIN_API_KEY"]
LANGSMITH_PROJECT = os.environ["LANGCHAIN_PROJECT"]

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL   = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL   = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def get_llm(**kwargs):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=kwargs.get("model", OPENAI_MODEL),
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=kwargs.get("temperature", 0),
        request_timeout=60,
    )


def get_embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        request_timeout=30,
    )


def check():
    print("Config loaded:")
    print(f"  LangSmith project : {LANGSMITH_PROJECT}")
    print(f"  OpenAI endpoint   : {OPENAI_BASE_URL}")
    print(f"  Default LLM model : {OPENAI_MODEL}")
    print(f"  Embedding model   : {EMBEDDING_MODEL}")
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not LANGSMITH_API_KEY:
        missing.append("LANGCHAIN_API_KEY")
    if missing:
        print(f"  WARNING: missing keys: {missing}")
    else:
        print("  All API keys present")


if __name__ == "__main__":
    check()
