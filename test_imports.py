print("Testing all imports...")

try:
    from langchain_groq import ChatGroq
    print("✅ langchain-groq")
except ImportError as e:
    print(f"❌ langchain-groq: {e}")

try:
    import chromadb
    print("✅ chromadb")
except ImportError as e:
    print(f"❌ chromadb: {e}")

try:
    from fastapi import FastAPI
    print("✅ fastapi")
except ImportError as e:
    print(f"❌ fastapi: {e}")

try:
    import groq
    print("✅ groq")
except ImportError as e:
    print(f"❌ groq: {e}")

try:
    import pydantic
    print("✅ pydantic")
except ImportError as e:
    print(f"❌ pydantic: {e}")

try:
    import uvicorn
    print("✅ uvicorn")
except ImportError as e:
    print(f"❌ uvicorn: {e}")

print("\n🎉 All checks complete!")
