import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

os.environ["LANGCHAIN_TRACING"] = "false"

load_dotenv(os.path.expanduser("~/calderr-ai-2026/.env"))
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ Error: GROQ_API_KEY not found")
    exit(1)

# Models to test
models = [
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "llama-3.3-70b-versatile"
]

question = "Explain the difference between machine learning and deep learning in 3 sentences"

print("🚀 MODEL COMPARISON")
print("=" * 50)

for model_name in models:
    print(f"\n📌 Model: {model_name}")
    print("-" * 40)
    
    llm = ChatGroq(
        model=model_name,
        temperature=0.3,
        api_key=api_key
    )
    
    message = HumanMessage(content=question)
    
    start = time.time()
    response = llm.invoke([message])
    end = time.time()
    
    print(f"📝 Response: {response.content}")
    print(f"⏱️ Time: {end - start:.2f} seconds")
    print("-" * 40)

print("\n✅ Comparison complete!")
