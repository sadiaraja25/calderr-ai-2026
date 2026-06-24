import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Disable LangSmith (optional)
os.environ["LANGCHAIN_TRACING"] = "false"

load_dotenv(os.path.expanduser("~/calderr-ai-2026/.env"))
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ Error: GROQ_API_KEY not found")
    exit(1)

# Test different temperatures
temperatures = [0.0, 1.0, 2.0]
question = "Write a short story about a robot learning to be human"

print("🌡️ TEMPERATURE EXPERIMENT")
print("=" * 50)

for temp in temperatures:
    print(f"\n🔥 Temperature: {temp}")
    print("-" * 30)
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temp,
        api_key=api_key
    )
    
    message = HumanMessage(content=question)
    response = llm.invoke([message])
    print(f"📝 Response: {response.content}")
    print("-" * 30)

print("\n✅ Experiment complete!")
