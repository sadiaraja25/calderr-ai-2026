import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found!")
    print("Make sure .env file exists with: GROQ_API_KEY=gsk_...")
    exit(1)

print(f"✅ API Key loaded: {api_key[:15]}...")

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=api_key
)

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI engineering assistant."),
    ("user", "{question}")
])

# Build chain
chain = prompt | llm | StrOutputParser()

# Test it
question = "What is an AI agent and why does it matter?"
print(f"\n❓ Question: {question}\n")
print("🤖 Thinking...\n")

response = chain.invoke({"question": question})
print(f"💬 Response: {response}\n")

print("✅ First LLM call successful!")
