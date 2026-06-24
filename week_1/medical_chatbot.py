import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv(os.path.expanduser("~/calderr-ai-2026/.env"))
api_key = os.getenv("GROQ_API_KEY")

# Check if API key exists
if not api_key:
    print("❌ Error: GROQ_API_KEY not found in .env file")
    exit(1)

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=api_key
)

# System prompt to make it a medical assistant
system_prompt = SystemMessage(content="""
You are MediBot, a helpful medical assistant.

RULES:
1. ONLY answer health and medical questions
2. If asked about non-medical topics, say: "I only answer medical questions."
3. Always say: "I am an AI, not a doctor. Consult a real doctor for serious issues."
4. Keep answers short and clear.
""")

print("\n🏥 MediBot - Your Medical Assistant")
print("=" * 40)
print("Type your medical question. Type 'exit' to quit.")
print("=" * 40)
conversation_history = []
# Chat loop
while True:
    # Get user input
    user_input = input("\n❓ You: ")
    
    # Check if user wants to exit
    if user_input.lower() == "exit":
        print("\n👋 Goodbye! Stay healthy!")
        break
    
    # Skip empty input
    if not user_input:
        continue
    conversation_history.append(HumanMessage(content=user_input))
    # Create the messages
    messages = [system_prompt] + conversation_history[-10:]
    
    # Get response from Groq
    response = llm.invoke(messages)
    print(f"\n🤖 MediBot: {response.content}")
