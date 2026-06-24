import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
load_dotenv(os.path.expanduser("~/calderr-ai-2026/.env"))
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ Error: GROQ_API_KEY not found in .env file")
    exit(1)
llm = ChatGroq(
	model="llama-3.1-8b-instant",
	temperature=0.7,
	api_key=api_key
)
message = HumanMessage(content="what is an AI Agent?")

response = llm.invoke([message])
print(response.content)
