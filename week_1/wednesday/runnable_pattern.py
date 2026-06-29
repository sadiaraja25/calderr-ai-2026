import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(os.path.expanduser("~/calderr-ai-2026/.env"))
api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=api_key
)

parser = StrOutputParser()


mock_db = {
    "france": "France has 68 million people. Capital: Paris",
    "germany": "Germany has 84 million people. Capital: Berlin",
    "python": "Python is a programming language created in 1991",
}

def get_context(query):
    query_lower = query.lower()
    for key, value in mock_db.items():
        if key in query_lower:
            print(f"Found context: '{value}'")
            return value
    return f"Sorry, no information found about '{query}'"

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided context to answer questions."),
    ("user", "Question: {question}\nContext: {context}")
])


chain = {
    "question": RunnablePassthrough(),
    "context": lambda x: get_context(x["question"])  # Extract question string
} | prompt | llm | parser


result = chain.invoke({"question": "Tell me about France"})
print(f"Answer: {result}")
