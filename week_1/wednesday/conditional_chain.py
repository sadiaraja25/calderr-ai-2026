from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=api_key
)

parser = StrOutputParser()

math_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math expert. Calculate the result."),
    ("user", "{question}")
])

search_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a knowledge assistant. Answer based on facts."),
    ("user", "{question}")
])

default_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])
math_chain = math_prompt | llm | parser
search_chain = search_prompt | llm | parser
default_chain = default_prompt | llm | parser

def is_math(input_dict):
    query = input_dict.get("question", "")
    return any(op in query for op in ["+", "-", "*", "/"])

def is_search(input_dict):
    query = input_dict.get("question", "").lower()
    keywords = ["france", "germany", "paris", "berlin"]
    return any(keyword in query for keyword in keywords)

conditional_chain = RunnableBranch(
    (is_math, math_chain),
    (is_search, search_chain),
    default_chain
)

result1 = conditional_chain.invoke({"question": "What is 5 + 3?"})
print(result1)

result2 = conditional_chain.invoke({"question": "Tell me about France"})
print(result2)

result3 = conditional_chain.invoke({"question": "What is the weather?"})
print(result3)
