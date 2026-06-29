# ============================================
# parallel_chain.py - RunnableParallel Example
# ============================================

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# ============================================
# 1. SETUP
# ============================================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=api_key
)

parser = StrOutputParser()

# Sample text to analyze
sample_text = """
Artificial Intelligence is transforming the world. 
It helps doctors diagnose diseases more accurately, 
assists teachers in creating personalized lessons, 
and powers self-driving cars that could reduce accidents.
"""

# ============================================
# 2. CREATE THREE DIFFERENT PROMPTS
# ============================================

# Prompt 1: Summarization
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a summarization expert."),
    ("user", "Summarize the following text in 2 sentences:\n\n{text}")
])

# Prompt 2: Sentiment Analysis
sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment analysis expert."),
    ("user", "Analyze the sentiment of this text. Answer with ONLY one word: Positive, Negative, or Neutral:\n\n{text}")
])

# Prompt 3: Keyword Extraction
keywords_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a keyword extraction expert."),
    ("user", "Extract the main keywords from this text. Return them as a comma-separated list:\n\n{text}")
])

# ============================================
# 3. CREATE THREE CHAINS
# ============================================

summary_chain = summary_prompt | llm | parser
sentiment_chain = sentiment_prompt | llm | parser
keywords_chain = keywords_prompt | llm | parser

# ============================================
# 4. COMBINE WITH RUNNABLEPARALLEL
# ============================================

parallel_chain = RunnableParallel({
    "summary": summary_chain,
    "sentiment": sentiment_chain,
    "keywords": keywords_chain
})

# ============================================
# 5. RUN THE PARALLEL CHAIN
# ============================================

print("=" * 60)
print("🔀 RUNNING PARALLEL CHAIN")
print("=" * 60)

print("\n📝 Input Text:")
print("-" * 40)
print(sample_text.strip())
print("-" * 40)

print("\n⏳ Processing in parallel...\n")

# Invoke the parallel chain
result = parallel_chain.invoke({"text": sample_text})

# Display results
print("=" * 60)
print("📊 RESULTS (All processed simultaneously!)")
print("=" * 60)

print(f"\n📌 Summary:\n{result['summary']}")
print(f"\n💬 Sentiment: {result['sentiment']}")
print(f"\n🏷️ Keywords: {result['keywords']}")

print("\n" + "=" * 60)
print("✅ Parallel chain complete!")
print("=" * 60)
