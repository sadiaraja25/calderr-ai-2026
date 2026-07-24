import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================
# LOAD .env FROM SCRIPT DIRECTORY
# ============================================
# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent
env_path = script_dir / ".env"

# Load .env file
load_dotenv(env_path)

# Check if API key is loaded
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ Error: GROQ_API_KEY not found in .env file!")
    print(f"Looking for .env at: {env_path}")
    exit(1)

print(f"✅ API Key loaded: {api_key[:15]}...")

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=api_key
)
parser = StrOutputParser()

# ============================================
# 10 PROMPT REWRITE TECHNIQUES
# ============================================

original_task = "Summarize this article: AI is transforming healthcare by improving diagnostics, personalizing treatment, and reducing costs. Machine learning algorithms can detect diseases earlier than humans."

print("=" * 60)
print("📝 10 PROMPT REWRITE TECHNIQUES")
print("=" * 60)

# 1. Add Role
print("\n1️⃣ ROLE-BASED:")
prompt1 = "You are a medical journalist. Summarize this article for a healthcare professional: " + original_task
print(prompt1)

# 2. Add Constraints
print("\n2️⃣ CONSTRAINED:")
prompt2 = "Summarize this article in exactly 3 sentences, using only facts from the text: " + original_task
print(prompt2)

# 3. Add Format
print("\n3️⃣ FORMATTED:")
prompt3 = f"Summarize this article using this format:\n- Key Point 1: ...\n- Key Point 2: ...\n- Key Point 3: ...\n\nArticle: {original_task}"
print(prompt3)

# 4. Add Examples (Few-shot)
print("\n4️⃣ FEW-SHOT:")
prompt4 = """Examples of good summaries:
Example 1: "AI improves healthcare through better diagnosis and treatment."
Example 2: "Machine learning enables earlier disease detection."

Now summarize this article in the same style: """ + original_task
print(prompt4)

# 5. Add Reasoning (CoT)
print("\n5️⃣ CHAIN-OF-THOUGHT:")
prompt5 = f"Let's think step by step about this article:\n1. What are the main topics?\n2. What are the key claims?\n3. How are they connected?\n\nArticle: {original_task}\n\nNow provide a summary:"
print(prompt5)

# 6. Simplify Language
print("\n6️⃣ SIMPLIFIED:")
prompt6 = "Explain this article to a 10-year-old using simple words: " + original_task
print(prompt6)

# 7. Add Context
print("\n7️⃣ CONTEXTUAL:")
prompt7 = "Given that the reader is a hospital administrator, summarize this article highlighting cost-saving implications: " + original_task
print(prompt7)

# 8. Add Specificity
print("\n8️⃣ SPECIFIC:")
prompt8 = f"Summarize this article in exactly 50 words, focusing only on the diagnostic applications: " + original_task
print(prompt8)

# 9. Add Tone
print("\n9️⃣ TONE-BASED:")
prompt9 = "Write an enthusiastic and optimistic summary of this article: " + original_task
print(prompt9)

# 10. Add Error Handling
print("\n🔟 ERROR-HANDLING:")
prompt10 = f"Summarize this article. If you don't have enough information, say 'I need more details.' Article: {original_task}"
print(prompt10)

# ============================================
# EVALUATE REWRITES (Sample)
# ============================================
print("\n" + "=" * 60)
print("🔄 EVALUATING REWRITES")
print("=" * 60)

def evaluate_prompt(prompt, method_name):
    print(f"\n📊 {method_name}:")
    try:
        response = llm.invoke(prompt)
        print(response.content[:200] + "...")
    except Exception as e:
        print(f"Error: {e}")

# Test 3 prompts as example
evaluate_prompt(prompt1, "Role-Based")
evaluate_prompt(prompt5, "CoT")
evaluate_prompt(prompt8, "Specific")
