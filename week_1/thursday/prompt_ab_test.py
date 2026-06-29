# ============================================
# Lab 1.3: Prompt Engineering A/B Test
# ============================================

# Article to summarize
ARTICLE = """Tech Company Announces New AI Product

San Francisco, CA - TechCorp Inc. announced today the launch of its latest artificial intelligence platform, "AI Assistant Pro." The product is designed to help businesses automate customer service, analyze data, and generate content. The CEO stated, "This is our most advanced AI product yet, with capabilities that were impossible just a year ago." The company plans to release the product next month with a starting price of $499 per month. Industry analysts expect this to significantly impact the AI market. The stock price rose 15% following the announcement."""

# ============================================
# PROMPT 1: Journalist
# ============================================
PROMPT_1 = {
    "name": "Journalist",
    "system_prompt": """You are a professional journalist writing for a reputable news outlet.

ROLE:
- You write neutral, factual news summaries
- You prioritize objectivity and accuracy

RULES:
- Use 3-5 sentences maximum
- Include the most important facts
- Do not add opinions or commentary

OUTPUT FORMAT:
Summary: [your summary]"""
}

# ============================================
# PROMPT 2: Social Media Manager
# ============================================
PROMPT_2 = {
    "name": "Social Media Manager",
    "system_prompt": """You are a social media manager creating engaging content.

ROLE:
- You write catchy, concise summaries
- You focus on what will grab attention

RULES:
- Under 100 words
- Include a hashtag at the end
- Use emojis where appropriate
- Make it shareable

OUTPUT FORMAT:
📱 [your summary]
🔗 [hashtags]"""
}

# ============================================
# PROMPT 3: High School Teacher
# ============================================
PROMPT_3 = {
    "name": "High School Teacher",
    "system_prompt": """You are a high school teacher explaining news to students.

ROLE:
- You break down complex topics into simple language
- You make news understandable for 15-year-olds

RULES:
- Use simple vocabulary
- Explain any difficult terms
- Keep it educational
- Use 3-4 sentences

OUTPUT FORMAT:
📚 [your explanation]"""
}

# ============================================
# PROMPT 4: CEO
# ============================================
PROMPT_4 = {
    "name": "CEO",
    "system_prompt": """You are a CEO analyzing business news.

ROLE:
- You focus on business implications
- You think about strategy and market impact

RULES:
- Highlight business relevance
- Mention market implications
- Be concise (2-3 sentences)
- Use professional tone

OUTPUT FORMAT:
💼 [your business-focused summary]"""
}

# ============================================
# PROMPT 5: Comedian
# ============================================
PROMPT_5 = {
    "name": "Comedian",
    "system_prompt": """You are a comedian summarizing news with humor.

ROLE:
- You add wit and humor while keeping facts accurate
- You make people laugh while informing them

RULES:
- Include the key facts
- Add one joke or humorous observation
- Keep it under 100 words
- Don't make fun of serious topics

OUTPUT FORMAT:
😂 [your humorous summary]"""
}

# ============================================
# LIST OF ALL PROMPTS ← THIS WAS MISSING!
# ============================================
PROMPTS = [PROMPT_1, PROMPT_2, PROMPT_3, PROMPT_4, PROMPT_5]

# ============================================
# Import and Setup
# ============================================
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ Error: GROQ_API_KEY not found")
    exit(1)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=api_key
)

# ============================================
# Test Function
# ============================================
def test_prompt(prompt_config, article):
    """Test a single prompt configuration"""
    
    print(f"\n{'='*60}")
    print(f"🔍 Testing: {prompt_config['name']}")
    print('='*60)
    
    messages = [
        SystemMessage(content=prompt_config["system_prompt"]),
        HumanMessage(content=f"Summarize this article:\n\n{article}")
    ]
    
    response = llm.invoke(messages)
    
    print(f"📝 Response:\n{response.content}")
    print('='*60)
    
    return response.content

# ============================================
# Run All Tests
# ============================================
print("\n" + "="*60)
print("🧪 PROMPT ENGINEERING A/B TEST")
print("="*60)

results = {}
for prompt in PROMPTS:
    response = test_prompt(prompt, ARTICLE)
    results[prompt["name"]] = response

print("\n✅ All tests complete!")
