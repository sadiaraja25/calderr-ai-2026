import os
import json
from pathlib import Path
from typing import List, Optional, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator, model_validator

project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY not found!")
    exit(1)

print(f"✅ GROQ_API_KEY loaded: {GROQ_API_KEY[:10]}...")

class JobPosting(BaseModel):
    """Structured job posting model."""
    
    # Required fields
    title: str = Field(..., description="Job title/position name")
    company: str = Field(..., description="Company name")
    
    # Salary (optional)
    salary_min: Optional[int] = Field(None, description="Minimum annual salary in USD")
    salary_max: Optional[int] = Field(None, description="Maximum annual salary in USD")
    salary_currency: str = Field("USD", description="Salary currency")
    
    # Location
    location: str = Field(..., description="Job location (city, state, country)")
    remote: bool = Field(False, description="Is this position remote?")
    
    # Skills & Requirements
    required_skills: List[str] = Field(default_factory=list, description="Required technical skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred skills")
    
    # Additional fields
    experience_years: Optional[int] = Field(None, description="Years of experience required")
    education: Optional[str] = Field(None, description="Required education level")
    
    # Validators
    @field_validator('salary_min', 'salary_max')
    def validate_salary(cls, v):
        """Validate salary is positive."""
        if v is not None and v < 0:
            raise ValueError('Salary must be positive')
        return v
    
    @field_validator('title', 'company', 'location')
    def validate_not_empty(cls, v):
        """Ensure required fields are not empty."""
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()
    
    @model_validator(mode='after')
    def validate_salary_range(self):
        """Validate that salary_min <= salary_max."""
        if self.salary_min is not None and self.salary_max is not None:
            if self.salary_min > self.salary_max:
                raise ValueError('salary_min cannot be greater than salary_max')
        return self
    
    # Model configuration
    model_config = {
        'str_strip_whitespace': True,
        'extra': 'ignore',
        'json_schema_extra': {
            'examples': [{
                'title': 'Senior Software Engineer',
                'company': 'TechCorp',
                'salary_min': 120000,
                'salary_max': 160000,
                'location': 'San Francisco, CA',
                'remote': True,
                'required_skills': ['Python', 'AWS', 'PostgreSQL'],
                'experience_years': 5,
                'education': 'Bachelor\'s in Computer Science'
            }]
        }
    }

class ResearchPaper(BaseModel):
    """Structured research paper model."""
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of authors")
    abstract: str = Field(..., description="Paper abstract")
    keywords: List[str] = Field(default_factory=list, description="Key terms")
    year: int = Field(..., description="Publication year", ge=1900, le=2026)
    citations: Optional[int] = Field(None, description="Number of citations")

class ProductReview(BaseModel):
    """Structured product review model."""
    product_name: str = Field(..., description="Name of the product")
    rating: float = Field(..., description="Rating out of 5", ge=0.0, le=5.0)
    review_text: str = Field(..., description="Full review text")
    sentiment: str = Field(..., description="Sentiment: positive, negative, neutral")
    pros: List[str] = Field(default_factory=list, description="Positive aspects")
    cons: List[str] = Field(default_factory=list, description="Negative aspects")

class Event(BaseModel):
    """Structured event model."""
    name: str = Field(..., description="Event name")
    date: str = Field(..., description="Event date (YYYY-MM-DD)")
    location: str = Field(..., description="Event location")
    description: Optional[str] = Field(None, description="Event description")
    attendees: Optional[int] = Field(None, description="Expected attendees")
    category: str = Field(..., description="Event category")

class CustomerSupport(BaseModel):
    """Structured customer support ticket model."""
    ticket_id: str = Field(..., description="Ticket ID")
    issue_type: str = Field(..., description="Type of issue")
    summary: str = Field(..., description="Brief issue summary")
    priority: str = Field(..., description="Priority: low, medium, high, critical")
    status: str = Field("open", description="Status: open, in_progress, resolved")
    assigned_to: Optional[str] = Field(None, description="Assigned team member")
    created_at: str = Field(..., description="Creation timestamp")

def extract_job_posting(text: str) -> JobPosting:
    """Extract structured job posting from unstructured text."""
    
    # Initialize LLM with structured output
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        api_key=GROQ_API_KEY
    ).with_structured_output(JobPosting)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract job posting information from the text.
        Be thorough but only include information explicitly mentioned.
        If information is not available, leave fields as None or empty."""),
        ("user", "Text: {text}")
    ])
    
    chain = prompt | llm
    
    try:
        result = chain.invoke({"text": text})
        return result
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        # Return default model on error
        return JobPosting(title="Unknown", company="Unknown", location="Unknown")

def run_job_extraction_tests():
    """Test the job extractor with various inputs."""
    
    print("\n" + "=" * 60)
    print("📋 LAB 2.1: STRUCTURED OUTPUT EXTRACTOR")
    print("=" * 60)
    
    # Test case 1: Clean input
    test1 = """
    We're hiring a Senior Software Engineer at Google. 
    The position is based in San Francisco, CA and offers remote flexibility.
    Salary range: $140,000 - $180,000 per year.
    Requirements:
    - 5+ years of Python experience
    - Experience with AWS
    - PostgreSQL knowledge
    Bachelor's in CS or related field required.
    """
    
    print("\n📝 Test 1: Clean Job Posting")
    print("-" * 40)
    result1 = extract_job_posting(test1)
    print(json.dumps(result1.model_dump(), indent=2))
    
    # Test case 2: Messy input
    test2 = """
    Job: Data Scientist
    Company: Amazon
    Location: Seattle
    Pay: 130k - 160k per year
    Need SQL, Python, ML
    Remote? yes
    3+ years experience
    """
    
    print("\n📝 Test 2: Messy Job Posting")
    print("-" * 40)
    result2 = extract_job_posting(test2)
    print(json.dumps(result2.model_dump(), indent=2))
    
    # Test case 3: Sparse input
    test3 = "Hiring a product manager. San Francisco."
    
    print("\n📝 Test 3: Sparse Job Posting")
    print("-" * 40)
    result3 = extract_job_posting(test3)
    print(json.dumps(result3.model_dump(), indent=2))
    
    # Test case 4: With validation errors
    test4 = """
    Job: Software Developer
    Company: Microsoft
    Salary: -10000 to 50000
    Location: Redmond
    """
    
    print("\n📝 Test 4: Invalid Salary (Should catch validation error)")
    print("-" * 40)
    try:
        result4 = extract_job_posting(test4)
        print(json.dumps(result4.model_dump(), indent=2))
    except Exception as e:
        print(f"✅ Validation error caught: {e}")

if __name__ == "__main__":
    run_job_extraction_tests()
