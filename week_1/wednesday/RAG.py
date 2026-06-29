import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# ============================================
# 1. LOAD DOCUMENT
# ============================================
loader = TextLoader("document.txt")
docs = loader.load()
print(f"✅ Loaded document: {len(docs)} documents")

# ============================================
# 2. SPLIT INTO CHUNKS
# ============================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(docs)
print(f"✅ Split into {len(chunks)} chunks")

# ============================================
# 3. SETUP CHROMADB
# ============================================
chroma_client = chromadb.PersistentClient(path="./my_chromadb")

hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="my_documents",
    embedding_function=hf_ef
)

# ============================================
# 4. ADD CHUNKS TO CHROMADB
# ============================================
# ✅ FIXED: Use actual chunks instead of hardcoded text
collection.add(
    documents=[chunk.page_content for chunk in chunks],
    metadatas=[chunk.metadata for chunk in chunks],
    ids=[f"id_{i}" for i in range(len(chunks))]
)

print(f"✅ Added {len(chunks)} chunks to ChromaDB")

# ============================================
# 5. TEST RETRIEVAL
# ============================================
print("\n🔍 Testing retrieval...")
query = "What is the capital of France?"
results = collection.query(
    query_texts=[query],
    n_results=2
)

print(f"Query: {query}")
print(f"Found {len(results['documents'][0])} relevant chunks:")
for i, doc in enumerate(results['documents'][0]):
    print(f"  {i+1}. {doc[:150]}...")

# ============================================
# 6. (OPTIONAL) GENERATE ANSWER WITH GROQ
# ============================================
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=api_key
)

# Build context from retrieved chunks
context = "\n\n".join(results['documents'][0])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the following context to answer questions."),
    ("user", "Context: {context}\n\nQuestion: {question}")
])

chain = prompt | llm | StrOutputParser()

answer = chain.invoke({
    "context": context,
    "question": query
})

print(f"\n🤖 Final Answer: {answer}")
