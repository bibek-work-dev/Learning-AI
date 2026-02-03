import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Setup the "Brain" and "Memory"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 2. Prepare the Data (The Knowledge)
policy_text = """
The employee vacation policy at TechCorp allows for 20 days of PTO per year. 
However, for employees based in the UK, this is increased to 25 days. 
All requests must be submitted 2 weeks in advance via the 'Portal-X' system.
"""
doc = Document(page_content=policy_text)

# 3. Store the Data
vectorstore = InMemoryVectorStore.from_documents([doc], embeddings)
retriever = vectorstore.as_retriever()

# 4. The Helper: Turn retrieved docs into a clean string for the LLM
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. The Prompt Template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. The LCEL Chain
# This is a 'Map'. It creates a dictionary with 'context' and 'question' 
# keys to satisfy the prompt template's requirements.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Execution
query = "I'm a UK employee, how many days off do I get and where do I request them?"
response = rag_chain.invoke(query)

print(f"User Query: {query}")
print("-" * 30)
print(f"AI Response: {response}")