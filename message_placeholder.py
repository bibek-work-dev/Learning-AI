from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagePlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import MessagePlaceHolder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

import os
from dotenv import load_dotenv
from typing import Any

load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

history = ChatMessageHistory()

# "Explain {topic} in simple terms. Return the response as JSON with keys: - definition(max 30 words)- example(max 20 words)"

# Concept of Prompt Template
explain_prompt = ChatPromptTemplate.from_template([
    ("system","Explain topics in simple terms. Return JSON with 'definition' and 'example'."),
    MessagePlaceHolder(variable_name="history"),
    ("human", "{topic}")
]

    )

quiz_prompt = ChatPromptTemplate.from_template("""
Create a one-sentence quiz question from this definition:
"{definition}"
Return as JSON with key:
- quiz_question
""")

print("type the topic what would you like to explained")
x = input()

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  
    google_api_key=gemini_api_key,  
    temperature=0.2, 
    max_tokens=70 
)

parser = JsonOutputParser()

# Without parser, the type of output is AImessage, Not a stirng, it contains, content, metadata and things like that
# That parser does take the AIMessage and return content as string 


chain = (
    explain_prompt 
    | gemini 
    | parser 
    | RunnablePassthrough.assign(
        quiz = quiz_prompt | gemini | parser
    )
)

# --- 3. THE STATE (The "Memory Bank") ---
# In a real app, you would save this to a list or database
memory_bank = [
    HumanMessage(content="Explain Loops"),
    AIMessage(content='{"definition": "A sequence of instructions that is repeated...", "example": "Running around a track"}')
]

try:
    output = chain.invoke({ 
        "topic": "Explain Photosynthesis", 
        "history": memory_bank 
    })
    print(output)
except Exception as e:
    print(f"Still hitting an error: {e}")