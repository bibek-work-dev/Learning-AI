from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv
from typing import Any

load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Concept of Prompt Template
explain_prompt = ChatPromptTemplate.from_template("""Explain {topic} in simple terms.
    Return the response as JSON with keys:
    - definition(max 30 words)
    - example(max 20 words)""")

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


try:
    output = chain.invoke({ "topic": x })
    print(output)
except Exception as e:
    print(f"Still hitting an error: {e}")