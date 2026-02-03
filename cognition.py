from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from typing import Any
from pydantic import BaseModel, Field
import pymongo

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
mongo_url = os.getenv("MONGO_URL")

mongo_client = pymongo.MongoClient(mongo_url)

mydb = mongo_client["grocery"]

class FinalResult(BaseModel):
    total_amount: int = Field(description="This is total amount")
    currency: str = Field(description="This is currency")

@tool 
def find_sum(a: int, b: int) -> int:
    """Function to add two numbers"""
    return a + b

@tool
def find_the_price_of_grocery(a: str) -> Any:
    """Function to fetch the data from mongoDb to know the price"""
    print("a ", a)

    collection = mongo_client["test"]["grocery"]

    document = collection.find_one({"name": a})
    print("document : ", document)
    if not document:
        return f"Error: {a} Not found in the database"
    return document

@tool
def find_multiplication(multlipicand: int, multiplier: int) -> int:
    """Function to multiply"""
    return multiplier * multlipicand

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  
    google_api_key=api_key,
    temperature=0.1,
    max_output_tokens=180,
)

gemini_with_tools = gemini.bind_tools([find_sum, find_the_price_of_grocery, find_multiplication])

system_instruction = SystemMessage( "You are a helpful assistant. Use tools when necessary, but answer general questions from your own knowledge if no tool is applicable. and All grocery prices MUST be fetched using the provided tools(just provide the argument as grocery name only it will give you a price).")

x = input("What did you bought ? ")
# query = "I spent $45 on groceries and $120 on a new jacket. How much did I spend in total?"
query =  HumanMessage(content= x)

print("query ", query)

messages: Any = [system_instruction, query]

ai_msg = gemini_with_tools.invoke(messages)
messages.append(ai_msg)
print(ai_msg)

TOOLS = {
    "find_sum": find_sum,
    "find_the_price_of_grocery": find_the_price_of_grocery,
    "find_multiplication": find_multiplication
}


if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        tool_func = TOOLS.get(tool_name)

        if not tool_func:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        result = tool_func.invoke(tool_args)

        messages.append(
            ToolMessage(content=str(result),tool_call_id=tool_id)
        )

    # structured_respose = gemini.with_structured_output(FinalResult)
    # final_response = structured_respose.invoke(messages)
    # print(final_response)
    final_response = gemini_with_tools.invoke(messages)
    print("final response : ", final_response)

else:
    print("no tools called")
    print(ai_msg)



