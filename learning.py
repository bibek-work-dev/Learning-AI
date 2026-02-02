from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from typing import Any
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

class FinalResult(BaseModel):
    total_amount: int = Field(description="This is total amount")
    currency: str = Field(description="This is currency")

@tool 
def find_sum(a: int, b: int) -> int:
    """Function to add two numbers"""
    return a + b

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  
    google_api_key=api_key,
    temperature=0.1,
    max_output_tokens=250,
)

gemini_with_tools = gemini.bind_tools([find_sum])

query = "I spent $45 on groceries and $120 on a new jacket. How much did I spend in total?"

messages: Any = [HumanMessage(content=query)]

ai_msg = gemini_with_tools.invoke(query)
messages.append(ai_msg)
print(ai_msg)

TOOLS = {
    "find_sum": find_sum
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

structured_respose = gemini.with_structured_output(FinalResult)

final_response = gemini.invoke(messages)
print(final_response)

