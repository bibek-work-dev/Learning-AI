from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import json

from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv
from typing import Any, Union, Optional, List, Dict, Tuple, Set, Callable, Literal, TypedDict, Generic, TypeVar, Protocol

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

@tool
def get_weather(location: str):
    """Returns the current weather for a given city."""
    print("location" + location)
    return f"The weather in {location} is 25°C and sunny."

# class TopicExplainer(BaseModel):
#     definition: str = Field(description="Definition")
#     example: str = Field(description="An example of the topic")

# explain_prompt = ChatPromptTemplate.from_template("Explain {topic} in simple terms.")
                                                  
# x = input("What topic do you like to be explained ? ")

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  
    google_api_key=api_key,
    temperature=0.1,
    max_output_tokens=250,
    # max_token  // it is legacy 
    # system_instruction // it is also legacy
)

gemini_with_tools = gemini.bind_tools([get_weather])

query = "What is the weather in Kathmandu ? "
messages: Any = [HumanMessage(content=query)]
ai_msg = gemini_with_tools.invoke(messages)
messages.append(ai_msg)

print(ai_msg)

# if(ai_msg.tool_calls):
#     print(f"AI decided to call the tools : {ai_msg.tool_calls[0]["name"]}")
#     print(f"Arguments: {ai_msg.tool_calls[0]['args']}")
#     tool_output = get_weather.invoke(ai_msg.tool_calls[0]['args'])
#     print(f"Tool Output: {tool_output}")
# else:
#     print(f"AI Response: {ai_msg.content}")

if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        # Run your actual function
        result = get_weather.invoke(tool_call["args"])
        
        # Create a ToolMessage. 
        # IMPORTANT: 'tool_call_id' must match the one from Gemini!
        tool_msg = ToolMessage(
            content=str(result), 
            tool_call_id=tool_call["id"]
        )
        messages.append(tool_msg)

# 4. Final call to Gemini with the FULL history
final_response = gemini_with_tools.invoke(messages)
print("final response" , final_response.content)



# structured_response = gemini.with_structured_output(TopicExplainer)

# chain = explain_prompt | structured_response 


# result = chain.invoke({
#     "topic": x,
# })


# print(type(result))
# print(result)


# Todo
# gemini.with_structured_output(TopicExplainer). for replacement of pydantic to get json everytime request
# Batching
