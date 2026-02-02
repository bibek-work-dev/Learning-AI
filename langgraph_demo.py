from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class RouterState(TypedDict):
    input_number: int
    message: str
    category: str

def check_even_node(state: RouterState):
    print("--processing even number--")
    return {"category": "even number", "message": "this number is smooth and balanced"}

def check_odd_node(state: RouterState):
    print("--- Processing Odd Number ---")
    return {"category": "ODD", "message": "This number has a lonely remainder."}

def route_decision(state: RouterState):
    if state["input_number"] % 2 == 0:
        return "even_path"
    else:
        return "odd_path"
    
workflow = StateGraph(RouterState)

workflow.add_node("even_path", check_even_node)
workflow.add_node("odd_path", check_odd_node)

workflow.add_conditional_edges(
    START,
    route_decision,
    {
        "even_path": "even_path",
        "odd_path": "odd_path"
    }

)

workflow.add_edge("even_path", END)
workflow.add_edge("odd_path", END)

app = workflow.compile()

print("Testing with 10:")
print(app.invoke({"input_number": 10}))

print("\nTesting with 7:")
print(app.invoke({"input_number": 7}))