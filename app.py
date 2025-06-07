import os
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from tools import extract_text_from_image, web_search_with_images
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini", api_key=openai_api_key, verbose=True)
tools = [extract_text_from_image, web_search_with_images]
chat_with_tools = llm.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }


## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()
 # Add a system prompt to every conversation
system_prompt = SystemMessage(content="""
You are an expert AI assistant designed to answer complex questions with accuracy, clarity, and depth. 
You have access to advanced tools, including web search and image analysis, 
to help you gather and synthesize information. Always reason step by step, 
cite your sources when possible, and ensure your answers are factual and well-explained. 
If you do not know the answer, state so honestly. 
Your goal is to provide responses that would pass rigorous evaluation for correctness and reasoning, 
such as the GAIA test.

[Calling tools]
IMPORTANT: Before calling any tool, you MUST first explain in plain text:
1. Which tool you are about to use
2. What arguments you are passing to it
3. Why you are using this tool

Example format:
"I will now use the web_searchweb_search_with_images tool with the argument 'Lady Ada Lovelace' to gather information about this person."
Tool Output:
continue conversation
[Calling tools]

""")


response = alfred.invoke({"messages": [system_prompt,HumanMessage(content="Which of the fruits shown in the 2008 painting “Embroidery from Uzbekistan” were served as part of the October 1949 breakfast menu for the ocean liner that was later used as a floating prop for the film “The Last Voyage”? Give the items as a comma-separated list, ordering them in clockwise order based on their arrangement in the painting starting from the 12 o’clock position. Use the plural form of each fruit.")]})

print("🎩 Alfred's Conversation:")
for i, msg in enumerate(response['messages']):
    role = getattr(msg, 'type', None) or msg.__class__.__name__
    print(f"[{i}] {role}: {getattr(msg, 'content', msg)}\n")

#alfred.get_graph(xray=True).draw_mermaid_png(output_file_path="rag.png")

