import os
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from tools import explain_image, web_search_with_images,wikipedia_search
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment variable
openai_api_key = os.getenv("gpt")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=openai_api_key, verbose=True)
tools = [explain_image, web_search_with_images,wikipedia_search]
chat_with_tools = llm.bind_tools(tools)

 # Add a system prompt to every conversation
system_prompt = SystemMessage(content="""
                            You are a general AI assistant. I will ask you a question.
                            Think step-by-step before answering. Double-check your reasoning before producing the final response.
                            and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
                            YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
                            If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
                            If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
                            If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
""")

def build_graph():
    # Generate the AgentState and Agent graph
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    def assistant(state: AgentState):
        # Ensure system_prompt is the first message
        messages = state["messages"]
        if not messages or not (isinstance(messages[0], SystemMessage) and messages[0].content == system_prompt.content):
            messages = [system_prompt] + messages
        return {
            "messages": [chat_with_tools.invoke(messages)],
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
    return builder.compile()