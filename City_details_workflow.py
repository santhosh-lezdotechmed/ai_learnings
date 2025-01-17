import warnings
warnings.filterwarnings('ignore')

import os
from cohere import Client
from transformers import pipeline
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain.llms import HuggingFacePipeline
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field

# Set your API keys (replace with your actual keys)
COHERE_API_KEY = "VAWmbnQRyn386DYaZKts0R8l8WGnHNJlNaqiMGZ0"
HUGGINGFACE_API_KEY = "hjl526KwinBvySEJWE7au8bC4llCaFt7G7THsYra"
TAVILY_API_KEY = "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"

# Set the TAVILY_API_KEY in environment variables
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

# Initialize Cohere Client
cohere_client = Client(api_key=COHERE_API_KEY)

# Initialize Hugging Face Model Pipeline
hf_pipeline = pipeline(
    "text-generation",
    model="bigscience/bloom-560m",  # Example Hugging Face model
    tokenizer="bigscience/bloom-560m",
    max_new_tokens=100,
    temperature=0.7,
    api_token=HUGGINGFACE_API_KEY
)

# Initialize Tavily search tool with API key
tavily_tool = TavilySearchResults(max_results=5, tavily_api_key=TAVILY_API_KEY)

# Function for generating text with Cohere
def cohere_generate(prompt):
    try:
        response = cohere_client.generate(
            model="command-xlarge-nightly",  # Replace with appropriate model
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"Error with Cohere: {str(e)}"

# Function for generating text with Hugging Face
def hf_generate(prompt):
    try:
        response = hf_pipeline(prompt)
        return response[0]["generated_text"].strip()
    except Exception as e:
        return f"Error with Hugging Face: {str(e)}"

# Define the tool for fetching city details
@tool
def get_city_details(prompt: str) -> str:
    """Fetch city details from Tavily and summarize."""
    try:
        # Use Tavily search
        search_results = tavily_tool.invoke(prompt)
        if not search_results:
            return "No relevant information found via Tavily."

        # Process summary using Hugging Face or Cohere
        summary = hf_generate(search_results[0]["snippet"])  # You can switch this to cohere_generate if needed
        return summary
    except Exception as e:
        return f"Error in fetching city details: {str(e)}"

# Define the CityDetails model (ensure it's defined before it's used)
class CityDetails(BaseModel):
    state_name: str = Field(description="State name of the city")
    state_capital: str = Field(description="State capital of the city")
    country_name: str = Field(description="Country name of the city")
    country_capital: str = Field(description="Country capital of the city")

# Bind tools to LLM
tools = [get_city_details]

# LLM placeholder (You can use a custom Hugging Face or Cohere model)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Workflow and graph setup
def call_model(state):
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

def respond(state):
    response = llm.invoke([HumanMessage(content=state['messages'][-2].content)])
    return {"final_response": response}

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "respond"
    else:
        return "continue"

# Define the workflow graph
workflow = StateGraph(CityDetails)

workflow.add_node("agent", call_model)
workflow.add_node("respond", respond)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "respond": "respond",
    },
)

workflow.add_edge("tools", "agent")
workflow.add_edge("respond", END)

# Compile the graph
graph = workflow.compile()

# Display the graph (if using an interactive environment like Jupyter Notebook)
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
