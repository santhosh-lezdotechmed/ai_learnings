from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatAnthropic
from langchain.tools import DuckDuckGoSearchResults
from tavily import TavilyClient

# Manually set the API keys
tavily_api_key = "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"
claude_api_key = "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"

# Initialize Tavily client
tavily_client = TavilyClient(api_key=tavily_api_key)

# Initialize Claude using ChatAnthropic
claude_llm = ChatAnthropic(anthropic_api_key=claude_api_key)

# DuckDuckGo search setup
ddg = DuckDuckGoSearchResults()

def estimate_token_count(text):
    return len(text) // 4  # Rough estimate based on average token size
# Define tool functions
def query_tavily(query):
    """Query Tavily for direct answers."""
    try:
        response = tavily_client.search(query, include_answer=True)
        return response.get("answer", "No answer found.")
    except Exception as e:
        return f"Tavily error: {e}"

def query_ddg(query):
    """Search DuckDuckGo for web results."""
    try:
        results = ddg.run(query, max_results=5)
        return [result['href'] for result in results] if results else "No results found."
    except Exception as e:
        return f"DuckDuckGo error: {e}"

# Define tools with descriptions
tools = [
    Tool(
        name="Tavily",
        func=query_tavily,
        description="Fetch answers from Tavily's knowledge base."
    ),
    Tool(
        name="DuckDuckGo",
        func=query_ddg,
        description="Search the web using DuckDuckGo."
    ),
]

# Initialize the agent with tools and a compatible LLM
agent = initialize_agent(
    tools=tools,
    llm=claude_llm,  # Use ChatAnthropic as the LLM
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Example query to run with the agent
query = "What is the weather forecast for Salem?"
result = agent.run(query)

# Print the result
print(f"Agent Response:\n{result}")
