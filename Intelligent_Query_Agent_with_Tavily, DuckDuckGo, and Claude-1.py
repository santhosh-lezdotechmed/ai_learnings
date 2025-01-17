# # Required libraries
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import DuckDuckGoSearchResults
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
import json
import anthropic  # For Claude API

# Manually set the API key for Tavily
api_key = "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"

if not api_key:
    raise ValueError("API key is missing!")

# Initialize Tavily client
client = TavilyClient(api_key=api_key)

# DuckDuckGo search setup for LangChain tool
ddg = DuckDuckGoSearchResults()

def search_with_ddg(query, max_results=6):
    """Search the web using DuckDuckGo."""
    try:
        results = ddg.run(query, max_results=max_results)
        return [result['href'] for result in results]
    except Exception as e:
        print(f"DDG search error: {e}")
        return []

# Function to scrape weather information
def scrape_weather_info(url):
    """Scrape content from the given URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return "Failed to retrieve the webpage."
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except Exception as e:
        return f"Scraping error: {e}"

# Define tools for the LangChain agent
tools = [
    Tool(
        name="Tavily",
        func=lambda query: client.search(query, include_answer=True).get("answer", "No answer found"),
        description="Use Tavily to get direct answers to questions."
    ),
    Tool(
        name="DuckDuckGo",
        func=search_with_ddg,
        description="Use DuckDuckGo to search the web."
    ),
    Tool(
        name="WeatherScraper",
        func=scrape_weather_info,
        description="Scrape weather information from a URL."
    ),
]

# Initialize Claude client with the correct API key (no positional arguments)
claude_client = anthropic.Client(api_key="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")

# Define the agent function that will be used for reasoning
def agent_function(query):
    """Agent to handle search queries intelligently using Claude."""
    try:
        # Attempt to get a direct answer from Tavily
        tavily_result = client.search(query, include_answer=True)
        if tavily_result.get("answer"):
            return {"source": "Tavily", "answer": tavily_result["answer"]}
    except Exception as e:
        print("Tavily error:", e)
    
    # Fall back to DuckDuckGo search if no answer from Tavily
    print("Using DuckDuckGo for search...")
    search_results = search_with_ddg(query)
    if search_results:
        url = search_results[0]  # Use the first result
        soup = scrape_weather_info(url)
        if soup:
            # Extract and clean data from the page
            weather_data = []
            for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
                text = tag.get_text(" ", strip=True)
                weather_data.append(text)
            combined_weather_data = " ".join(weather_data)
            return {"source": url, "data": combined_weather_data}
    
    # If no results are found, call Claude to generate a response
    response = claude_client.completions.create(
        model="claude-1",  # Using Claude-1 as the language model
        prompt=query,
        max_tokens=100
    )
    
    return {"source": "Claude", "response": response['completion']}

# Example query
query = "when will be the next match for manchester united in premier league against to?"

# Run the agent
result = agent_function(query)
print(f"Final Result: {result}")

# from langchain.agents import initialize_agent, Tool, AgentType
# from langchain.chat_models import ChatAnthropic
# from langchain.tools import DuckDuckGoSearchResults
# from tavily import TavilyClient
# import requests
# from bs4 import BeautifulSoup

# # Weather Scraper
# def scrape_weather_info(url: str) -> str:
#     """Scrape weather information from a URL."""
#     try:
#         headers = {"User-Agent": "Mozilla/5.0"}
#         response = requests.get(url, headers=headers)
#         if response.status_code != 200:
#             return "Failed to retrieve weather data."
#         soup = BeautifulSoup(response.text, "html.parser")
#         weather_data = [tag.get_text(strip=True) for tag in soup.find_all(["h1", "h2", "h3", "p"])]
#         return " ".join(weather_data[:10])
#     except Exception as e:
#         return f"Scraping error: {e}"

# # Tavily Tool
# def tavily_search(api_key: str, query: str) -> str:
#     """Query Tavily for direct answers."""
#     try:
#         client = TavilyClient(api_key=api_key)
#         result = client.search(query, include_answer=True)
#         return result.get("answer", "No answer found.")
#     except Exception as e:
#         return f"Tavily error: {e}"

# # DuckDuckGo Search
# def ddg_search(query: str) -> list[str]:
#     """Search using DuckDuckGo and return relevant URLs."""
#     try:
#         ddg_tool = DuckDuckGoSearchResults()
#         results = ddg_tool.run(query, max_results=5)
#         return [result["href"] for result in results]
#     except Exception as e:
#         return f"Error with DuckDuckGo search: {e}"

# # API Keys
# anthropic_api_key ="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"  
# tavily_api_key = "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"        # Replace with your Tavily API key

# # Initialize Claude LLM using LangChain's built-in integration
# claude_llm = ChatAnthropic(
#     anthropic_api_key=anthropic_api_key,
#     model="claude-1",
#     max_tokens_to_sample=512,
# )

# # Initialize Tools
# weather_scraper_tool = Tool(
#     name="WeatherScraper",
#     func=scrape_weather_info,
#     description="Scrapes weather information from a URL."
# )
# tavily_tool = Tool(
#     name="Tavily",
#     func=lambda query: tavily_search(tavily_api_key, query),
#     description="Fetch direct answers using Tavily."
# )
# ddg_tool = Tool(
#     name="DuckDuckGo",
#     func=ddg_search,
#     description="Search for information using DuckDuckGo."
# )

# # Combine Tools
# tools = [weather_scraper_tool, tavily_tool, ddg_tool]

# # Initialize Agent
# agent = initialize_agent(
#     tools=tools,
#     llm=claude_llm,  # Use ChatAnthropic directly as the LLM
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )

# # Example Query
# query = "What is the weather forecast for New York?"
# result = agent.run(query)

# # Print the result
# print(f"Agent Response:\n{result}")








