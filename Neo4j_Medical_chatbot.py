from neo4j import GraphDatabase
from anthropic import Client
from tavily import TavilyClient
from langchain.tools import DuckDuckGoSearchResults
import time

# Setup Neo4j Connection
class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        """Execute a Cypher query with a timeout."""
        parameters = parameters or {}
        with self.driver.session() as session:
            try:
                print(f"Executing Cypher query: {query} with parameters: {parameters}")  # Debug print
                return session.run(query, parameters).data()
            except Exception as e:
                print(f"Error running query: {e}")  # Detailed error print
                return None

# Setup Claude 2 API
claude_client = Client(api_key="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")

# Setup Tavily API
tavily_client = TavilyClient(api_key="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")

# Setup DuckDuckGo
duckduckgo_client = DuckDuckGoSearchResults()

# Initialize Neo4j
neo4j_handler = Neo4jHandler(
    uri="neo4j://3.89.217.152:7687",
    user="neo4j",
    password="administrations-paragraph-leakage"
)

# Log Interactions in Neo4j with Debugging
# Log Interactions in Neo4j with Debugging
def log_interaction(user_query, response):
    """Log the interaction in Neo4j for auditing or learning."""
    cypher_query = (
        "CREATE (log:Interaction {query: $query, response: $response, timestamp: $timestamp}) "
        "RETURN log"
    )
    parameters = {
        "query": user_query,
        "response": response,
        "timestamp": time.time(),
    }

    # Print query and parameters for debugging
    print(f"Logging interaction with query: {user_query}, response: {response}")  # Debug print

    try:
        with neo4j_handler.driver.session() as session:
            # Start a transaction to explicitly commit the data
            session.write_transaction(lambda tx: tx.run(cypher_query, parameters))
            print("Interaction logged successfully!")
    except Exception as e:
        print(f"Error logging interaction: {e}")





# Main Chatbot Logic
def medical_chatbot(user_query):
    # Step 1: Query Tavily
    print("Querying Tavily...")
    tavily_result = tavily_client.search(user_query, include_answer=True)
    if tavily_result.get("answer"):
        response = tavily_result["answer"]
        log_interaction(user_query, response)  # Log the interaction
        return {"source": "Tavily", "response": response}

    # Step 2: Query DuckDuckGo
    print("Querying DuckDuckGo...")
    ddg_results = duckduckgo_client.run(user_query, max_results=3)
    if ddg_results:
        url = ddg_results[0]['href']
        response = f"Check this link: {url}"
        log_interaction(user_query, response)  # Log the interaction
        return {"source": "DuckDuckGo", "response": response}

    # Step 3: Query Claude 2
    print("Querying Claude 2...")
    try:
        claude_response = claude_client.completions.create(
            model="claude-2",
            prompt=f"Provide a detailed medical explanation for: {user_query}",
            max_tokens=200
        )
        response = claude_response['completion']
        log_interaction(user_query, response)  # Log the interaction
        return {"source": "Claude", "response": response}
    except Exception as e:
        response = str(e)
        log_interaction(user_query, response)  # Log the interaction
        return {"source": "Error", "response": response}

# Example Interaction
if __name__ == "__main__":
    user_input = input("Ask your medical question: ")
    result = medical_chatbot(user_input)
    print(f"Source: {result['source']}\nResponse: {result['response']}")

    # Provide Cypher query for Neo4j browser to inspect chat history
    print("\nYou can view the stored interactions in Neo4j using the following query:")
    print('''
    MATCH (log:Interaction)
    RETURN log.query AS user_query, log.response AS bot_response, log.timestamp AS timestamp
    ORDER BY log.timestamp DESC
    LIMIT 10
    ''')

# Close Neo4j Connection
neo4j_handler.close()
