from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
# Import the Google GenAI chat model instead of OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient   # This is used for creating custom tools


load_dotenv()


# Initialize the Tavily client
tavily_client = TavilyClient()

"""
Create a custom tool for searching the web using the Tavily API
"""
@tool
def search(query: str) -> str:
    """
    Tool that searches the web for information
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Search results for {query}")
    return tavily_client.search(query=query)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools = [search]

agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from agentic-ai-engineering-course!")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
    print(result)

if __name__ == "__main__":
    main()
