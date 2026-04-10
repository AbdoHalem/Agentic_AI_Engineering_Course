from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
# Import the Google GenAI chat model instead of OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

class Source(BaseModel):
    """
    Schema for a source used by the agent
    """
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """
    Schema for agent response with answer and sources
    """
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# ==============================================================================
# MODEL BEHAVIOR NOTE: GPT vs. Gemini Tool Calling
# LangChain's built-in tools were initially heavily optimized for OpenAI's API. 
# GPT models (e.g., gpt-3.5/4) are extensively fine-tuned for complex tool-calling 
# and can often infer when to use default LangChain tools out-of-the-box (zero-shot).
# 
# Conversely, extremely fast and lightweight models like 'gemini-2.5-flash' 
# might skip using complex tools if the schema is too broad or instructions 
# aren't absolutely direct. We instantiate 'search_tool' here with a very strict, 
# explicit 'description' to forcefully guide Gemini into recognizing and utilizing 
# the tool for real-time queries, bridging that behavioral gap.
# ==============================================================================
search_tool = TavilySearch(
    max_results=3,
    description="A web search tool. You MUST use this tool to answer questions about current events, weather, or real-time information."
)

tools = [search_tool]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from agentic-ai-engineering-course!")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Cairo?")]})
    print(result)

if __name__ == "__main__":
    main()
