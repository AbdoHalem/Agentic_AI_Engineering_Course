from dotenv import load_dotenv
# Load environment variables (e.g., GOOGLE_API_KEY, LANGSMITH_API_KEY) from .env file
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

# Define the maximum number of reasoning steps the agent can take to avoid infinite loops
MAX_ITERATIONS = 10
MODEL = "gemini-2.5-flash"      # Specify the Gemini model to be used

# ------- Tools (LangChain @tool Decorator) ------- #
# ============================================================================
# TOOLS DEFINITION
# The @tool decorator converts standard Python functions into tools that the LLM 
# can invoke. The docstrings (""") are critical as the LLM reads them to understand 
# the tool's purpose and expected inputs.
# ============================================================================
@tool
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog.
    """
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold.
    """
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# ------- Agent Loop ------- #
# ============================================================================
# AGENT LOOP IMPLEMENTATION
# The @traceable decorator logs the execution flow and metrics to LangSmith
# ============================================================================
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    # Initialize the tools and create a dictionary for fast lookup by tool name
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    # Initialize the Gemini chat model with temperature 0 for deterministic outputs
    llm = init_chat_model(f"google_genai:{MODEL}", temperature=0)
    # Bind the tools to the LLM so it knows what external functions it can call
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    # Initialize the conversation history with a System Prompt containing strict rules
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant."
                "You have access to a product catatlog tool"
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question)
    ]

    # Start the custom ReAct (Reasoning and Acting) loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Send the current conversation history to the LLM
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # Base Case: If the LLM doesn't request any tools --> it has formulated the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        # For simplicity, extract and process only the FIRST tool requested by the LLM
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        # Retrieve the actual Python function from our dictionary using the tool's name
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' is not found")

        # Execute the Python tool with the arguments provided by the LLM
        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        # Update the conversation history to maintain context for the next iteration:
        # 1. Append the LLM's message (which includes the tool invocation request)
        messages.append(ai_message)
        # 2. Append the actual result (observation) returned by the Python function
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    # If the loop exhausts the MAX_ITERATIONS without a final answer, abort to prevent infinite loops
    print("ERROR: Max iterations reached without a final answer")
    return None




def main():
    print(f"Hello LangChain Agent (.bind_tools)")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")


if __name__ == "__main__":
    main()
