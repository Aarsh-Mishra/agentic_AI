import asyncio
from dotenv import load_dotenv
import os
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from langchain_community.utilities import GoogleSerperAPIWrapper

# Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",  
        api_key=api_key,
        
        model_info=ModelInfo(
            vision=True,
            function_calling=True,
            json_output=True,
            structured_output=True,
            family="gemini"
        )
    )

# Set up Serper API key
serper_api_key = os.getenv('SERPER_API_KEY')

# Initialize the search tool wrapper
search_tool_wrapper = GoogleSerperAPIWrapper(type='news')

# Create a simple function to use as a tool
def search_web(query: str) -> str:
    """Search the web using Serper API"""
    try:
        return search_tool_wrapper.run(query)
    except Exception as e:
        return f"Search failed: {str(e)}"


# Create an agent with the search tool
search_agent = AssistantAgent(
    name='SearchAgent',
    model_client=model_client,
    system_message="""You are a helpful assistant that can search the web to find current information. 
    When asked a question, use the search_web tool to find relevant information and provide a comprehensive answer based on the search results.""",
    description='Searches the internet and provides detailed answers based on search results',
    tools=[search_web],
    reflect_on_tool_use=True
)

# Test function to demonstrate the tool
async def demonstrate_search():
    """Demonstrate the search functionality"""
    print("=== AutoGen Third-Party Tools Demonstration ===\n")
    
    # Test queries for demonstration
    test_queries = [
        "who is the current prime minister of India?",
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 50)
        
        try:
            # Run the agent with the query
            result = await search_agent.run(task=query)
            print(f"Response: {result.messages[-1].content}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "="*70 + "\n")

# Main execution
async def main():
    await demonstrate_search()

if __name__ == "__main__":
    asyncio.run(main())