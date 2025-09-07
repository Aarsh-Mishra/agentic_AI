import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
import os
from dotenv import load_dotenv
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from autogen_ext.tools.http import HttpTool

# Load environment variables
# load_dotenv()
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

schema = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "A random cat fact"
            },
            "length": {
                "type": "integer",
                "description": "Length of the cat fact"
            }
        },
        "required": ["fact", "length"],
    }

http_tool=HttpTool(
    name='cat_facts_api',
    description='Fetch random cat facts from the Cat Facts API',
    scheme='https',
    host='catfact.ninja',
    port=443,
    path='/fact',
    method='GET',
    return_type='json',
    json_schema=schema)

# Define a custom function to reverse a string
def reverse_string(text: str,) -> str:
    """Reverse the given text."""
    return text[::-1]

async def main():
    # Create an assistant with the base64 tool
    assistant = AssistantAgent(
        "cat_fact_agent", 
        model_client=model_client, 
        tools=[http_tool,reverse_string],
        system_message="You are a helpful assistant that can fetch random cat facts (fdirectly call the tool, no changes/inputs) and reverse strings using Tools.",)

    # The assistant can now use the base64 tool to decode the string
    response = await assistant.on_messages(
        [TextMessage(content="Can you please fetch a cat fact using the tool?.", source="user")],
        CancellationToken(),
    )
    print(response.chat_message)


asyncio.run(main())