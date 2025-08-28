import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key)

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        description="A Basic first Agent"
    )

    result = await assistant.run(task="What's the capital of USA & write some more about the same ?")
    print(result)

    await model_client.close() 

if __name__ == "__main__":
    asyncio.run(main())
