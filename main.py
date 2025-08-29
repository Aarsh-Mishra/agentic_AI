import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")  

async def main():
   
    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash",  
        api_key=api_key,
        
        model_info=ModelInfo(
            vision=True,
            function_calling=True,
            json_output=True,
            structured_output=True,
            family="gemini"
        )
    )


    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        description="A Gemini-powered Agent"
    )

   
    result = await assistant.run(task="What's the capital of India & write some more about it?")
    # print("Result:", result)
    print(result.messages[-1].content)

  
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
