import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")  

async def web_search(query: str) -> str:
    """Find information on the web"""
    return "The Labrador Retriever or simply Labrador is a British breed of retriever gun dog. "

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


    # assistant = AssistantAgent(
    #     name="assistant",
    #     model_client=model_client,
    #     description="A Gemini-powered Agent"
    # )

    agent = AssistantAgent(
        name = 'assistant',
        model_client=model_client,
        tools = [web_search],
        system_message='Use Tools to solve tasks',
        description = "An agent that uses tool to help solve tasks"
    )


    result = await agent.run(task="Find information about Labrador Retriever")
    # print("Result:", result)
    print(result.messages[-1].content)

     

    # call on_messages via a helper (shows how to pass CancellationToken)
    
    async def assistant_run()-> None:
        response = await agent.on_messages(
            messages= [TextMessage(content='Find information about Labrador Retriever via the tool',source='User')],
            cancellation_token=CancellationToken()
        )

        print(response.inner_messages)
        print('\n\n\n\n')
        print(response.chat_message)

    await assistant_run()

  
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
