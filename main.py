import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_core import CancellationToken
from autogen_agentchat.ui import Console
from autogen_core import Image as AGImage

from PIL import Image
from io import BytesIO
import requests


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")  

async def web_search(query: str) -> str:
    """Find information on the web"""
    return "The Labrador Retriever or simply Labrador is a British breed of retriever gun dog. "

async def main():
   
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


    # assistant = AssistantAgent(
    #     name="assistant",
    #     model_client=model_client,
    #     description="A Gemini-powered Agent"
    # )

    # agent = AssistantAgent(
    #     name = 'assistant',
    #     model_client=model_client,
    #     tools = [web_search],
    #     system_message='Use Tools to solve tasks',
    #     description = "An agent that uses tool to help solve tasks"
    # )

    multiModelAgent = AssistantAgent(
        name = 'assistant',
        model_client= model_client,
        system_message='you are an funny agent but helpful so answer questions with humor and accuracy'
    )



    # result = await multiModelAgent.run(task="Find information about Labrador Retriever")
    # # print("Result:", result)
    # print(result.messages[-1].content)

     


    # call on_messages via a helper (shows how to pass CancellationToken)



    # async def assistant_run()-> None:
    #     response = await agent.on_messages(
    #         messages= [TextMessage(content='Find information about Labrador Retriever via the tool',source='User')],
    #         cancellation_token=CancellationToken()
    #     )

    #     print(response.inner_messages)
    #     print('\n\n\n\n')
    #     print(response.chat_message)

    # await assistant_run()



    # async def assistant_run_stream() -> None:

    #     await Console(
    #         agent.on_messages_stream(
    #         messages= [TextMessage(content='Find information about Labrador Retriever via the tool',source='User')],
    #         cancellation_token=CancellationToken()
    #     ),
    #     output_stats=True # Enable stats Printing
    #     )

    # await assistant_run_stream()

    # async def assistant_run_stream_2() -> None:
    #     await Console(
    #         agent.on_messages_stream(
    #         messages= [TextMessage(content='What was the last question I asked ?',source='User')],
    #         cancellation_token=CancellationToken()
    #     ),
    #     output_stats=True # Enable stats Printing
    #     )

    # await assistant_run_stream_2()


    async def test_multi_modal():
        
        response = requests.get('https://picsum.photos/id/237/200/300')
        pil_image = Image.open(BytesIO(response.content))
        ag_image = AGImage(pil_image)

        multi_modal_msg = MultiModalMessage(
            content = ['What is in the image?',ag_image],
            source='user'
        )

        result = await multiModelAgent.run(task=multi_modal_msg)
        print(result.messages[-1].content)


    await test_multi_modal()

    await model_client.close()

   


if __name__ == "__main__":
    asyncio.run(main())
