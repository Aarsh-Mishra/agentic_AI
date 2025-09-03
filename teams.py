import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import (AssistantAgent)
from autogen_core.models import ModelInfo
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.conditions import MaxMessageTermination,TextMentionTermination
from autogen_agentchat.base import TaskResult
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") 


async def teams():
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
     
    plot_agent = AssistantAgent(
        name = 'plot_writer',
        model_client=model_client,
        system_message="You create engaging plots for stories. Focus on the Pokemon's journey"
    )

    character_agent = AssistantAgent(
        name = 'character_writer',
        model_client=model_client,
        system_message="You develop characters. Describe the pokemon and the villian in detail, including their motivations and backgrounds."
    )

    ending_agent = AssistantAgent(
        name = 'ending_writer',
        model_client=model_client,
        system_message="You write engaging endings. conclude the story with a twist."
    )

    review_agent = AssistantAgent(
        name = 'review_writer',
        model_client=model_client,
        system_message="You have to say 'APPROVE' once the story is complete and you like the full flow else provide your feedback."
    )

    termination_condition = MaxMessageTermination(10) | TextMentionTermination('APPROVE')

    team = RoundRobinGroupChat(
        participants= [plot_agent, character_agent, ending_agent, review_agent],
        termination_condition=termination_condition
        # max_turns=6
    )

    async def test_team():
        task = TextMessage(
            content='Write a short story a brave boy and his Pokemon. Keep it up to 20 words',
            source='user'
        )

        result = await team.run(task=task)
        for each_agent_message in result.messages:
            print(f'{each_agent_message.source} : {each_agent_message.content}')
        
        await team.reset() 
        async for message in team.run_stream(task="Write a short poem about the summer season in India."):  
            if isinstance(message, TaskResult):
                print("Stop Reason:", message.stop_reason)
            else:
                print(message)
        

    await test_team()
    

    await model_client.close()

   


if __name__ == "__main__":
    asyncio.run(teams())