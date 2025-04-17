import os
import autogen
from autogen.io.websockets import IOWebsockets


llm_config={
    "config_list": [
        {
            "model": "llama3.2:3b", 
            "api_type" : "ollama",
            "temperature": 0.9, 
            "max_tokens": 500,
            "base_url" : "http://127.0.0.1:11434"
        }
    ]
}


agent = autogen.ConversableAgent(
    name="chatbot",
    system_message="Complete a task given to you and reply TERMINATE when the task is done. If asked about the weather, use tool 'weather_forecast(city)' to get the weather forecast for a city.",
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    system_message="A proxy for the user.",
    max_consecutive_auto_reply=10,
    code_execution_config=False,
)

user_proxy.initiate_chat(
    agent,
    message="what is the weather in Taipei?",
)

