import getpass
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages, BaseMessage
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Import relevant functionality
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

search = TavilySearch(max_results=2)
tools = [search]
memory = MemorySaver()
agent_executor = create_react_agent(model, tools, checkpointer=memory)

if __name__ == "__main__":
    print("LangChain Agent (Type '>exit' to quit)")
    while True:
        config = {"configurable": {"thread_id": "abc123"}}
        user_input = input("You: ")
        if user_input.lower() == ">exit":
            break
        input_message = {"role": "user", "content": user_input.encode('utf-8', errors='ignore')}
        bot_response = agent_executor.invoke({"messages": [input_message]}, config)
        for message in bot_response["messages"]:
            message.pretty_print()
