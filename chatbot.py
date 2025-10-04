import getpass
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages, BaseMessage
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from typing import Sequence


CHARACTER_SYSTEM_MESSAGES = {
    "spiderman": "You are the character Spiderman from Marvel Comics and Cinematic Universe. Answer all questions to the best of your ability in {language}.",
    "harry_potter": "You are the character Harry Potter from the novel and movie Harry Potter. Answer all questions to the best of your ability in {language}.",
    "default": "You are a helpful assistant. Answer all questions to the best of your ability in {language}."
}

CHARACTER = "default"
THREAD_ID = 0
LANGUAGE = "english"

CHARACTER_PROMPT_TEMPLATES = {
    character: ChatPromptTemplate.from_messages(
        [
            (
                "system", prompt
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    ) for character, prompt in CHARACTER_SYSTEM_MESSAGES.items()
}

THREAD_MESSAGES = {}

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# identify message schema
class CustomState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
    character: str

# handle long conversation context
trimmer = trim_messages(
    max_tokens=1024,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

# Define a new graph
workflow = StateGraph(state_schema=CustomState)

def call_model(state: CustomState):
    # print("State messages", state["messages"])
    trimmed_message = trimmer.invoke(state["messages"])
    # print("Trimmed messages", trimmed_message)
    character = state["character"]
    prompt_template = CHARACTER_PROMPT_TEMPLATES.get(character, CHARACTER_PROMPT_TEMPLATES.get("default"))
    prompt = prompt_template.invoke(
        {"messages": trimmed_message, "language": state["language"]}
    )
    # print("Prompt:", prompt)
    response = model.invoke(prompt)
    return {"messages": [response]}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def chat_with_bot(user_input):
    config = {"configurable": {"thread_id": THREAD_ID}}

    if THREAD_ID not in THREAD_MESSAGES:
        system_message_template = CHARACTER_SYSTEM_MESSAGES.get(character, CHARACTER_SYSTEM_MESSAGES["default"])
        system_message = system_message_template.format(language=LANGUAGE)

        THREAD_MESSAGES[THREAD_ID] = [SystemMessage(system_message)]
    input_messages = THREAD_MESSAGES[THREAD_ID] + [HumanMessage(user_input)]
    output = app.invoke(
        {"messages": input_messages, "character": CHARACTER, "language": LANGUAGE},
        config,
    )
    next_message = output["messages"][-1]
    THREAD_MESSAGES[thread_id] = input_messages + [AIMessage(next_message.content)]
    return next_message.content

if __name__ == "__main__":
    print("LangChain Chatbot (Type '>exit' to quit, '>select [character] [thread_id] [language]' to change character and thread id)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == ">exit":
            break
        if user_input.lower().startswith(">select"):
            tokens = user_input.lower().split(" ")
            character, thread_id, language = tokens[1], tokens[2], tokens[3]
            CHARACTER = character
            THREAD_ID = thread_id
            LANGUAGE = language
            print(f"System: Change to chracter: {character} thread_id: {thread_id} language: {language}")
        else:
            bot_response = chat_with_bot(user_input.encode('utf-8', errors='ignore'))
            print(f"Bot: {bot_response}")