# day11_memory_persist.py
import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Database setup
DB_PATH = "chat_memory.db"
table_name = "chat_history"

# Create a persistent SQL chat history (NEW syntax)
history = SQLChatMessageHistory(
    session_id="user1",  # 👈 required new parameter
    connection=f"sqlite:///{DB_PATH}",
    table_name=table_name
)

# Memory setup
memory = ConversationBufferMemory(chat_memory=history, return_messages=True)

# LLM setup
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.7)

# Conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Chat loop
print("💬 Persistent Chat Memory Enabled! Type 'exit' to quit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("👋 Goodbye! Memory saved to SQLite database.")
        break
    response = conversation.predict(input=user_input)
    print("Assistant:", response)
