from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# Load .env variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize GPT-4o mini
llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini")

# Memory buffer for conversation context
memory = ConversationBufferMemory()

# Create conversation chain
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

print("✅ PowerAI Memory Bot ready to chat!")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("👋 Goodbye from PowerAI!")
        break
    response = conversation.invoke({"input": user_input})
    print("PowerAI:", response["response"])
