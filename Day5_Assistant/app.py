import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

# ✅ Load the OpenAI key from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ✅ Set up the Streamlit page
st.set_page_config(
    page_title="🤖 PowerAI — Your Smart AI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 PowerAI — Smart AI Assistant")
st.write("Built with **LangChain + OpenAI GPT-4o-mini** ⚡")

# Sidebar settings
st.sidebar.header("⚙️ Configuration")
temperature = st.sidebar.slider("Response Creativity", 0.0, 1.0, 0.6)
memory_buffer = st.sidebar.slider("Memory Retention (messages)", 2, 10, 5)

# ✅ Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", k=memory_buffer)

# ✅ Load the GPT-4o-mini model
llm = ChatOpenAI(
    model="gpt-4o-mini",  # ✅ Use GPT-4o-mini
    temperature=temperature,
)

# ✅ Conversation chain setup
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Chat UI
user_input = st.text_input("💬 Type your message here...")

if user_input:
    response = conversation.predict(input=user_input)
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(response)

# Footer
st.markdown("---")
st.caption("🚀 Developed by Vikas | Powered by LangChain + OpenAI")
