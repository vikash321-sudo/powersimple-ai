import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

# --- PAGE SETUP ---
st.set_page_config(
    page_title="PowerAI • Smart Assistant",
    page_icon="🤖",
    layout="centered"
)

# --- HEADER ---
st.title("🤖 PowerAI — Your Smart AI Assistant")
st.write("Hello! I’m PowerAI, powered by GPT-4o mini & LangChain. Let’s build something powerful together 🚀")

# --- SETTINGS SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Model: **gpt-4o-mini**")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.6, 0.1)
    st.markdown("Session Active ✅")
    if st.button("Clear chat"):
        st.session_state.memory = ConversationBufferMemory()

# --- OPENAI SETUP ---
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature, openai_api_key=api_key)

# --- MEMORY ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

memory = st.session_state.memory

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# --- CHAT INTERFACE ---
user_input = st.chat_input("Type your message…")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking…"):
            response = conversation.run(user_input)
            st.write(response)

# --- FOOTER ---
st.markdown("<br><hr><center>Built with ❤️ using Streamlit & LangChain by Vikash Jaishi</center>", unsafe_allow_html=True)
