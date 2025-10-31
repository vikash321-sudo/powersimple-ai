import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="PowerAI – Smart AI Assistant", page_icon="🤖")

# --- APP TITLE ---
st.markdown("<h1 style='text-align: center;'>🤖 PowerAI – Smart AI Assistant</h1>", unsafe_allow_html=True)

# --- SESSION SETUP ---
if "memory_k" not in st.session_state:
    st.session_state.memory_k = 6

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=st.session_state.memory_k)

# --- SIDEBAR SETTINGS ---
st.sidebar.title("⚙️ Memory Settings")
st.sidebar.write("Adjust how many past messages PowerAI remembers.")
st.session_state.memory_k = st.sidebar.slider("K (last messages to keep)", 2, 12, st.session_state.memory_k)

# --- LLM SETUP ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ Missing OPENAI_API_KEY! Add it in Streamlit → Settings → Secrets.")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, api_key=api_key)

# --- CONVERSATION CHAIN ---
conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=False
)

# --- CHAT UI ---
user_input = st.text_input("💬 Type your message…")

if user_input:
    with st.spinner("🤔 Thinking..."):
        response = conversation.run(user_input)
        st.write(f"**PowerAI:** {response}")

# --- SIDEBAR ACTIONS ---
st.sidebar.write("---")
st.sidebar.subheader("🧠 Memory Control")
if st.sidebar.button("Clear Chat Memory"):
    st.session_state.memory.clear()
    st.experimental_rerun()
