import os
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

# ✅ Load environment variables (.env)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="PowerAI • Smart Assistant",
    page_icon="🤖",
    layout="centered"
)

# --- Header UI ---
st.title("🤖 PowerAI — Your Smart OpenAI Assistant")
st.caption("Built with LangChain + Streamlit + OpenAI GPT Models")

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Session Settings")

session_id = st.sidebar.text_input("Session ID", "powerai-session-1")
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.6)
memory_k = st.sidebar.slider("Memory (Last K Messages)", 2, 12, 6)

# --- Memory Setup ---
memory = ConversationBufferWindowMemory(k=memory_k)

# --- Initialize LLM ---
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=temperature
)

# --- Create Conversation Chain ---
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# --- Chat UI ---
st.divider()
st.subheader("💬 Chat with PowerAI")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display previous messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User input ---
if prompt := st.chat_input("Ask PowerAI anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Generate response ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversation.predict(input=prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# --- Footer ---
st.divider()
st.caption("⚡ Powered by OpenAI GPT • LangChain • Streamlit • Developed by Vikas Joshi")
