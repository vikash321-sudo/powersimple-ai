# PowerAI_MemoryChat.py
# Streamlit chat app with persistent memory (SQLite) using LangChain

import os
from dotenv import load_dotenv
import streamlit as st
from sqlalchemy import create_engine

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory

# --------- App Config ---------
st.set_page_config(page_title="PowerAI — Memory Chat", page_icon="🤖", layout="wide")
load_dotenv()

# --------- Sidebar Controls ---------
st.sidebar.title("⚙️ PowerAI Settings")

# API key: from env or sidebar
env_key = os.getenv("OPENAI_API_KEY") or ""
openai_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=env_key,
    type="password",
    placeholder="sk-proj-********",
    help="Stored in memory for this session only. Prefer Streamlit secrets in production.",
)

# User / session id (each gets its own persistent memory)
session_id = st.sidebar.text_input(
    "Session ID (user id / email / handle)", value="vikas", help="Each ID keeps its own chat history"
)

# System prompt to give your bot a personality
default_system_prompt = (
    "You are PowerAI, a friendly, concise AI assistant created by Vikas Joshi. "
    "You help with AI automation, agents, LangChain, and practical coding. "
    "Be clear. If unsure, say so and suggest next steps. End with a short, motivating line."
)
system_prompt = st.sidebar.text_area("System Prompt (optional)", value=default_system_prompt)

# DB path (local SQLite)
db_path = st.sidebar.text_input("SQLite DB path", value="powerai_memory.db")
engine = create_engine(f"sqlite:///{db_path}")

# Buttons
col_a, col_b = st.sidebar.columns(2)
with col_a:
    clear_runtime = st.button("🧹 Clear Runtime Chat", help="Clears only the visible Streamlit messages")
with col_b:
    hard_reset = st.button("🗑️ Hard Reset Memory", help="Deletes history in SQLite for this Session ID")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Use different Session IDs to keep separate memories per user/client.")

# --------- Header ---------
st.title("🤖 PowerAI — Memory Chat")
st.caption("LangChain + OpenAI + SQLite persistent memory")

# --------- Guards ---------
if not openai_key:
    st.warning("Please provide your OpenAI API Key in the left sidebar to start.", icon="🔑")
    st.stop()

# --------- Init persistent chat history for this session_id ---------
# Using new `connection=` param (avoids deprecation warning)
sql_history = SQLChatMessageHistory(session_id=session_id, connection=engine)

# Wrap it with a Memory object so ConversationChain can use it
memory = ConversationBufferMemory(chat_memory=sql_history, return_messages=True)

# Create / reuse a chain in session_state (keeps the same LLM + memory during the Streamlit session)
def build_chain():
    llm = ChatOpenAI(api_key=openai_key, model="gpt-4o-mini", temperature=0.4)
    # We inject a system prompt by priming with an initial “message” into memory if not present.
    # A clean way: prepend to the first turn by writing to memory once.
    # If you prefer, you can switch to custom prompt templates; this is minimalist & robust.
    if not sql_history.messages:
        # Save an initial assistant “system intro” so the next answers follow personality
        sql_history.add_ai_message(f"[SYSTEM NOTE]\n{system_prompt}")
    return ConversationChain(llm=llm, memory=memory, verbose=False)

if "chain" not in st.session_state:
    st.session_state.chain = build_chain()

# --------- Hard reset memory in DB for this user ---------
if hard_reset:
    # Drop only this session's messages
    # SQLChatMessageHistory has no native delete-all; we recreate the history by dropping rows manually via engine.
    # Minimal safe approach: recreate a fresh SQLChatMessageHistory instance and overwrite table rows for this session.
    # Implementation detail: table is `chat_message` by default; we delete by session_id.
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM chat_message WHERE session_id = :sid", {"sid": session_id})
    # Rebuild in-memory objects after wipe
    sql_history = SQLChatMessageHistory(session_id=session_id, connection=engine)
    memory = ConversationBufferMemory(chat_memory=sql_history, return_messages=True)
    st.session_state.chain = build_chain()
    st.success(f"Memory wiped for session: {session_id}")

# --------- Show existing history (from DB) in the UI ---------
with st.container():
    st.subheader("Chat")
    # Render messages from persistent SQL history except our [SYSTEM NOTE]
    for msg in sql_history.messages:
        role = getattr(msg, "type", getattr(msg, "role", "ai"))  # compatibility
        content = getattr(msg, "content", "")
        if content.startswith("[SYSTEM NOTE]"):
            continue
        if role in ("human", "user"):
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)

# --------- Clear runtime (UI) only ---------
if clear_runtime:
    # UI clears by re-rendering; persistent history is not deleted
    st.rerun()

# --------- Chat input ---------
user_input = st.chat_input("Type your message...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # ConversationChain expects {"input": "..."}
                resp_dict = st.session_state.chain.invoke({"input": user_input})
                reply = resp_dict.get("response", "").strip()
            except Exception as e:
                reply = f"Sorry, something went wrong: `{e}`"
        st.markdown(reply)
