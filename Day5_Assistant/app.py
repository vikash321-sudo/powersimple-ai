# app.py — Day 5 • Simple AI Assistant (OpenAI version)
# Stack: Streamlit + LangChain (ChatOpenAI)
# Requires: langchain==0.2.13, langchain-core==0.2.34, langchain-openai==0.1.7,
#           streamlit==1.39.0, python-dotenv==1.0.1

import os
import json
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
import streamlit as st

# ---- Page config MUST be the first Streamlit call ----
st.set_page_config(
    page_title="Day 5 • Simple AI Assistant (OpenAI)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Load env ----
load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI

# ---- Config from .env with safe defaults ----
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
    st.error("⚠️ OPENAI_API_KEY not found or invalid. Set it in .env (sk- / sk-proj-).")
    st.stop()

# ---- Initialize LLM ----
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    api_key=OPENAI_API_KEY,
)

# ---------------- Utilities ----------------
K_WINDOW = 6  # windowed memory size for short context

def build_context(messages, user_prompt):
    """Builds a compact prompt using last K messages + user input."""
    recent = messages[-K_WINDOW:]
    hist = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        text = m["content"].strip()
        if len(text) > 800:
            text = text[:800] + "…"
        hist.append(f"{role}: {text}")
    history_block = "\n".join(hist) if hist else "(no prior context yet)"
    system = (
        "You are a concise, helpful AI assistant. "
        "Use the recent chat history for continuity. "
        "If you aren't sure, ask a brief clarification question."
    )
    return f"{system}\n\n--- Recent history ---\n{history_block}\n\n--- New question ---\nUser: {user_prompt}\nAssistant:"

def export_transcript(rows):
    lines = []
    for r in rows:
        lines.append(f"{r['role'].upper()}:\n{r['content']}\n")
    return "\n".join(lines)

# ---------------- UI State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role: "user"|"assistant", content: "..."}]

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")
st.sidebar.write(f"**Model:** `{MODEL_NAME}`")
st.sidebar.write(f"**Temperature:** `{TEMPERATURE}`")

colA, colB = st.sidebar.columns(2)
with colA:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
with colB:
    if st.button("Export .txt"):
        txt = export_transcript(st.session_state.messages)
        st.sidebar.download_button("Download", txt, file_name=f"powersimple_day5_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")

st.sidebar.success("Session Active ✅")

# ---------------- Title ----------------
st.markdown("<h1 style='text-align:center;'>🤖 Day 5 — Simple AI Assistant (OpenAI)</h1>", unsafe_allow_html=True)

# ---------------- Render History ----------------
for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

# ---------------- Chat Input ----------------
user_msg = st.chat_input("Type your message…")
if user_msg:
    # Show + store user message
    st.chat_message("user").markdown(user_msg)
    st.session_state.messages.append({"role": "user", "content": user_msg})

    # Build compact context and invoke LLM
    ctx = build_context(st.session_state.messages, user_msg)
    try:
        with st.spinner("🤔 Thinking..."):
            resp = llm.invoke(ctx)
        reply = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        reply = f"⚠️ Error: {e}"

    # Show + store assistant reply
    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------------- Footer ----------------
st.markdown(
    "<p style='text-align:center;color:gray;'>Built by <b>Vikas</b> • OpenAI + LangChain + Streamlit</p>",
    unsafe_allow_html=True
)
