import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

# ✅ Page config FIRST
st.set_page_config(page_title="PowerAI • Your Smart Assistant", page_icon="🤖")

# ✅ App title
st.title("🤖 PowerAI — Your Smart AI Assistant")

# ✅ Environment setup
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🚨 Missing OPENAI_API_KEY! Add it under Streamlit → Settings → Secrets.")
    st.stop()

# ✅ Prompt Template
template = """You are PowerAI, a personal AI assistant built by Vikas.
Keep your tone friendly and concise.

Conversation so far:
{history}

User: {user_input}
PowerAI:"""

prompt = PromptTemplate(input_variables=["history", "user_input"], template=template)

# ✅ Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# ✅ LLM (using OpenAI GPT)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, api_key=api_key)

# ✅ Chain
chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.memory, verbose=False)

# ✅ Chat UI
user_input = st.text_input("💬 Type your message here:")

if user_input:
    with st.spinner("Thinking..."):
        response = chain.run({"history": st.session_state.memory.load_memory_variables({})["history"], "user_input": user_input})
        st.session_state.memory.save_context({"input": user_input}, {"output": response})
        st.write(f"**PowerAI:** {response}")

# ✅ Sidebar session tools
st.sidebar.title("Session Controls")
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.memory.clear()
    st.experimental_rerun()
