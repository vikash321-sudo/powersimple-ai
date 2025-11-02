# 🤖 PowerAI — Smart Assistant (OpenAI + LangChain + Streamlit)

### 🚀 Overview
PowerAI is an intelligent conversational assistant built using **OpenAI's GPT-4o mini**, **LangChain**, and **Streamlit**.  
It demonstrates how to integrate LLMs into a fully functional web app — complete with memory, chat interface, and real-time conversation.

---

### ⚙️ Features
- 💬 Real-time Chat Interface
- 🧠 Conversation Memory (Buffer)
- 🌈 Clean Modern UI with Robot Theme
- 🔄 OpenAI GPT-4o mini integration
- 🚀 Deployable on Streamlit Cloud

---

### 🧩 Tech Stack
- **Python 3.10+**
- **LangChain**
- **OpenAI API**
- **Streamlit**
- **dotenv**

---

### 🛠️ Setup
1. Clone the repository  
   ```bash
   git clone https://github.com/vikash321-sudo/powersimple-ai.git
   cd powersimple-ai

   # PowerAI — AI Agents & Memory (Week 2)

Production-ready experiments while I learn & build AI agents:
- ✅ Day 5: **PowerAI (OpenAI + Streamlit)**
- ✅ Day 11: **Persistent Memory Bot (SQLite + LangChain)**
- 🚧 Next: Day 12 Tools (weather/news/crypto), Multi-turn Chatbot, Docs & Deploys

---

## Day 11 — Persistent Memory Bot (SQLite + LangChain)

**What it is:**  
A console chatbot that **remembers across sessions** using `SQLChatMessageHistory` on SQLite.
Close the app, reopen it—your past messages are still there.

### Features
- Long-term memory with SQLite
- Clean LangChain memory pattern (`ConversationBufferMemory`)
- OpenAI `gpt-4o-mini` (easy to swap models)
- Minimal, dependency-stable setup

### Tech Stack
- Python, LangChain, langchain-openai
- SQLite (via `SQLChatMessageHistory`)
- dotenv for secrets

### Folder

