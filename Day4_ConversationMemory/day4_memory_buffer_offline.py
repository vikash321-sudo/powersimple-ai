from langchain.memory import ConversationBufferMemory
import re

def extract_name_from_history(history_text: str) -> str | None:
    # look for "my name is <...>" (case-insensitive), grab first match
    m = re.search(r"\bmy\s+name\s+is\s+([A-Za-z][A-Za-z\s]+)", history_text, flags=re.I)
    if m:
        return m.group(1).strip()
    # also try "I'm <...>" or "I am <...>" as fallback
    m = re.search(r"\b(i\'m|i am)\s+([A-Za-z][A-Za-z\s]+)", history_text, flags=re.I)
    if m:
        return m.group(2).strip()
    return None

def fake_llm_reply(context: str, user_msg: str) -> str:
    lower_user = user_msg.lower()
    name = extract_name_from_history(context)

    # explicit QAs using memory
    if "what's my name" in lower_user or "what is my name" in lower_user or "who am i" in lower_user:
        if name:
            return f"Your name is {name}."
        return "I don't see your name yet. Tell me: 'My name is <your name>'."

    if "remind me what i said earlier" in lower_user or "remind me" in lower_user:
        # show last couple of user lines from context
        lines = [ln for ln in context.splitlines() if ln.startswith("HUMAN:")]
        last = " | ".join(ln.replace("HUMAN:", "").strip() for ln in lines[-3:])
        return f"Recently you said: {last or '(no prior messages)'}"

    # generic reply with a friendly memory hint
    hint = "I can see our previous chat in context."
    if name:
        hint += f" I remember your name is {name}."
    return f"{hint} You said: '{user_msg[:140]}'"

def main():
    print("🤖 Day 4 — ConversationBufferMemory (Offline). Type 'exit' to quit.")
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Assistant: Bye! 👋")
            break

        # 1) Pull current memory into the prompt variables
        vars_now = memory.load_memory_variables({})
        history_msgs = vars_now.get("history", [])

        # Build a simple text context for our offline LLM
        context_text = "\n".join(
            f"{m.type.upper()}: {m.content}" for m in history_msgs
        )

        # 2) Get a reply using the buffered history
        reply = fake_llm_reply(context_text, user)

        # 3) Save this turn back to memory
        memory.save_context({"input": user}, {"output": reply})

        # 4) Show assistant reply
        print(f"Assistant: {reply}")

if __name__ == "__main__":
    main()
