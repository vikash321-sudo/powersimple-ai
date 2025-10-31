import re
from textwrap import shorten

# ---------------------------
# Simple in-memory structures
# ---------------------------
class ConversationMemory:
    def __init__(self, window_size=4, summary_target_len=500):
        self.turns = []              # full buffer: list of {"role": "user"/"assistant", "text": "..."}
        self.summary = ""            # rolling summary
        self.entities = {}           # entity store: {entity -> {facts}}
        self.window_size = window_size
        self.summary_target_len = summary_target_len

    # ---- buffer operations ----
    def add(self, role, text):
        self.turns.append({"role": role, "text": text})

    def last_window(self):
        return self.turns[-self.window_size*2:]  # roughly last K exchanges

    # ---- naive summary (offline) ----
    def update_summary(self):
        """
        Offline summary: compresses the conversation turns to a short bullet overview.
        In a real app, you'd call an LLM to improve this.
        """
        bullets = []
        for t in self.turns[-10:]:  # summarise recent context
            who = "U" if t["role"] == "user" else "A"
            line = t["text"].strip().replace("\n", " ")
            bullets.append(f"{who}: {line}")
        joined = " • ".join(bullets)
        # keep it compact
        self.summary = shorten(joined, width=self.summary_target_len, placeholder=" ...")

    # ---- naive entity extraction (offline) ----
    def extract_entities(self, text):
        """
        Extremely naive: capture patterns like "My name is X", "I like Y", "I use Z".
        Replace with real NER later.
        """
        name_match = re.search(r"\bmy name is ([A-Za-z][A-Za-z\s]+)", text, re.I)
        if name_match:
            self.entities.setdefault("user", {})["name"] = name_match.group(1).strip()

        like_match = re.findall(r"\bI (?:like|love|prefer)\s+([A-Za-z0-9\-\_ ]+)", text, re.I)
        if like_match:
            self.entities.setdefault("user", {}).setdefault("likes", set()).update(map(str.strip, like_match))

        use_match = re.findall(r"\bI (?:use|work with)\s+([A-Za-z0-9\-\_ ]+)", text, re.I)
        if use_match:
            self.entities.setdefault("user", {}).setdefault("tools", set()).update(map(str.strip, use_match))

    def persona_card(self):
        u = self.entities.get("user", {})
        name = u.get("name", "User")
        likes = ", ".join(sorted(u.get("likes", []))) if u.get("likes") else "—"
        tools = ", ".join(sorted(u.get("tools", []))) if u.get("tools") else "—"
        return f"Persona → name: {name} | likes: {likes} | tools: {tools}"

# ---------------------------
# Offline "LLM" (placeholder)
# ---------------------------
def offline_llm(context, user_msg):
    """
    A local rule-based responder so you can test memory without any API.
    Later, you will replace this with your real LLM call (OpenAI/Gemini).
    """
    # simple canned logic
    if "your name" in user_msg.lower():
        return "I'm your local study assistant. You can call me EchoTutor."
    if "summary" in user_msg.lower():
        return f"Here’s the running summary I have:\n{context.get('summary','(empty)')}"
    if "who am i" in user_msg.lower() or "my name" in user_msg.lower():
        return f"From what I remember: {context.get('persona','(no persona yet)')}"

    # default response that references memory window size
    last_user = [t['text'] for t in context.get('window', []) if t['role'] == 'user']
    hint = f"(I see your recent {len(last_user)} message(s).)"
    return f"{hint} You said: '{user_msg[:140]}'. Tell me more so I can help."

# ---------------------------
# Chat loop with memory
# ---------------------------
def build_context(mem: ConversationMemory):
    return {
        "summary": mem.summary,
        "window": mem.last_window(),
        "persona": mem.persona_card()
    }

def main():
    print("🤖 Offline Chat with Memory (no API) — type 'exit' to quit.")
    mem = ConversationMemory(window_size=4, summary_target_len=500)

    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Assistant: Bye! 👋")
            break

        # update memory with user turn
        mem.add("user", user)
        mem.extract_entities(user)
        mem.update_summary()

        # build context & get reply
        ctx = build_context(mem)
        reply = offline_llm(ctx, user)

        # store assistant turn
        mem.add("assistant", reply)

        # show assistant
        print(f"Assistant: {reply}")

        # (debug) show memory snippets
        # print("[DEBUG] Summary:", mem.summary[:120])
        # print("[DEBUG] Persona:", mem.persona_card())

if __name__ == "__main__":
    main()

