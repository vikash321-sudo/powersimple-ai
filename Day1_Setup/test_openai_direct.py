from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

resp = client.chat.completions.create(
    model="gpt-4o-mini",  # fallback: "gpt-4o"
    messages=[
        {"role": "system", "content": "You are a clear, concise AI tutor."},
        {"role": "user", "content": "Say hello and explain what an AI agent is in 3 simple lines."},
    ],
)
print(resp.choices[0].message.content)
