import os, requests
from dotenv import load_dotenv

# Always load from parent directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
print("Looking for .env at:", ENV_PATH)
load_dotenv(ENV_PATH)

key = os.getenv("OPENAI_API_KEY")
model = os.getenv("MODEL_NAME", "gpt-4o-mini")
assert key, "OPENAI_API_KEY missing (check .env location or spelling)"

print("✅ .env loaded successfully!")

url = "https://api.openai.com/v1/chat/completions"
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Say OK if working"}],
}
headers = {"Authorization": f"Bearer {key}"}

r = requests.post(url, headers=headers, json=payload)
print("Response:", r.status_code, r.text[:500])
