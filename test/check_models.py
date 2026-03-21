# check_models.py
import os
import requests
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: GOOGLE_API_KEY not set in environment")
    exit(1)

print(f"API key found: {api_key[:8]}...")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

response = requests.get(url)

print(f"Status code: {response.status_code}")
print(f"Raw response: {response.text[:2000]}")

data = response.json()
models = data.get("models", [])

print(f"\nTotal models found: {len(models)}")

for model in models:
    print(f"  - {model['name']}")