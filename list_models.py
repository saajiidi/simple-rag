import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
# If you don't have it in .env, I'll ask the user to provide it or just list models if possible
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Please provide a Google API Key.")
else:
    genai.configure(api_key=api_key)
    print("Available models:")
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            print(f"- {m.name}")
