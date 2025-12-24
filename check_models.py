# check_models.py - Test script to list available Gemini models
# Usage: Set GEMINI_API_KEY environment variable first

import google.generativeai as genai
import os

# Get API key from environment - NEVER hardcode!
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("❌ Error: GEMINI_API_KEY environment variable not set")
    print("   Set it with: export GEMINI_API_KEY='your_key_here'")
    exit(1)

print(f"Checking models with key: {api_key[:10]}...")

try:
    genai.configure(api_key=api_key)
    print("\n--- Available Models ---")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
    print("------------------------\n")
except Exception as e:
    print(f"\n❌ Error listing models: {e}")
