
import google.generativeai as genai
import os

# Default Key from the codebase
api_key = "AIzaSyC_0gsd0E7_Xf3g64ReTlVCvrgM7m2spwE"

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
