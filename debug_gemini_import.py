import sys
print(f"Python Executable: {sys.executable}")
print(f"Path: {sys.path}")

try:
    import google.generativeai as genai
    print("SUCCESS: google.generativeai imported.")
    print(f"Version: {genai.__version__}")
except ImportError as e:
    print(f"FAILURE: ImportError - {e}")
except Exception as e:
    print(f"FAILURE: Exception - {e}")

try:
    import groq
    print("SUCCESS: groq imported.")
except ImportError as e:
    print(f"FAILURE: groq ImportError - {e}")
