# update_secrets.py - TEMPLATE ONLY
# This file creates a template for secrets.toml
# DO NOT put actual API keys here!

import os

content = """# .streamlit/secrets.toml
# Fill in your actual API keys below.
# NEVER commit real keys to version control!

[api_keys]
GROQ_API_KEY = "your_groq_api_key_here"
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
GEMINI_API_KEY = "your_gemini_api_key_here"

[supabase]
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your_supabase_anon_key_here"

[email]
EMAIL_PROVIDER = "resend"
RESEND_API_KEY = "your_resend_api_key_here"
EMAIL_FROM = "noreply@geteasydata.com"

[payment]
# Paddle or LemonSqueezy Keys
# PADDLE_VENDOR_ID = "..."
# PADDLE_API_KEY = "..."
"""

with open(".streamlit/secrets.toml", "w", encoding="utf-8") as f:
    f.write(content)

print("secrets.toml template created - please fill in your actual keys!")
