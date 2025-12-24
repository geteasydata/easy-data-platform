
import os

content = """# .streamlit/secrets.toml
# These keys are for LOCAL development only.

[api_keys]
GROQ_API_KEY = "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"
DEEPSEEK_API_KEY = "sk-0d8b9806c944495387bd466460a53932"
GEMINI_API_KEY = "AIzaSyC_0gsd0E7_Xf3g64ReTlVCvrgM7m2spwE"

[supabase]
SUPABASE_URL = "https://banmjznaaunsnlerzqfg.supabase.co"
SUPABASE_KEY = "sb_publishable_CowgsCfN6xjd37B3IoVLBg_omAs2HoQ"

[email]
# SendGrid or Resend API Key
# SENDGRID_API_KEY = "SG.your_key_here"
EMAIL_FROM = "noreply@geteasydata.com"

[payment]
# Paddle or LemonSqueezy Keys (Pending Approval)
# PADDLE_VENDOR_ID = "..."
# PADDLE_API_KEY = "..."
"""

with open(".streamlit/secrets.toml", "w", encoding="utf-8") as f:
    f.write(content)

print("secrets.toml updated successfully")
