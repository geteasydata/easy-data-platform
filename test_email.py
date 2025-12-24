# test_email.py - Test script for email functionality
# Usage: Set RESEND_API_KEY environment variable first

import resend
import os

# Get API key from environment - NEVER hardcode!
API_KEY = os.getenv('RESEND_API_KEY')
FROM_EMAIL = "onboarding@resend.dev"
TO_EMAIL = os.getenv('TEST_EMAIL', 'your_email@example.com')

def test_email():
    if not API_KEY:
        print("❌ Error: RESEND_API_KEY environment variable not set")
        print("   Set it with: export RESEND_API_KEY='your_key_here'")
        return False
    
    print(f"Testing email from {FROM_EMAIL} to {TO_EMAIL}...")
    resend.api_key = API_KEY
    
    try:
        params = {
            "from": f"Easy Data <{FROM_EMAIL}>",
            "to": [TO_EMAIL],
            "subject": "الاختبار النهائي لبريد المنصة 💎",
            "html": "<h3>مبروك! 🎉</h3><p>إذا وصلك هذا البريد، فهذا يعني أن نظام الإرسال يعمل بنجاح.</p>",
        }

        email = resend.Emails.send(params)
        print(f"Success! ID: {email.get('id')}")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_email()
