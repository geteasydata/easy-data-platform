import resend
import os

# Manual Config for Testing
API_KEY = "re_jDmamPb1_3SWgAXdo9NNHrMy8JjxQRcVf"
FROM_EMAIL = "onboarding@resend.dev"
TO_EMAIL = "sameh599samir@gmail.com" # المسموح به حالياً فقط في Resend

def test_email():
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
