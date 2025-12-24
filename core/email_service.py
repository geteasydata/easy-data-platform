"""
Email Service Module for Easy Data Platform
Handles transactional emails: verification, password reset, notifications
"""

import os
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
import secrets
import hashlib

# Email providers - will be imported when available
SENDGRID_AVAILABLE = False
RESEND_AVAILABLE = False

try:
    import sendgrid
    from sendgrid.helpers.mail import Mail, Email, To, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    pass

try:
    import resend
    RESEND_AVAILABLE = True
except ImportError:
    pass


# ============================================
# Configuration
# ============================================

def get_email_config() -> Dict:
    """Get email configuration from secrets or environment"""
    config = {
        "provider": None,
        "api_key": None,
        "from_email": "noreply@geteasydata.com",
        "from_name": "Easy Data",
    }
    
    # Try Streamlit secrets first
    try:
        import streamlit as st
        # Check top-level
        config["provider"] = st.secrets.get("EMAIL_PROVIDER")
        config["api_key"] = st.secrets.get("RESEND_API_KEY") or st.secrets.get("SENDGRID_API_KEY")
        config["from_email"] = st.secrets.get("EMAIL_FROM")
        
        # Check [email] section as fallback
        if "email" in st.secrets:
            sec = st.secrets["email"]
            if not config["provider"]: config["provider"] = sec.get("EMAIL_PROVIDER")
            if not config["api_key"]: config["api_key"] = sec.get("RESEND_API_KEY") or sec.get("SENDGRID_API_KEY")
            if not config["from_email"]: config["from_email"] = sec.get("EMAIL_FROM")
            
        # Default provider if API key exists
        if config["api_key"] and not config["provider"]:
            config["provider"] = "resend" if config["api_key"].startswith("re_") else "sendgrid"
            
        # Default from_email if missing
        if not config["from_email"]:
            config["from_email"] = "onboarding@resend.dev" if config["provider"] == "resend" else "noreply@geteasydata.com"
            
    except:
        pass
    
    # Fall back to environment variables
    if not config["api_key"]:
        config["provider"] = os.environ.get("EMAIL_PROVIDER", "resend")
        config["api_key"] = os.environ.get("RESEND_API_KEY") or os.environ.get("SENDGRID_API_KEY")
        config["from_email"] = os.environ.get("EMAIL_FROM", config["from_email"])
    
    return config


def is_configured() -> bool:
    """Check if email service is properly configured"""
    config = get_email_config()
    return bool(config["api_key"] and (SENDGRID_AVAILABLE or RESEND_AVAILABLE))


# ============================================
# Token Generation
# ============================================

def generate_verification_token() -> str:
    """Generate a secure verification token"""
    return secrets.token_urlsafe(32)


def generate_reset_token() -> str:
    """Generate a secure password reset token"""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """Hash a token for secure storage"""
    return hashlib.sha256(token.encode()).hexdigest()


# ============================================
# Email Sending
# ============================================

def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    text_content: str = None
) -> Tuple[bool, str]:
    """
    Send an email using configured provider
    Returns: (success, message)
    """
    config = get_email_config()
    
    if not config["api_key"]:
        return False, "Email service not configured"
    
    if config["provider"] == "resend" and RESEND_AVAILABLE:
        return _send_via_resend(config, to_email, subject, html_content)
    elif SENDGRID_AVAILABLE:
        return _send_via_sendgrid(config, to_email, subject, html_content, text_content)
    else:
        return False, "No email provider available"


def _send_via_sendgrid(config, to_email, subject, html_content, text_content=None):
    """Send email via SendGrid"""
    try:
        sg = sendgrid.SendGridAPIClient(api_key=config["api_key"])
        
        from_email = Email(config["from_email"], config["from_name"])
        to = To(to_email)
        content = Content("text/html", html_content)
        
        mail = Mail(from_email, to, subject, content)
        
        response = sg.client.mail.send.post(request_body=mail.get())
        
        if response.status_code in [200, 201, 202]:
            return True, "Email sent successfully"
        else:
            return False, f"SendGrid error: {response.status_code}"
            
    except Exception as e:
        return False, f"SendGrid error: {str(e)}"


def _send_via_resend(config, to_email, subject, html_content):
    """Send email via Resend"""
    try:
        resend.api_key = config["api_key"]
        
        response = resend.Emails.send({
            "from": f"{config['from_name']} <{config['from_email']}>",
            "to": [to_email],
            "subject": subject,
            "html": html_content,
        })
        
        if response.get("id"):
            return True, "Email sent successfully"
        else:
            return False, "Resend error: No ID returned"
            
    except Exception as e:
        return False, f"Resend error: {str(e)}"


# ============================================
# Email Templates
# ============================================

def get_base_template(content: str, lang: str = "ar") -> str:
    """Wrap content in base email template"""
    
    direction = "rtl" if lang == "ar" else "ltr"
    font_family = "Cairo, Arial, sans-serif" if lang == "ar" else "Inter, Arial, sans-serif"
    
    return f"""
    <!DOCTYPE html>
    <html dir="{direction}">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: {font_family};
                background-color: #0f172a;
                color: #e2e8f0;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border-radius: 16px;
                padding: 32px;
                border: 1px solid rgba(99, 102, 241, 0.2);
            }}
            .header {{
                text-align: center;
                margin-bottom: 32px;
            }}
            .logo {{
                font-size: 32px;
                margin-bottom: 8px;
            }}
            .brand {{
                font-size: 24px;
                font-weight: 700;
                background: linear-gradient(135deg, #6366f1, #8b5cf6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .content {{
                line-height: 1.6;
            }}
            .button {{
                display: inline-block;
                background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
                color: white !important;
                text-decoration: none;
                padding: 12px 32px;
                border-radius: 8px;
                font-weight: 600;
                margin: 24px 0;
            }}
            .button:hover {{
                background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
            }}
            .footer {{
                text-align: center;
                margin-top: 32px;
                padding-top: 24px;
                border-top: 1px solid rgba(99, 102, 241, 0.2);
                color: #64748b;
                font-size: 14px;
            }}
            .code {{
                background: rgba(99, 102, 241, 0.1);
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 8px;
                padding: 16px;
                font-size: 24px;
                font-weight: 700;
                text-align: center;
                letter-spacing: 4px;
                color: #6366f1;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">💎</div>
                <div class="brand">Easy Data</div>
            </div>
            <div class="content">
                {content}
            </div>
            <div class="footer">
                © {datetime.now().year} Easy Data. {"جميع الحقوق محفوظة" if lang == "ar" else "All rights reserved."}
                <br><br>
                {"إذا لم تطلب هذا البريد، يمكنك تجاهله" if lang == "ar" else "If you didn't request this email, you can ignore it."}
            </div>
        </div>
    </body>
    </html>
    """


# ============================================
# Email Types
# ============================================

def send_verification_email(
    to_email: str,
    name: str,
    verification_url: str,
    lang: str = "ar"
) -> Tuple[bool, str]:
    """Send email verification email"""
    
    if lang == "ar":
        subject = "تأكيد بريدك الإلكتروني - Easy Data"
        content = f"""
        <h2>مرحباً {name}! 👋</h2>
        <p>شكراً لتسجيلك في Easy Data. لتفعيل حسابك، يرجى النقر على الزر أدناه:</p>
        <p style="text-align: center;">
            <a href="{verification_url}" class="button">تأكيد البريد الإلكتروني ✓</a>
        </p>
        <p>أو انسخ الرابط التالي والصقه في متصفحك:</p>
        <p style="word-break: break-all; color: #6366f1;">{verification_url}</p>
        <p><strong>ملاحظة:</strong> هذا الرابط صالح لمدة 24 ساعة فقط.</p>
        """
    else:
        subject = "Verify Your Email - Easy Data"
        content = f"""
        <h2>Hello {name}! 👋</h2>
        <p>Thank you for signing up for Easy Data. To activate your account, please click the button below:</p>
        <p style="text-align: center;">
            <a href="{verification_url}" class="button">Verify Email ✓</a>
        </p>
        <p>Or copy and paste this link in your browser:</p>
        <p style="word-break: break-all; color: #6366f1;">{verification_url}</p>
        <p><strong>Note:</strong> This link is valid for 24 hours only.</p>
        """
    
    html = get_base_template(content, lang)
    return send_email(to_email, subject, html)


def send_password_reset_email(
    to_email: str,
    name: str,
    reset_url: str,
    lang: str = "ar"
) -> Tuple[bool, str]:
    """Send password reset email"""
    
    if lang == "ar":
        subject = "إعادة تعيين كلمة المرور - Easy Data"
        content = f"""
        <h2>مرحباً {name}</h2>
        <p>تلقينا طلباً لإعادة تعيين كلمة المرور الخاصة بك. انقر على الزر أدناه لإنشاء كلمة مرور جديدة:</p>
        <p style="text-align: center;">
            <a href="{reset_url}" class="button">إعادة تعيين كلمة المرور 🔐</a>
        </p>
        <p>أو انسخ الرابط التالي:</p>
        <p style="word-break: break-all; color: #6366f1;">{reset_url}</p>
        <p><strong>ملاحظة:</strong> هذا الرابط صالح لمدة ساعة واحدة فقط.</p>
        <p>إذا لم تطلب إعادة تعيين كلمة المرور، يرجى تجاهل هذا البريد.</p>
        """
    else:
        subject = "Reset Your Password - Easy Data"
        content = f"""
        <h2>Hello {name}</h2>
        <p>We received a request to reset your password. Click the button below to create a new password:</p>
        <p style="text-align: center;">
            <a href="{reset_url}" class="button">Reset Password 🔐</a>
        </p>
        <p>Or copy and paste this link:</p>
        <p style="word-break: break-all; color: #6366f1;">{reset_url}</p>
        <p><strong>Note:</strong> This link is valid for 1 hour only.</p>
        <p>If you didn't request a password reset, please ignore this email.</p>
        """
    
    html = get_base_template(content, lang)
    return send_email(to_email, subject, html)


def send_welcome_email(
    to_email: str,
    name: str,
    app_url: str,
    lang: str = "ar"
) -> Tuple[bool, str]:
    """Send welcome email after verification"""
    
    if lang == "ar":
        subject = "مرحباً بك في Easy Data! 🎉"
        content = f"""
        <h2>أهلاً {name}! 🎉</h2>
        <p>تم تفعيل حسابك بنجاح. أنت الآن جاهز لاستخدام Easy Data!</p>
        
        <h3>🚀 ابدأ الآن:</h3>
        <ul>
            <li>📤 ارفع ملف البيانات الخاص بك (CSV, Excel, JSON)</li>
            <li>🎯 حدد العمود الهدف للتنبؤ</li>
            <li>🤖 دع الذكاء الاصطناعي يقوم بالباقي!</li>
        </ul>
        
        <p style="text-align: center;">
            <a href="{app_url}" class="button">ابدأ التحليل الآن 🚀</a>
        </p>
        
        <h3>✨ خطتك الحالية: مجاني</h3>
        <p>لديك 3 تحليلات يومياً. للحصول على تحليلات غير محدودة، قم بالترقية إلى Pro!</p>
        """
    else:
        subject = "Welcome to Easy Data! 🎉"
        content = f"""
        <h2>Welcome {name}! 🎉</h2>
        <p>Your account has been activated. You're now ready to use Easy Data!</p>
        
        <h3>🚀 Get Started:</h3>
        <ul>
            <li>📤 Upload your data file (CSV, Excel, JSON)</li>
            <li>🎯 Select the target column for prediction</li>
            <li>🤖 Let AI do the rest!</li>
        </ul>
        
        <p style="text-align: center;">
            <a href="{app_url}" class="button">Start Analysis Now 🚀</a>
        </p>
        
        <h3>✨ Your Current Plan: Free</h3>
        <p>You have 3 analyses per day. Upgrade to Pro for unlimited analyses!</p>
        """
    
    html = get_base_template(content, lang)
    return send_email(to_email, subject, html)


def send_subscription_confirmation(
    to_email: str,
    name: str,
    plan: str,
    amount: str,
    next_billing_date: str,
    lang: str = "ar"
) -> Tuple[bool, str]:
    """Send subscription confirmation email"""
    
    plan_names = {
        "pro": ("احترافي ⭐", "Professional ⭐"),
        "enterprise": ("مؤسسي 👑", "Enterprise 👑"),
    }
    plan_display = plan_names.get(plan, ("Pro", "Pro"))[0 if lang == "ar" else 1]
    
    if lang == "ar":
        subject = f"تم تفعيل اشتراكك - {plan_display}"
        content = f"""
        <h2>🎉 شكراً لك {name}!</h2>
        <p>تم تفعيل اشتراكك بنجاح. إليك تفاصيل اشتراكك:</p>
        
        <div style="background: rgba(99, 102, 241, 0.1); border-radius: 12px; padding: 20px; margin: 20px 0;">
            <p><strong>الخطة:</strong> {plan_display}</p>
            <p><strong>المبلغ:</strong> {amount}</p>
            <p><strong>تاريخ التجديد:</strong> {next_billing_date}</p>
        </div>
        
        <h3>🎁 ميزاتك الجديدة:</h3>
        <ul>
            <li>✅ تحليلات غير محدودة</li>
            <li>✅ ملفات حتى 100MB</li>
            <li>✅ جميع أنواع التقارير</li>
            <li>✅ تصدير Power BI</li>
            <li>✅ وصول API</li>
        </ul>
        
        <p>استمتع بتجربة Easy Data الكاملة!</p>
        """
    else:
        subject = f"Your Subscription is Active - {plan_display}"
        content = f"""
        <h2>🎉 Thank you {name}!</h2>
        <p>Your subscription has been activated successfully. Here are your subscription details:</p>
        
        <div style="background: rgba(99, 102, 241, 0.1); border-radius: 12px; padding: 20px; margin: 20px 0;">
            <p><strong>Plan:</strong> {plan_display}</p>
            <p><strong>Amount:</strong> {amount}</p>
            <p><strong>Next Billing Date:</strong> {next_billing_date}</p>
        </div>
        
        <h3>🎁 Your New Features:</h3>
        <ul>
            <li>✅ Unlimited analyses</li>
            <li>✅ Files up to 100MB</li>
            <li>✅ All report types</li>
            <li>✅ Power BI export</li>
            <li>✅ API access</li>
        </ul>
        
        <p>Enjoy the full Easy Data experience!</p>
        """
    
    html = get_base_template(content, lang)
    return send_email(to_email, subject, html)


def send_subscription_ending_reminder(
    to_email: str,
    name: str,
    end_date: str,
    renew_url: str,
    lang: str = "ar"
) -> Tuple[bool, str]:
    """Send reminder that subscription is ending soon"""
    
    if lang == "ar":
        subject = "⚠️ اشتراكك على وشك الانتهاء - Easy Data"
        content = f"""
        <h2>مرحباً {name}</h2>
        <p>نود إعلامك بأن اشتراكك في Easy Data سينتهي في <strong>{end_date}</strong>.</p>
        
        <p>للاستمرار في الاستمتاع بجميع الميزات:</p>
        <ul>
            <li>تحليلات غير محدودة</li>
            <li>جميع أنواع التقارير</li>
            <li>تصدير Power BI</li>
        </ul>
        
        <p style="text-align: center;">
            <a href="{renew_url}" class="button">تجديد الاشتراك الآن 🔄</a>
        </p>
        
        <p>إذا كان لديك أي أسئلة، لا تتردد في التواصل معنا!</p>
        """
    else:
        subject = "⚠️ Your Subscription is Ending Soon - Easy Data"
        content = f"""
        <h2>Hello {name}</h2>
        <p>We wanted to let you know that your Easy Data subscription will end on <strong>{end_date}</strong>.</p>
        
        <p>To continue enjoying all features:</p>
        <ul>
            <li>Unlimited analyses</li>
            <li>All report types</li>
            <li>Power BI export</li>
        </ul>
        
        <p style="text-align: center;">
            <a href="{renew_url}" class="button">Renew Subscription Now 🔄</a>
        </p>
        
        <p>If you have any questions, feel free to contact us!</p>
        """
    
    html = get_base_template(content, lang)
    return send_email(to_email, subject, html)


def send_payment_failed_email(
    to_email: str,
    name: str,
    update_payment_url: str,
    lang: str = "ar"
) -> Tuple[bool, str]:
    """Send notification that payment failed"""
    
    if lang == "ar":
        subject = "⚠️ فشل في الدفع - يرجى تحديث معلومات الدفع"
        content = f"""
        <h2>مرحباً {name}</h2>
        <p>للأسف، لم نتمكن من معالجة دفعتك الأخيرة.</p>
        
        <p>لتجنب فقدان الوصول إلى ميزات Pro، يرجى تحديث معلومات الدفع الخاصة بك:</p>
        
        <p style="text-align: center;">
            <a href="{update_payment_url}" class="button">تحديث معلومات الدفع 💳</a>
        </p>
        
        <p>إذا كنت تعتقد أن هذا خطأ، يرجى التواصل مع فريق الدعم.</p>
        """
    else:
        subject = "⚠️ Payment Failed - Please Update Payment Info"
        content = f"""
        <h2>Hello {name}</h2>
        <p>Unfortunately, we couldn't process your latest payment.</p>
        
        <p>To avoid losing access to Pro features, please update your payment information:</p>
        
        <p style="text-align: center;">
            <a href="{update_payment_url}" class="button">Update Payment Info 💳</a>
        </p>
        
        <p>If you believe this is an error, please contact our support team.</p>
        """
    
    html = get_base_template(content, lang)
    return send_email(to_email, subject, html)


# ============================================
# Verification Code (Alternative to Link)
# ============================================

def generate_verification_code() -> str:
    """Generate a 6-digit verification code"""
    return ''.join([str(secrets.randbelow(10)) for _ in range(6)])


def send_verification_code_email(
    to_email: str,
    name: str,
    code: str,
    lang: str = "ar"
) -> Tuple[bool, str]:
    """Send verification code email"""
    
    if lang == "ar":
        subject = "رمز التحقق الخاص بك - Easy Data"
        content = f"""
        <h2>مرحباً {name}!</h2>
        <p>رمز التحقق الخاص بك هو:</p>
        <div class="code">{code}</div>
        <p>أدخل هذا الرمز في التطبيق لتأكيد بريدك الإلكتروني.</p>
        <p><strong>ملاحظة:</strong> هذا الرمز صالح لمدة 15 دقيقة فقط.</p>
        """
    else:
        subject = "Your Verification Code - Easy Data"
        content = f"""
        <h2>Hello {name}!</h2>
        <p>Your verification code is:</p>
        <div class="code">{code}</div>
        <p>Enter this code in the app to verify your email.</p>
        <p><strong>Note:</strong> This code is valid for 15 minutes only.</p>
        """
    
    html = get_base_template(content, lang)
    return send_email(to_email, subject, html)
