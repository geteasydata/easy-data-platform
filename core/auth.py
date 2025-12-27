"""
Authentication Module for Easy Data
Handles user login, registration, and session management
Now integrated with Supabase, Stripe, and Email services
"""

import streamlit as st
import yaml
import bcrypt
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import json
import secrets

# Import new infrastructure modules
try:
    from core import database as db
    from core import email_service as email
    from core import payment
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False

# Path to config files (fallback for local mode)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'auth_config.yaml')
USERS_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'user_data', 'users.json')
TOKENS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'user_data', 'tokens.json')


def ensure_user_data_dir():
    """Create user_data directory if it doesn't exist"""
    user_data_dir = os.path.dirname(USERS_DB_PATH)
    if not os.path.exists(user_data_dir):
        os.makedirs(user_data_dir)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except:
        return False


def load_users() -> Dict:
    """Load users from JSON database"""
    ensure_user_data_dir()
    if os.path.exists(USERS_DB_PATH):
        with open(USERS_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"users": {}}


def save_users(users_data: Dict):
    """Save users to JSON database"""
    ensure_user_data_dir()
    with open(USERS_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(users_data, f, ensure_ascii=False, indent=2)


def register_user(username: str, email_addr: str, password: str, name: str) -> Tuple[bool, str]:
    """
    Register a new user
    Returns: (success, message)
    """
    # Try Supabase first if available
    if INFRASTRUCTURE_AVAILABLE and db.is_configured():
        success, message, user_data = db.create_user(username, email_addr, password, name)
        
        # Send welcome email if registration succeeded
        if success and user_data:
            try:
                from core import email_service
                email_service.send_welcome_email(
                    to_email=email_addr,
                    name=name,
                    app_url="http://localhost:8501", # Will be updated for production
                    lang="ar" if "ar" in message else "en"
                )
            except Exception as e:
                # Don't block registration if email fails
                print(f"Failed to send welcome email: {str(e)}")
                
        return success, message

    users_data = load_users()
    
    # Check if username exists
    if username.lower() in [u.lower() for u in users_data["users"].keys()]:
        return False, "اسم المستخدم موجود بالفعل | Username already exists"
    
    # Check if email exists
    for user in users_data["users"].values():
        if user.get("email", "").lower() == email_addr.lower():
            return False, "البريد الإلكتروني مستخدم بالفعل | Email already in use"
    
    # Create new user
    users_data["users"][username] = {
        "email": email_addr,
        "name": name,
        "password": hash_password(password),
        "plan": "free",
        "role": "user",
        "created_at": datetime.now().isoformat(),
        "usage_today": 0,
        "last_usage_date": datetime.now().strftime("%Y-%m-%d")
    }
    
    save_users(users_data)
    return True, "تم إنشاء الحساب بنجاح! | Account created successfully!"


def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[Dict], str]:
    """
    Authenticate a user
    Returns: (success, user_data, message)
    """
    # Try Supabase first if available
    if INFRASTRUCTURE_AVAILABLE and db.is_configured():
        return db.authenticate_user(username, password)

    users_data = load_users()
    
    # Find user (case-insensitive)
    user = None
    actual_username = None
    for uname, udata in users_data["users"].items():
        if uname.lower() == username.lower():
            user = udata
            actual_username = uname
            break
    
    if not user:
        return False, None, "اسم المستخدم غير موجود | Username not found"
    
    if not verify_password(password, user["password"]):
        return False, None, "كلمة المرور غير صحيحة | Incorrect password"
    
    # Update last login
    user["last_login"] = datetime.now().isoformat()
    users_data["users"][actual_username] = user
    save_users(users_data)
    
    return True, user, "تم تسجيل الدخول بنجاح | Login successful"


def check_usage_limit(username: str) -> Tuple[bool, int, int]:
    """
    Check if user has exceeded daily usage limit
    Returns: (can_use, used_today, daily_limit)
    """
    users_data = load_users()
    user = users_data["users"].get(username)
    
    if not user:
        return False, 0, 0
    
    # Define limits by plan
    limits = {
        "free": 3,
        "pro": 999999,  # Unlimited
        "enterprise": 999999
    }
    
    plan = user.get("plan", "free")
    daily_limit = limits.get(plan, 3)
    
    # Reset counter if new day
    today = datetime.now().strftime("%Y-%m-%d")
    if user.get("last_usage_date") != today:
        user["usage_today"] = 0
        user["last_usage_date"] = today
        users_data["users"][username] = user
        save_users(users_data)
    
    used_today = user.get("usage_today", 0)
    can_use = used_today < daily_limit
    
    return can_use, used_today, daily_limit


def increment_usage(username: str):
    """Increment user's daily usage counter"""
    users_data = load_users()
    if username in users_data["users"]:
        users_data["users"][username]["usage_today"] = \
            users_data["users"][username].get("usage_today", 0) + 1
        save_users(users_data)


def get_user_plan(username: str) -> str:
    """Get user's subscription plan"""
    users_data = load_users()
    user = users_data["users"].get(username)
    return user.get("plan", "free") if user else "free"


def upgrade_plan(username: str, new_plan: str) -> bool:
    """Upgrade user's plan"""
    users_data = load_users()
    if username in users_data["users"]:
        users_data["users"][username]["plan"] = new_plan
        save_users(users_data)
        return True
    return False



def show_login_page(lang: str = "ar") -> Optional[Dict]:
    """
    Display login/register page - Original Easy Data Design
    Returns user data if logged in, None otherwise
    """
    
    # Header with Logo (Styled via assets/style.css)
    # Removed header as it is now part of the main app layout
    pass
    
    # Message handling
    if "login_message" not in st.session_state:
        st.session_state.login_message = None

    if st.session_state.login_message:
        success, msg = st.session_state.login_message
        if success:
            st.success(msg)
            st.session_state.login_message = None
            st.rerun()
        else:
            st.error(msg)
            st.session_state.login_message = None
    
    # Tabs for Login/Register
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        # Login Form
        with st.form("login_form_original"):
            username = st.text_input("Username", placeholder="username", key="login_username_orig")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="login_password_orig")
            
            submitted = st.form_submit_button("🔑 Login", use_container_width=True)
            
            if submitted:
                if username and password:
                    success, user_data, msg = authenticate_user(username, password)
                    if success:
                        st.session_state.update({
                            "authenticated": True,
                            "username": username,
                            "user_data": user_data
                        })
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please enter username and password" if lang == 'en' else "يرجى إدخال اسم المستخدم وكلمة المرور")
    
    with tab2:
        # Register Form
        with st.form("register_form_original"):
            reg_username = st.text_input("Username", placeholder="Choose a username", key="reg_username_orig")
            reg_email = st.text_input("Email", placeholder="your@email.com", key="reg_email_orig")
            reg_password = st.text_input("Password", type="password", placeholder="••••••••", key="reg_password_orig")
            reg_name = st.text_input("Full Name", placeholder="Your full name", key="reg_name_orig")
            
            submitted = st.form_submit_button("📝 Register", use_container_width=True)
            
            if submitted:
                if all([reg_username, reg_email, reg_password, reg_name]):
                    success, msg = register_user(reg_username, reg_email, reg_password, reg_name)
                    if success:
                        st.success(msg)
                        st.balloons()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please fill all fields" if lang == 'en' else "يرجى ملء جميع الحقول")
    
    # Guest Access Removed
    pass
    
    # Language toggle
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        lang_label = "🌐 العربية" if lang == 'en' else "🌐 English"
        if st.button(lang_label, key="lang_toggle_orig", use_container_width=True):
            st.session_state.lang = 'ar' if lang == 'en' else 'en'
            st.rerun()

    return None


def show_user_info(lang: str = "ar"):
    """Display user info in sidebar"""
    if st.session_state.get("authenticated", False):
        user = st.session_state.get("user_data", {})
        username = st.session_state.get("username", "Guest")
        
        # Plan badge colors
        plan_colors = {
            "free": "#6b7280",
            "pro": "#6366f1",
            "enterprise": "#8b5cf6"
        }
        plan = user.get("plan", "free")
        plan_color = plan_colors.get(plan, "#6b7280")
        plan_display = {"free": "Free", "pro": "Pro ⭐", "enterprise": "Enterprise 👑"}
        
        st.sidebar.markdown(f"""
        <div style="
            padding: 1rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
            border-radius: 12px;
            border: 1px solid rgba(99, 102, 241, 0.2);
            margin-bottom: 1rem;
        ">
            <div style="font-size: 1.1rem; font-weight: 600; color: #e2e8f0;">
                👤 {user.get('name', username)}
            </div>
            <div style="
                display: inline-block;
                padding: 0.25rem 0.75rem;
                background: {plan_color};
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 600;
                color: white;
                margin-top: 0.5rem;
            ">
                {plan_display.get(plan, "Free")}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Usage info for free users
        if plan == "free" and username != "guest":
            can_use, used, limit = check_usage_limit(username)
            remaining = max(0, limit - used)
            
            if lang == "ar":
                st.sidebar.info(f"📊 التحليلات المتبقية اليوم: {remaining}/{limit}")
            else:
                st.sidebar.info(f"📊 Analyses remaining today: {remaining}/{limit}")
            
            if not can_use:
                if lang == "ar":
                    st.sidebar.warning("⚠️ وصلت للحد اليومي. ترقية للـ Pro للمزيد!")
                else:
                    st.sidebar.warning("⚠️ Daily limit reached. Upgrade to Pro!")
        
        # Logout button
        if st.sidebar.button("🚪 تسجيل الخروج | Logout" if lang == "ar" else "🚪 Logout", use_container_width=True):
            for key in ["authenticated", "username", "user_data"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def require_auth(lang: str = "ar"):
    """
    Decorator-like function to require authentication
    Call at the beginning of your main function
    Returns True if authenticated, False otherwise
    """
    if not st.session_state.get("authenticated", False):
        show_login_page(lang)
        return False
    return True


def can_access_feature(feature: str) -> bool:
    """
    Check if current user can access a specific feature
    Features: 'pdf_report', 'word_report', 'api', 'unlimited_analysis', 'priority_support'
    """
    user = st.session_state.get("user_data", {})
    plan = user.get("plan", "free")
    
    feature_access = {
        "pdf_report": ["pro", "enterprise"],
        "word_report": ["pro", "enterprise"],
        "api": ["enterprise"],
        "unlimited_analysis": ["pro", "enterprise"],
        "priority_support": ["enterprise"],
        "notebook_export": ["pro", "enterprise"],
        "powerbi_export": ["pro", "enterprise"],
    }
    
    allowed_plans = feature_access.get(feature, [])
    return plan in allowed_plans or plan == "admin"


# ============================================
# New Infrastructure Integration
# ============================================

def register_user_v2(username: str, email: str, password: str, name: str, lang: str = "ar") -> Tuple[bool, str]:
    """
    Register a new user with email verification
    Uses Supabase if available, falls back to local JSON
    """
    if INFRASTRUCTURE_AVAILABLE and db.is_configured():
        # Use Supabase
        success, message, user_data = db.create_user(username, email, password, name)
        
        if success and email.is_configured():
            # Generate verification token
            token = secrets.token_urlsafe(32)
            save_verification_token(email, token, "verify")
            
            # Send verification email
            app_url = get_app_url()
            verify_url = f"{app_url}?verify={token}"
            email.send_verification_email(email, name, verify_url, lang)
        
        return success, message
    else:
        # Fallback to original local registration
        return register_user(username, email, password, name)


def save_verification_token(email: str, token: str, token_type: str):
    """Save verification/reset token"""
    ensure_user_data_dir()
    
    tokens = {}
    if os.path.exists(TOKENS_PATH):
        with open(TOKENS_PATH, 'r', encoding='utf-8') as f:
            tokens = json.load(f)
    
    tokens[token] = {
        "email": email,
        "type": token_type,
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=24 if token_type == "verify" else 1)).isoformat()
    }
    
    with open(TOKENS_PATH, 'w', encoding='utf-8') as f:
        json.dump(tokens, f)


def verify_token(token: str) -> Tuple[bool, str, str]:
    """
    Verify a token
    Returns: (valid, email, token_type)
    """
    if not os.path.exists(TOKENS_PATH):
        return False, "", ""
    
    with open(TOKENS_PATH, 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    
    if token not in tokens:
        return False, "", ""
    
    token_data = tokens[token]
    expires_at = datetime.fromisoformat(token_data["expires_at"])
    
    if datetime.now() > expires_at:
        # Token expired, remove it
        del tokens[token]
        with open(TOKENS_PATH, 'w', encoding='utf-8') as f:
            json.dump(tokens, f)
        return False, "", ""
    
    return True, token_data["email"], token_data["type"]


def consume_token(token: str) -> bool:
    """Remove a used token"""
    if not os.path.exists(TOKENS_PATH):
        return False
    
    with open(TOKENS_PATH, 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    
    if token in tokens:
        del tokens[token]
        with open(TOKENS_PATH, 'w', encoding='utf-8') as f:
            json.dump(tokens, f)
        return True
    return False


def request_password_reset(email_address: str, lang: str = "ar") -> Tuple[bool, str]:
    """
    Request password reset
    Sends reset email if user exists
    """
    if INFRASTRUCTURE_AVAILABLE and db.is_configured():
        user = db.get_user(email=email_address)
    else:
        user = None
        users_data = load_users()
        for uname, udata in users_data["users"].items():
            if udata.get("email", "").lower() == email_address.lower():
                user = {**udata, "username": uname, "name": udata.get("name", uname)}
                break
    
    if not user:
        # Don't reveal if email exists
        return True, "إذا كان البريد مسجلاً، ستتلقى رابط إعادة التعيين | If the email is registered, you'll receive a reset link"
    
    if INFRASTRUCTURE_AVAILABLE and email.is_configured():
        token = secrets.token_urlsafe(32)
        save_verification_token(email_address, token, "reset")
        
        app_url = get_app_url()
        reset_url = f"{app_url}?reset={token}"
        email.send_password_reset_email(email_address, user.get("name", ""), reset_url, lang)
    
    return True, "إذا كان البريد مسجلاً، ستتلقى رابط إعادة التعيين | If the email is registered, you'll receive a reset link"


def reset_password(token: str, new_password: str) -> Tuple[bool, str]:
    """Reset password using token"""
    valid, email_address, token_type = verify_token(token)
    
    if not valid or token_type != "reset":
        return False, "رابط غير صالح أو منتهي الصلاحية | Invalid or expired link"
    
    if len(new_password) < 6:
        return False, "كلمة المرور قصيرة جداً | Password too short"
    
    if INFRASTRUCTURE_AVAILABLE and db.is_configured():
        user = db.get_user(email=email_address)
        if user:
            db.update_password(user["id"], new_password)
    else:
        users_data = load_users()
        for uname, udata in users_data["users"].items():
            if udata.get("email", "").lower() == email_address.lower():
                users_data["users"][uname]["password"] = hash_password(new_password)
                save_users(users_data)
                break
    
    consume_token(token)
    return True, "تم تغيير كلمة المرور بنجاح | Password changed successfully"


def verify_email(token: str) -> Tuple[bool, str]:
    """Verify email using token"""
    valid, email_address, token_type = verify_token(token)
    
    if not valid or token_type != "verify":
        return False, "رابط غير صالح أو منتهي الصلاحية | Invalid or expired link"
    
    if INFRASTRUCTURE_AVAILABLE and db.is_configured():
        user = db.get_user(email=email_address)
        if user:
            db.verify_email(user["id"])
    else:
        users_data = load_users()
        for uname, udata in users_data["users"].items():
            if udata.get("email", "").lower() == email_address.lower():
                users_data["users"][uname]["email_verified"] = True
                save_users(users_data)
                break
    
    consume_token(token)
    return True, "تم تأكيد البريد بنجاح! | Email verified successfully!"


def get_app_url() -> str:
    """Get the application URL"""
    try:
        # Try to get from Streamlit secrets
        return st.secrets.get("APP_URL", "https://geteasydata.streamlit.app")
    except:
        return os.environ.get("APP_URL", "https://geteasydata.streamlit.app")


def get_stripe_customer_id(username: str) -> Optional[str]:
    """Get or create Stripe customer ID for user"""
    if not INFRASTRUCTURE_AVAILABLE:
        return None
    
    if db.is_configured():
        user = db.get_user(username=username)
        if user and user.get("stripe_customer_id"):
            return user["stripe_customer_id"]
        
        # Create new Stripe customer
        if user and payment.is_stripe_configured():
            customer_id = payment.create_customer(
                email=user.get("email", ""),
                name=user.get("name", username),
                metadata={"username": username}
            )
            if customer_id:
                db.update_user(user["id"], {"stripe_customer_id": customer_id})
                return customer_id
    
    return None


def show_upgrade_page(lang: str = "ar"):
    """Show upgrade page with pricing and checkout"""
    st.markdown("## " + ("🚀 ترقية حسابك" if lang == "ar" else "🚀 Upgrade Your Account"))
    
    if not INFRASTRUCTURE_AVAILABLE or not payment.is_stripe_configured():
        st.warning("نظام الدفع غير مفعل حالياً" if lang == "ar" else "Payment system is not currently active")
        return
    
    # Show pricing cards
    payment.show_pricing_cards(lang)
    
    # Handle upgrade
    username = st.session_state.get("username")
    if username and st.session_state.get("show_upgrade_modal"):
        customer_id = get_stripe_customer_id(username)
        if customer_id:
            app_url = get_app_url()
            payment.show_upgrade_modal(
                customer_id=customer_id,
                success_url=f"{app_url}?payment=success",
                cancel_url=f"{app_url}?payment=cancelled",
                lang=lang
            )


def handle_payment_callback():
    """Handle payment success/cancel callbacks"""
    if "payment" in st.query_params:
        status = st.query_params["payment"]
        
        if status == "success":
            st.success("🎉 تم الدفع بنجاح! تم ترقية حسابك | Payment successful! Your account has been upgraded")
            
            # Refresh user data
            username = st.session_state.get("username")
            if username and INFRASTRUCTURE_AVAILABLE and payment.is_stripe_configured():
                customer_id = get_stripe_customer_id(username)
                if customer_id:
                    sub_status = payment.get_subscription_status(customer_id)
                    if sub_status["plan"] != "free":
                        upgrade_plan(username, sub_status["plan"])
                        st.session_state["user_data"]["plan"] = sub_status["plan"]
        
        elif status == "cancelled":
            st.info("تم إلغاء الدفع | Payment was cancelled")
        
        # Clear query params
        st.query_params.clear()


def handle_verification_callback():
    """Handle email verification callbacks"""
    if "verify" in st.query_params:
        token = st.query_params["verify"]
        success, message = verify_email(token)
        
        if success:
            st.success(message)
        else:
            st.error(message)
        
        st.query_params.clear()
    
    if "reset" in st.query_params:
        token = st.query_params["reset"]
        valid, email_address, token_type = verify_token(token)
        
        if valid and token_type == "reset":
            st.session_state["reset_token"] = token
            st.session_state["show_reset_form"] = True
        else:
            st.error("رابط غير صالح أو منتهي الصلاحية | Invalid or expired link")
        
        st.query_params.clear()


def show_reset_password_form(lang: str = "ar"):
    """Show password reset form"""
    if not st.session_state.get("show_reset_form"):
        return
    
    st.markdown("### 🔐 " + ("إعادة تعيين كلمة المرور" if lang == "ar" else "Reset Password"))
    
    with st.form("reset_password_form"):
        new_password = st.text_input(
            "كلمة المرور الجديدة" if lang == "ar" else "New Password",
            type="password"
        )
        confirm_password = st.text_input(
            "تأكيد كلمة المرور" if lang == "ar" else "Confirm Password",
            type="password"
        )
        
        submit = st.form_submit_button(
            "تغيير كلمة المرور" if lang == "ar" else "Change Password",
            use_container_width=True
        )
        
        if submit:
            if new_password != confirm_password:
                st.error("كلمات المرور غير متطابقة | Passwords don't match")
            elif len(new_password) < 6:
                st.error("كلمة المرور قصيرة | Password too short")
            else:
                token = st.session_state.get("reset_token")
                success, message = reset_password(token, new_password)
                
                if success:
                    st.success(message)
                    st.session_state["show_reset_form"] = False
                    st.session_state["reset_token"] = None
                else:
                    st.error(message)


def show_forgot_password_link(lang: str = "ar"):
    """Show forgot password link in login form"""
    if st.button("🔑 " + ("نسيت كلمة المرور؟" if lang == "ar" else "Forgot Password?"), key="forgot_password"):
        st.session_state["show_forgot_password"] = True


def show_forgot_password_form(lang: str = "ar"):
    """Show forgot password request form"""
    if not st.session_state.get("show_forgot_password"):
        return
    
    st.markdown("### 📧 " + ("استعادة كلمة المرور" if lang == "ar" else "Password Recovery"))
    
    with st.form("forgot_password_form"):
        email_input = st.text_input(
            "البريد الإلكتروني" if lang == "ar" else "Email",
            placeholder="email@example.com"
        )
        
        submit = st.form_submit_button(
            "إرسال رابط الاستعادة" if lang == "ar" else "Send Reset Link",
            use_container_width=True
        )
        
        if submit:
            if email_input:
                success, message = request_password_reset(email_input, lang)
                st.info(message)
                st.session_state["show_forgot_password"] = False
            else:
                st.warning("الرجاء إدخال البريد الإلكتروني | Please enter email")
    
    if st.button("◀ " + ("رجوع" if lang == "ar" else "Back")):
        st.session_state["show_forgot_password"] = False
