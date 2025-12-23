"""
Authentication Module for Easy Data
Handles user login, registration, and session management
"""

import streamlit as st
import yaml
import bcrypt
import os
from datetime import datetime
from typing import Dict, Optional, Tuple
import json

# Path to config files
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'auth_config.yaml')
USERS_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'user_data', 'users.json')


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


def register_user(username: str, email: str, password: str, name: str) -> Tuple[bool, str]:
    """
    Register a new user
    Returns: (success, message)
    """
    users_data = load_users()
    
    # Check if username exists
    if username.lower() in [u.lower() for u in users_data["users"].keys()]:
        return False, "اسم المستخدم موجود بالفعل | Username already exists"
    
    # Check if email exists
    for user in users_data["users"].values():
        if user.get("email", "").lower() == email.lower():
            return False, "البريد الإلكتروني مستخدم بالفعل | Email already in use"
    
    # Create new user
    users_data["users"][username] = {
        "email": email,
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
    Display login/register page
    Returns user data if logged in, None otherwise
    """
    
    # Check if already logged in
    if st.session_state.get("authenticated", False):
        return st.session_state.get("user_data")
    
    # Custom CSS for auth page
    st.markdown("""
    <style>
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
        border-radius: 20px;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .auth-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .auth-subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="auth-title">💎 Easy Data</div>', unsafe_allow_html=True)
    
    if lang == "ar":
        st.markdown('<div class="auth-subtitle">منصة تحليل البيانات بالذكاء الاصطناعي</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="auth-subtitle">AI-Powered Data Analysis Platform</div>', unsafe_allow_html=True)
    
    # Tabs for Login/Register
    if lang == "ar":
        tab1, tab2 = st.tabs(["🔐 تسجيل الدخول", "📝 حساب جديد"])
    else:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        # Login Form
        with st.form("login_form"):
            username = st.text_input(
                "اسم المستخدم | Username" if lang == "ar" else "Username",
                placeholder="username"
            )
            password = st.text_input(
                "كلمة المرور | Password" if lang == "ar" else "Password",
                type="password",
                placeholder="••••••••"
            )
            
            submit = st.form_submit_button(
                "🚀 دخول | Login" if lang == "ar" else "🚀 Login",
                use_container_width=True
            )
            
            if submit:
                if username and password:
                    success, user_data, message = authenticate_user(username, password)
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        st.session_state["user_data"] = user_data
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("الرجاء إدخال جميع البيانات | Please fill all fields")
    
    with tab2:
        # Register Form
        with st.form("register_form"):
            new_username = st.text_input(
                "اسم المستخدم | Username" if lang == "ar" else "Username",
                placeholder="username",
                key="reg_username"
            )
            new_email = st.text_input(
                "البريد الإلكتروني | Email" if lang == "ar" else "Email",
                placeholder="email@example.com"
            )
            new_name = st.text_input(
                "الاسم الكامل | Full Name" if lang == "ar" else "Full Name",
                placeholder="Ahmed Mohamed"
            )
            new_password = st.text_input(
                "كلمة المرور | Password" if lang == "ar" else "Password",
                type="password",
                placeholder="••••••••",
                key="reg_password"
            )
            confirm_password = st.text_input(
                "تأكيد كلمة المرور | Confirm Password" if lang == "ar" else "Confirm Password",
                type="password",
                placeholder="••••••••"
            )
            
            register = st.form_submit_button(
                "📝 إنشاء حساب | Create Account" if lang == "ar" else "📝 Create Account",
                use_container_width=True
            )
            
            if register:
                if new_username and new_email and new_name and new_password:
                    if new_password != confirm_password:
                        st.error("كلمات المرور غير متطابقة | Passwords don't match")
                    elif len(new_password) < 6:
                        st.error("كلمة المرور قصيرة جداً (6 أحرف على الأقل) | Password too short (min 6 chars)")
                    else:
                        success, message = register_user(new_username, new_email, new_password, new_name)
                        if success:
                            st.success(message)
                            st.info("يمكنك الآن تسجيل الدخول | You can now login")
                        else:
                            st.error(message)
                else:
                    st.warning("الرجاء إدخال جميع البيانات | Please fill all fields")
    
    # Guest mode option
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("👤 تجربة كضيف | Try as Guest" if lang == "ar" else "👤 Try as Guest", use_container_width=True):
            st.session_state["authenticated"] = True
            st.session_state["username"] = "guest"
            st.session_state["user_data"] = {
                "name": "Guest",
                "plan": "free",
                "role": "guest"
            }
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
