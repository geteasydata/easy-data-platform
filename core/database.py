"""
Supabase Database Module for Easy Data Platform
Handles user management, subscriptions, and usage logging
"""

import os
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import json

# Supabase will be imported when available
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# For password hashing
import bcrypt


# ============================================
# Configuration
# ============================================

def get_supabase_config() -> Tuple[Optional[str], Optional[str]]:
    """Get Supabase configuration from secrets or environment"""
    url = None
    key = None
    
    # Try Streamlit secrets first
    try:
        import streamlit as st
        # Check top-level
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
        
        # Check [supabase] section
        if not url and "supabase" in st.secrets:
            url = st.secrets["supabase"].get("SUPABASE_URL")
            key = st.secrets["supabase"].get("SUPABASE_KEY")
            
    except:
        pass
    
    # Fall back to environment variables
    if not url:
        url = os.environ.get("SUPABASE_URL")
    if not key:
        key = os.environ.get("SUPABASE_KEY")
    
    return url, key


# Global client instance
_supabase_client: Optional[Client] = None


def get_client() -> Optional[Client]:
    """Get or create Supabase client"""
    global _supabase_client
    
    if _supabase_client:
        return _supabase_client
    
    if not SUPABASE_AVAILABLE:
        return None
    
    url, key = get_supabase_config()
    if not url or not key:
        return None
    
    try:
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception:
        return None


def is_configured() -> bool:
    """Check if Supabase is properly configured"""
    url, key = get_supabase_config()
    return bool(url and key and SUPABASE_AVAILABLE)


# ============================================
# Password Utilities
# ============================================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except:
        return False


# ============================================
# User Management
# ============================================

def create_user(
    username: str,
    email: str,
    password: str,
    name: str,
    plan: str = "free"
) -> Tuple[bool, str, Optional[Dict]]:
    """
    Create a new user
    Returns: (success, message, user_data)
    """
    client = get_client()
    
    if not client:
        # Fallback to local JSON storage
        return _create_user_local(username, email, password, name, plan)
    
    try:
        # Check if username exists
        existing = client.table("users").select("id").eq("username", username.lower()).execute()
        if existing.data:
            return False, "اسم المستخدم موجود بالفعل | Username already exists", None
        
        # Check if email exists
        existing = client.table("users").select("id").eq("email", email.lower()).execute()
        if existing.data:
            return False, "البريد الإلكتروني مستخدم بالفعل | Email already in use", None
        
        # Create user
        user_data = {
            "username": username.lower(),
            "email": email.lower(),
            "password_hash": hash_password(password),
            "name": name,
            "plan": plan,
            "email_verified": False,
            "created_at": datetime.now().isoformat(),
        }
        
        result = client.table("users").insert(user_data).execute()
        
        if result.data:
            return True, "تم إنشاء الحساب بنجاح! | Account created successfully!", result.data[0]
        else:
            return False, "فشل إنشاء الحساب | Failed to create account", None
            
    except Exception as e:
        return False, f"خطأ في قاعدة البيانات | Database error: {str(e)}", None


def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[Dict], str]:
    """
    Authenticate a user
    Returns: (success, user_data, message)
    """
    client = get_client()
    
    if not client:
        return _authenticate_user_local(username, password)
    
    try:
        # Find user
        result = client.table("users").select("*").eq("username", username.lower()).execute()
        
        if not result.data:
            # Try by email
            result = client.table("users").select("*").eq("email", username.lower()).execute()
        
        if not result.data:
            return False, None, "اسم المستخدم غير موجود | User not found"
        
        user = result.data[0]
        
        if not verify_password(password, user["password_hash"]):
            return False, None, "كلمة المرور غير صحيحة | Incorrect password"
        
        # Update last login
        client.table("users").update({
            "last_login": datetime.now().isoformat()
        }).eq("id", user["id"]).execute()
        
        # Remove password hash from returned data
        user.pop("password_hash", None)
        
        return True, user, "تم تسجيل الدخول بنجاح | Login successful"
        
    except Exception as e:
        return False, None, f"خطأ في قاعدة البيانات | Database error: {str(e)}"


def get_user(user_id: str = None, username: str = None, email: str = None) -> Optional[Dict]:
    """Get user by ID, username, or email"""
    client = get_client()
    
    if not client:
        return _get_user_local(username=username, email=email)
    
    try:
        query = client.table("users").select("*")
        
        if user_id:
            query = query.eq("id", user_id)
        elif username:
            query = query.eq("username", username.lower())
        elif email:
            query = query.eq("email", email.lower())
        else:
            return None
        
        result = query.execute()
        
        if result.data:
            user = result.data[0]
            user.pop("password_hash", None)
            return user
        return None
        
    except Exception:
        return None


def update_user(user_id: str, updates: Dict) -> bool:
    """Update user data"""
    client = get_client()
    
    if not client:
        return _update_user_local(user_id, updates)
    
    try:
        # Don't allow updating sensitive fields directly
        safe_updates = {k: v for k, v in updates.items() 
                       if k not in ["id", "password_hash", "created_at"]}
        
        result = client.table("users").update(safe_updates).eq("id", user_id).execute()
        return bool(result.data)
        
    except Exception:
        return False


def update_user_plan(user_id: str, plan: str, stripe_customer_id: str = None) -> bool:
    """Update user's subscription plan"""
    updates = {"plan": plan}
    if stripe_customer_id:
        updates["stripe_customer_id"] = stripe_customer_id
    return update_user(user_id, updates)


def verify_email(user_id: str) -> bool:
    """Mark user's email as verified"""
    return update_user(user_id, {"email_verified": True})


def update_password(user_id: str, new_password: str) -> bool:
    """Update user's password"""
    client = get_client()
    
    if not client:
        return False
    
    try:
        result = client.table("users").update({
            "password_hash": hash_password(new_password)
        }).eq("id", user_id).execute()
        return bool(result.data)
    except Exception:
        return False


# ============================================
# Usage Tracking
# ============================================

def log_usage(user_id: str, action: str, metadata: Dict = None) -> bool:
    """Log user action for analytics"""
    client = get_client()
    
    if not client:
        return False
    
    try:
        client.table("usage_logs").insert({
            "user_id": user_id,
            "action": action,
            "metadata": json.dumps(metadata) if metadata else None,
            "timestamp": datetime.now().isoformat(),
        }).execute()
        return True
    except Exception:
        return False


def get_usage_today(user_id: str) -> int:
    """Get user's usage count for today"""
    client = get_client()
    
    if not client:
        return _get_usage_today_local(user_id)
    
    try:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        result = client.table("usage_logs").select("id", count="exact").eq(
            "user_id", user_id
        ).gte("timestamp", today.isoformat()).execute()
        
        return result.count or 0
        
    except Exception:
        return 0


def check_usage_limit(user_id: str, plan: str) -> Tuple[bool, int, int]:
    """
    Check if user has exceeded daily usage limit
    Returns: (can_use, used_today, daily_limit)
    """
    limits = {
        "free": 3,
        "pro": 999999,
        "enterprise": 999999,
    }
    
    daily_limit = limits.get(plan, 3)
    used_today = get_usage_today(user_id)
    can_use = used_today < daily_limit
    
    return can_use, used_today, daily_limit


def increment_usage(user_id: str) -> bool:
    """Increment user's usage (log an analysis action)"""
    return log_usage(user_id, "analysis")


# ============================================
# Subscription Management
# ============================================

def save_subscription(
    user_id: str,
    stripe_subscription_id: str,
    status: str,
    current_period_end: datetime
) -> bool:
    """Save or update subscription record"""
    client = get_client()
    
    if not client:
        return False
    
    try:
        # Check if subscription exists
        existing = client.table("subscriptions").select("id").eq(
            "stripe_subscription_id", stripe_subscription_id
        ).execute()
        
        sub_data = {
            "user_id": user_id,
            "stripe_subscription_id": stripe_subscription_id,
            "status": status,
            "current_period_end": current_period_end.isoformat(),
        }
        
        if existing.data:
            client.table("subscriptions").update(sub_data).eq(
                "stripe_subscription_id", stripe_subscription_id
            ).execute()
        else:
            client.table("subscriptions").insert(sub_data).execute()
        
        return True
        
    except Exception:
        return False


def get_active_subscription(user_id: str) -> Optional[Dict]:
    """Get user's active subscription"""
    client = get_client()
    
    if not client:
        return None
    
    try:
        result = client.table("subscriptions").select("*").eq(
            "user_id", user_id
        ).eq("status", "active").execute()
        
        if result.data:
            return result.data[0]
        return None
        
    except Exception:
        return None


# ============================================
# Admin Functions
# ============================================

def get_all_users(limit: int = 100, offset: int = 0) -> List[Dict]:
    """Get all users (for admin panel)"""
    client = get_client()
    
    if not client:
        return []
    
    try:
        result = client.table("users").select(
            "id, username, email, name, plan, email_verified, created_at, last_login"
        ).order("created_at", desc=True).range(offset, offset + limit - 1).execute()
        
        return result.data or []
        
    except Exception:
        return []


def get_user_count() -> int:
    """Get total user count"""
    client = get_client()
    
    if not client:
        return 0
    
    try:
        result = client.table("users").select("id", count="exact").execute()
        return result.count or 0
    except Exception:
        return 0


def get_plan_distribution() -> Dict[str, int]:
    """Get count of users per plan"""
    client = get_client()
    
    if not client:
        return {}
    
    try:
        result = client.table("users").select("plan").execute()
        
        distribution = {"free": 0, "pro": 0, "enterprise": 0}
        for user in result.data or []:
            plan = user.get("plan", "free")
            distribution[plan] = distribution.get(plan, 0) + 1
        
        return distribution
        
    except Exception:
        return {}


def get_recent_signups(days: int = 7) -> int:
    """Get count of signups in the last N days"""
    client = get_client()
    
    if not client:
        return 0
    
    try:
        since = (datetime.now() - timedelta(days=days)).isoformat()
        result = client.table("users").select("id", count="exact").gte(
            "created_at", since
        ).execute()
        
        return result.count or 0
        
    except Exception:
        return 0


# ============================================
# Migration Functions
# ============================================

def migrate_from_json(json_path: str) -> Tuple[int, int]:
    """
    Migrate users from local JSON file to Supabase
    Returns: (success_count, error_count)
    """
    client = get_client()
    
    if not client:
        return 0, 0
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        users = data.get("users", {})
        success = 0
        errors = 0
        
        for username, user_data in users.items():
            try:
                # Check if already exists
                existing = client.table("users").select("id").eq(
                    "username", username.lower()
                ).execute()
                
                if existing.data:
                    continue
                
                client.table("users").insert({
                    "username": username.lower(),
                    "email": user_data.get("email", "").lower(),
                    "password_hash": user_data.get("password", ""),
                    "name": user_data.get("name", username),
                    "plan": user_data.get("plan", "free"),
                    "email_verified": True,  # Assume existing users are verified
                    "created_at": user_data.get("created_at", datetime.now().isoformat()),
                }).execute()
                
                success += 1
                
            except Exception:
                errors += 1
        
        return success, errors
        
    except Exception:
        return 0, 0


# ============================================
# Local Fallback (when Supabase is not available)
# ============================================

LOCAL_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'user_data', 'users.json')


def _ensure_local_db():
    """Ensure local database file exists"""
    os.makedirs(os.path.dirname(LOCAL_DB_PATH), exist_ok=True)
    if not os.path.exists(LOCAL_DB_PATH):
        with open(LOCAL_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump({"users": {}}, f)


def _load_local_db() -> Dict:
    """Load local database"""
    _ensure_local_db()
    try:
        with open(LOCAL_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {"users": {}}


def _save_local_db(data: Dict):
    """Save local database"""
    _ensure_local_db()
    with open(LOCAL_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _create_user_local(username, email, password, name, plan):
    """Create user in local JSON database"""
    db = _load_local_db()
    
    if username.lower() in [u.lower() for u in db["users"].keys()]:
        return False, "اسم المستخدم موجود بالفعل | Username already exists", None
    
    for user in db["users"].values():
        if user.get("email", "").lower() == email.lower():
            return False, "البريد الإلكتروني مستخدم بالفعل | Email already in use", None
    
    user_data = {
        "email": email,
        "name": name,
        "password": hash_password(password),
        "plan": plan,
        "email_verified": False,
        "created_at": datetime.now().isoformat(),
        "usage_today": 0,
        "last_usage_date": datetime.now().strftime("%Y-%m-%d"),
    }
    
    db["users"][username] = user_data
    _save_local_db(db)
    
    return True, "تم إنشاء الحساب بنجاح! | Account created successfully!", user_data


def _authenticate_user_local(username, password):
    """Authenticate user from local JSON database"""
    db = _load_local_db()
    
    user = None
    actual_username = None
    for uname, udata in db["users"].items():
        if uname.lower() == username.lower() or udata.get("email", "").lower() == username.lower():
            user = udata
            actual_username = uname
            break
    
    if not user:
        return False, None, "اسم المستخدم غير موجود | User not found"
    
    if not verify_password(password, user.get("password", "")):
        return False, None, "كلمة المرور غير صحيحة | Incorrect password"
    
    user["last_login"] = datetime.now().isoformat()
    db["users"][actual_username] = user
    _save_local_db(db)
    
    # Don't return password
    safe_user = {k: v for k, v in user.items() if k != "password"}
    safe_user["username"] = actual_username
    
    return True, safe_user, "تم تسجيل الدخول بنجاح | Login successful"


def _get_user_local(username=None, email=None):
    """Get user from local JSON database"""
    db = _load_local_db()
    
    for uname, udata in db["users"].items():
        if username and uname.lower() == username.lower():
            return {**udata, "username": uname}
        if email and udata.get("email", "").lower() == email.lower():
            return {**udata, "username": uname}
    
    return None


def _update_user_local(username, updates):
    """Update user in local JSON database"""
    db = _load_local_db()
    
    for uname in db["users"]:
        if uname.lower() == username.lower():
            db["users"][uname].update(updates)
            _save_local_db(db)
            return True
    
    return False


def _get_usage_today_local(username):
    """Get usage from local JSON database"""
    db = _load_local_db()
    user = db["users"].get(username, {})
    
    today = datetime.now().strftime("%Y-%m-%d")
    if user.get("last_usage_date") != today:
        return 0
    
    return user.get("usage_today", 0)
