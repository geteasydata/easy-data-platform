"""
Centralized Configuration Utility
Handles API keys and platform-wide settings safely from multiple sources.
"""

import os
import streamlit as st

def get_api_key(key_name: str) -> str:
    """
    Retrieve API key from best available source.
    Priority: 1. Streamlit Secrets, 2. Environment Variable
    """
    # 1. Try Streamlit Secrets (Cloud/Local standard)
    try:
        # Check standard api_keys section
        if "api_keys" in st.secrets:
            if key_name in st.secrets["api_keys"]:
                return st.secrets["api_keys"][key_name]
        
        # Check root level
        if key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
        
    # 2. Try Environment Variables (Local dev fallback)
    return os.environ.get(key_name)

def is_ai_configured() -> bool:
    """Check if at least one AI provider is configured."""
    return any([
        get_api_key("GEMINI_API_KEY"),
        get_api_key("GROQ_API_KEY"),
        get_api_key("DEEPSEEK_API_KEY")
    ])
