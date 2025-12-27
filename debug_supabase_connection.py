import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock streamlit secrets if running as a standalone script
import streamlit as st

# Check if .streamlit/secrets.toml exists
secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
print(f"Checking {secrets_path}: {'Exists' if secrets_path.exists() else 'Missing'}")

try:
    from core import database as db
    
    print(f"Supabase Library Available: {db.SUPABASE_AVAILABLE}")
    
    url, key = db.get_supabase_config()
    print(f"Config Found: URL={'Yes' if url else 'No'}, Key={'Yes' if key else 'No'}")
    
    if url:
        print(f"URL starts with: {url[:10]}...")
        # Check if it's a placeholder
        placeholders = ["https://your-project.supabase.co", "YOUR_SUPABASE_URL", "SUPABASE_URL"]
        if url in placeholders:
            print("WARNING: URL is a placeholder value!")
            
    is_config = db.is_configured()
    print(f"Final Configuration Status: {'ACTIVE' if is_config else 'INACTIVE'}")
    
    if is_config:
        client = db.get_client()
        if client:
            print("Successfully initialized Supabase client.")
            # Try a simple select to test connectivity
            try:
                res = client.table("users").select("id").limit(1).execute()
                print("Database connectivity: SUCCESS")
            except Exception as e:
                print(f"Database connectivity: FAILED - {str(e)}")
        else:
            print("Failed to initialize Supabase client.")
    else:
        print("Supabase is not configured. Falling back to local storage.")

except Exception as e:
    print(f"Diagnostic Error: {str(e)}")
