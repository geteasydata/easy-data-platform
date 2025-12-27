import os

secrets_path = ".streamlit/secrets.toml"
new_url = "https://banmjznaaunsnlerzqfg.supabase.co"
new_key = "sb_publishable_CowgsCfN6xjd37B3IoVLBg_omAs2HoQ"

if not os.path.exists(secrets_path):
    # Ensure directory exists
    os.makedirs(os.path.dirname(secrets_path), exist_ok=True)
    with open(secrets_path, "w") as f:
        f.write(f'SUPABASE_URL = "{new_url}"\nSUPABASE_KEY = "{new_key}"\n')
    print("Created new secrets.toml with Supabase keys.")
else:
    with open(secrets_path, "r") as f:
        lines = f.readlines()
    
    updated_url = False
    updated_key = False
    new_lines = []
    
    for line in lines:
        if line.strip().startswith("SUPABASE_URL"):
            new_lines.append(f'SUPABASE_URL = "{new_url}"\n')
            updated_url = True
        elif line.strip().startswith("SUPABASE_KEY"):
            new_lines.append(f'SUPABASE_KEY = "{new_key}"\n')
            updated_key = True
        else:
            new_lines.append(line)
            
    if not updated_url:
        new_lines.append(f'SUPABASE_URL = "{new_url}"\n')
    if not updated_key:
        new_lines.append(f'SUPABASE_KEY = "{new_key}"\n')
        
    with open(secrets_path, "w") as f:
        f.writelines(new_lines)
    print("Updated Supabase keys in existing secrets.toml.")
