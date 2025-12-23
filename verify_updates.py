
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath('d:/The drive/AG IDE/Data Scientist/AI_Expert_App'))

def verify_modules():
    print("1. Testing Imports...")
    try:
        from reports.excel_output import create_excel_report
        from reports.powerbi_export import create_powerbi_package
        from core.data_loader import DataLoader
        print("✅ Imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return

    # Create dummy data
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['x', 'y', 'x', 'y', 'x'],
        'Target': [0, 1, 0, 1, 0]
    })
    
    metrics = {'accuracy': 0.95, 'f1': 0.94}
    analysis = {'overview': {'rows': 5}, 'columns': ['A', 'B', 'Target']}
    feature_importance = pd.DataFrame({'Feature': ['A', 'B'], 'Importance': [0.8, 0.2]})
    
    print("\n2. Testing Excel Dashboard Generation...")
    try:
        excel_bytes = create_excel_report(df, None, feature_importance, metrics, analysis, 'en')
        if len(excel_bytes) > 0:
            print(f"✅ Excel generated ({len(excel_bytes)} bytes)")
        else:
            print("❌ Excel generation returned empty bytes")
    except Exception as e:
        print(f"❌ Excel generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n3. Testing Power BI Package Generation...")
    try:
        pbi_bytes = create_powerbi_package(df, None, analysis, 'en')
        if len(pbi_bytes) > 0:
            print(f"✅ Power BI Package generated ({len(pbi_bytes)} bytes)")
        else:
            print("❌ Power BI generation returned empty bytes")
    except Exception as e:
        print(f"❌ Power BI generation failed: {e}")

    print("\n4. Testing SQL Loader...")
    try:
        loader = DataLoader()
        # Test imports of sqlalchemy inside the method
        # We can't easily test connection without a DB, but we can verify the method exists and import logic works
        # Let's try to load from a non-existent DB to check it attempts connection (and thus imported sqlalchemy)
        res = loader.load_sql("sqlite:///non_existent.db", "SELECT 1")
        # It should return None/Empty but NOT error with "ImportError"
        if "SQLAlchemy not installed" in loader.errors:
             print("❌ SQLAlchemy import failed in loader")
        else:
             print("✅ SQL Loader initialized and attempted connection")
    except Exception as e:
        print(f"❌ SQL Loader failed: {e}")

if __name__ == "__main__":
    verify_modules()
