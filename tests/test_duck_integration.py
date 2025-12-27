
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.duck_engine import DuckDBEngine

def test_duck_integration():
    print("🧪 Testing DuckDB Integration...")
    
    # 1. Create Dummy Data (Large-ish)
    temp_dir = PROJECT_ROOT / "temp_data"
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / "test_large.csv"
    
    print("📝 Generating dummy CSV...")
    df = pd.DataFrame({
        'id': range(10000),
        'value': np.random.randn(10000),
        'category': np.random.choice(['A', 'B', 'C'], 10000),
        'missing_col': [None if i % 10 == 0 else i for i in range(10000)]
    })
    df.to_csv(temp_file, index=False)
    
    # 2. Init Engine
    print("⚙️ Initializing Engine...")
    engine = DuckDBEngine()
    
    # 3. Register
    print("📥 Registering file...")
    success = engine.register_file(str(temp_file), 'test_table')
    
    if not success:
        print("❌ Registration Failed")
        return
        
    # 4. Get Stats
    print("📊 Getting Stats...")
    stats = engine.get_summary_stats('test_table')
    
    print(f"Stats: {stats}")
    
    assert stats['total_rows'] == 10000
    assert stats['total_missing'] == 1000 # 10% of 10000
    
    # 5. Get Sample
    print("🔬 Getting Sample...")
    sample = engine.get_sample('test_table', limit=5)
    print(sample)
    assert len(sample) == 5
    
    print("✅ Test Passed!")

if __name__ == "__main__":
    test_duck_integration()
