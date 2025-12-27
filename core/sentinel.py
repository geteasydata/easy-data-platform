"""
AI Sentinel - Autonomous Maintenance & Error Recovery System
Monitors application health, diagnoses crashes using Gemini, and suggests fixes.
"""

import json
import traceback
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import streamlit as st

class AISentinel:
    """
    Autonomous Guardian for the Easy Data Platform.
    Catches errors, analyzes with AI, and logs for maintenance.
    """
    
    def show_maintenance_ui(self, lang: str = "en"):
        """Display the maintenance dashboard in Streamlit."""
        st.markdown("### 🛡️ " + ("Maintenance Center" if lang == 'en' else "مركز الصيانة الذكية"))
        st.info("AI Sentinel is monitoring the platform for errors." if lang == 'en' else "الحارس الذكي يراقب المنصة بحثاً عن أي أخطاء.")
        
        # Test Button
        if st.button("🧪 " + ("Trigger Test Error" if lang == 'en' else "تحفيز خطأ تجريبي"), key="global_sentinel_test"):
            raise ValueError("AI Sentinel Test: This is a simulated crash to verify the guardian system.")
        
        # Load logs
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except:
            logs = []
            
        if not logs:
            st.success("No system errors detected recently." if lang == 'en' else "لم يتم رصد أي أخطاء في النظام مؤخراً.")
            return
            
        for log in logs:
            status_icon = '🔴' if log['status']=='new' else '🔍'
            with st.expander(f"{status_icon} {log['type']}: {log['message'][:50]}... ({log['timestamp'][:16]})"):
                st.code(log['traceback'], language='python')
                
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if st.button(f"Analyze with AI" if lang == 'en' else "تحليل بالذكاء الاصطناعي", key=f"analyze_{log['id']}"):
                        with st.spinner("Gemini is diagnosing..."):
                            if self.analyze_error(log['id']):
                                st.success("Diagnosis Complete!" if lang == 'en' else "اكتمل التشخيص!")
                                st.rerun()
                                
                with col2:
                    if log['status'] == 'analyzed':
                        st.markdown("**AI Diagnosis:**")
                        st.write(log['diagnosis'])
                        if log['suggested_fix']:
                            st.markdown("**Suggested Fix:**")
                            st.code(log['suggested_fix'], language='python')
                            if st.button("Apply Fix (Coming Soon)" if lang == 'en' else "تطبيق الإصلاح (قريباً)", key=f"apply_{log['id']}"):
                                st.info("Automatic patching is being calibrated." if lang == 'en' else "يتم الآن معايرة نظام الترقيع التلقائي.")

    
    def __init__(self, log_path: str = "user_data/sentinel_logs.json"):
        self.log_path = Path(log_path)
        self._ensure_log_exists()
        
    def _ensure_log_exists(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Record an error and trigger AI analysis if possible."""
        error_trace = traceback.format_exc()
        error_type = type(error).__name__
        error_msg = str(error)
        
        entry = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S_%f"),
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_msg,
            "traceback": error_trace,
            "context": context or {},
            "status": "new",  # new, analyzed, repairing, fixed, ignored
            "diagnosis": None,
            "suggested_fix": None
        }
        
        # Save to local log
        try:
            with open(self.log_path, 'r+', encoding='utf-8') as f:
                logs = json.load(f)
                logs.insert(0, entry) # Newest first
                f.seek(0)
                json.dump(logs[:100], f, indent=4, ensure_ascii=False) # Keep last 100
                f.truncate()
        except Exception as e:
            print(f"Sentinel Logging Failed: {e}")
            
        return entry["id"]

    def analyze_error(self, log_id: str):
        """Use Gemini to diagnose the specific error."""
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
                
            entry = next((l for l in logs if l["id"] == log_id), None)
            if not entry:
                return False
                
            # Connect to AI Ensemble
            from core.ai_ensemble import get_ensemble
            ensemble = get_ensemble()
            
            # Prepare special maintenance prompt
            prompt = self._build_maintenance_prompt(entry)
            
            # Use Gemini 1.5 Pro for best context understanding
            # If not available, ensemble.chat will fallback to others
            response = ensemble.chat(prompt, context={"type": "maintenance_diagnosis"})
            
            # Parse AI response (Expecting JSON-like structure or clear sections)
            diagnosis, fix_code = self._parse_ai_response(response)
            
            # Update log
            entry["diagnosis"] = diagnosis
            entry["suggested_fix"] = fix_code
            entry["status"] = "analyzed"
            
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=4, ensure_ascii=False)
                
            return True
        except Exception as e:
            print(f"AI Analysis Failed: {e}")
            return False

    def _build_maintenance_prompt(self, entry: Dict[str, Any]) -> str:
        return f"""
YOU ARE THE CHIEF MAINTENANCE ENGINEER FOR THE 'EASY DATA' PLATFORM.
A CRITICAL ERROR HAS OCCURRED IN THE SYSTEM.

ERROR TYPE: {entry['type']}
MESSAGE: {entry['message']}
TIMESTAMP: {entry['timestamp']}

TRACEBACK:
{entry['traceback']}

USER CONTEXT:
{json.dumps(entry['context'], indent=2)}

TASK:
1. Diagnose the ROOT CAUSE of this error.
2. Provide a technical explanation of WHY it happened.
3. PROVIDE THE EXACT PYTHON CODE required to fix it.

FORMAT YOUR RESPONSE AS FOLLOWS:
DIAGNOSIS: [Detailed technical explanation]
REPAIR_CODE: 
```python
# The complete fixed code or snippet
```
"""

    def _parse_ai_response(self, response: str):
        """Extract diagnosis and code from AI response string."""
        diagnosis = "No diagnosis generated."
        fix_code = ""
        
        if "DIAGNOSIS:" in response:
            diagnosis = response.split("DIAGNOSIS:")[1].split("REPAIR_CODE:")[0].strip()
            
        if "```python" in response:
            fix_code = response.split("```python")[1].split("```")[0].strip()
            
        return diagnosis, fix_code

def get_sentinel():
    """Singleton-like accessor for Streamlit session state."""
    if 'sentinel' not in st.session_state:
        st.session_state.sentinel = AISentinel()
    return st.session_state.sentinel
