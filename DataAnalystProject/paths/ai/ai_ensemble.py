"""
AI Ensemble - Combines Multiple AI Providers for Superior Analysis
Automatically uses Groq, DeepSeek, and Gemini together
"""

import os
from typing import Dict, Any, Optional, List
import pandas as pd
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import Groq
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# Try to import Gemini
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Try to import requests for DeepSeek
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class AIEnsemble:
    """
    AI Ensemble Engine - Uses ALL available AI providers together.
    Automatically combines insights from Groq, DeepSeek, and Gemini.
    """
    
    # Default API keys (user provided)
    DEFAULT_GROQ_KEY = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    DEFAULT_DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY")
    DEFAULT_GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    
    def __init__(self):
        """Initialize all AI providers automatically."""
        self.providers = {}
        self.log_messages = []
        
        # Setup Groq
        if HAS_GROQ:
            try:
                self.providers['groq'] = {
                    'client': Groq(api_key=self.DEFAULT_GROQ_KEY),
                    'name': 'Groq (Llama 3.3)',
                    'emoji': '🚀'
                }
                self.log("✅ Groq متصل")
            except Exception as e:
                self.log(f"⚠️ Groq: {e}")
        
        # Setup DeepSeek
        if HAS_REQUESTS:
            self.providers['deepseek'] = {
                'key': self.DEFAULT_DEEPSEEK_KEY,
                'name': 'DeepSeek',
                'emoji': '🔮'
            }
            self.log("✅ DeepSeek متصل")
        
        # Setup Gemini
        if HAS_GEMINI:
            try:
                genai.configure(api_key=self.DEFAULT_GEMINI_KEY)
                self.providers['gemini'] = {
                    'model': genai.GenerativeModel('gemini-pro'),
                    'name': 'Gemini',
                    'emoji': '✨'
                }
                self.log("✅ Gemini متصل")
            except Exception as e:
                self.log(f"⚠️ Gemini: {e}")
        
        self.log(f"🤖 {len(self.providers)} مزودي ذكاء اصطناعي جاهزون")
    
    def set_user_key(self, api_key: str):
        """Update Gemini key from user input."""
        if not api_key:
            return
            
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.providers['gemini'] = {
                'model': genai.GenerativeModel('gemini-pro'),
                'name': 'Gemini (User Key)',
                'emoji': '✨'
            }
            self.log(f"✅ Gemini Key updated by user")
        except Exception as e:
            self.log(f"⚠️ Failed to update Gemini Key: {e}")
    
    def log(self, message: str):
        """Add log message."""
        self.log_messages.append(message)
        print(message)
    
    def get_log(self) -> List[str]:
        """Get all log messages."""
        return self.log_messages
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API."""
        try:
            response = self.providers['groq']['client'].chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "أنت خبير بيانات محترف. أجب بإيجاز ووضوح."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"خطأ: {e}"
    
    def _call_deepseek(self, prompt: str) -> str:
        """Call DeepSeek API."""
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.providers['deepseek']['key']}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500
            }
            response = requests.post(url, headers=headers, json=data, timeout=30)
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"خطأ: {e}"
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        try:
            response = self.providers['gemini']['model'].generate_content(prompt)
            return response.text
        except Exception as e:
            return f"خطأ: {e}"
    
    def _call_all_providers(self, prompt: str) -> Dict[str, str]:
        """Call all providers and collect responses."""
        results = {}
        
        # Use ThreadPoolExecutor for parallel calls
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            if 'groq' in self.providers:
                futures[executor.submit(self._call_groq, prompt)] = 'groq'
            if 'deepseek' in self.providers:
                futures[executor.submit(self._call_deepseek, prompt)] = 'deepseek'
            if 'gemini' in self.providers:
                futures[executor.submit(self._call_gemini, prompt)] = 'gemini'
            
            for future in as_completed(futures):
                provider = futures[future]
                try:
                    results[provider] = future.result()
                except Exception as e:
                    results[provider] = f"خطأ: {e}"
        
        return results
    
    def _combine_insights(self, responses: Dict[str, str], lang: str = 'ar') -> str:
        """Combine insights from all providers into one comprehensive response."""
        valid_responses = {k: v for k, v in responses.items() if not v.startswith("خطأ")}
        
        if not valid_responses:
            return "لم يتمكن أي من مزودي الذكاء الاصطناعي من التحليل."
        
        # If only one provider responded, return its response
        if len(valid_responses) == 1:
            provider = list(valid_responses.keys())[0]
            emoji = self.providers[provider]['emoji']
            return f"{emoji} {valid_responses[provider]}"
        
        # Combine multiple responses
        if lang == 'ar':
            combined = "## 🤖 تحليل الذكاء الاصطناعي المجمّع\n\n"
            
            for provider, response in valid_responses.items():
                emoji = self.providers[provider]['emoji']
                name = self.providers[provider]['name']
                combined += f"### {emoji} {name}:\n{response}\n\n"
            
            combined += "---\n"
            combined += f"*تم التحليل بواسطة {len(valid_responses)} نماذج ذكاء اصطناعي*"
        else:
            combined = "## 🤖 Combined AI Analysis\n\n"
            
            for provider, response in valid_responses.items():
                emoji = self.providers[provider]['emoji']
                name = self.providers[provider]['name']
                combined += f"### {emoji} {name}:\n{response}\n\n"
            
            combined += "---\n"
            combined += f"*Analyzed by {len(valid_responses)} AI models*"
        
        return combined
    
    def analyze_data_quality(self, analysis: Dict[str, Any], lang: str = 'ar') -> str:
        """Analyze data quality using all AI providers."""
        prompt = f"""
        أنت خبير بيانات. حلل هذه البيانات بإيجاز (3-4 نقاط فقط):
        
        - الصفوف: {analysis.get('rows', 0)}
        - الأعمدة: {analysis.get('columns', 0)}
        - أعمدة رقمية: {len(analysis.get('numeric_columns', []))}
        - أعمدة نصية: {len(analysis.get('categorical_columns', []))}
        - قيم مفقودة: {analysis.get('total_missing', 0)}
        - مكررات: {analysis.get('duplicates', 0)}
        
        اللغة: {'العربية' if lang == 'ar' else 'English'}
        استخدم إيموجي. كن موجزاً.
        """
        
        responses = self._call_all_providers(prompt)
        return self._combine_insights(responses, lang)
    
    def generate_insights(self, results: Dict[str, Any], target_col: str, lang: str = 'ar') -> str:
        """Generate strategic insights using all AI providers."""
        # Get top features safely
        feature_importance = results.get('feature_importance')
        if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
            top_features = feature_importance.head(5)['Feature'].tolist()
        else:
            top_features = []
        
        prompt = f"""
        أنت خبير بيانات يقدم توصيات استراتيجية. بناءً على:
        
        - نوع المشكلة: {results.get('problem_type', 'غير محدد')}
        - أفضل نموذج: {results.get('best_model', 'غير محدد')}
        - الأداء: {results.get('metrics', {})}
        - أهم المتغيرات: {top_features}
        - الهدف: {target_col}
        
        قدم 3-4 توصيات استراتيجية موجزة.
        اللغة: {'العربية' if lang == 'ar' else 'English'}
        استخدم إيموجي.
        """
        
        responses = self._call_all_providers(prompt)
        return self._combine_insights(responses, lang)
    
    def suggest_feature_engineering(self, df: pd.DataFrame, lang: str = 'ar') -> str:
        """Suggest feature engineering using all AI providers."""
        columns_info = {col: str(df[col].dtype) for col in list(df.columns)[:15]}
        
        prompt = f"""
        أنت خبير هندسة ميزات. بناءً على هذه الأعمدة:
        {columns_info}
        
        اقترح 3-4 ميزات جديدة يمكن إنشاؤها.
        اللغة: {'العربية' if lang == 'ar' else 'English'}
        كن محدداً وعملياً.
        """
        
        responses = self._call_all_providers(prompt)
        return self._combine_insights(responses, lang)
    
    def understand_data(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Deep data understanding using ensemble."""
        summary = self._create_data_summary(df, target_col)
        
        prompt = f"""
        حلل هذه البيانات وأعطني JSON فقط:
        
        {summary}
        
        {{
            "column_meanings": {{"col": "المعنى"}},
            "cleaning_strategy": {{"col": "الطريقة"}},
            "suggested_features": ["ميزة1"],
            "potential_issues": ["مشكلة1"],
            "recommended_model": "اسم النموذج"
        }}
        """
        
        # Use Groq for structured responses (fastest and most reliable for JSON)
        if 'groq' in self.providers:
            try:
                response = self._call_groq(prompt)
                result = self._parse_json_response(response)
                result['ai_powered'] = True
                result['ensemble'] = True
                return result
            except:
                pass
        
        # Fallback
        return self._rule_based_understanding(df)
    
    def _create_data_summary(self, df: pd.DataFrame, target_col: Optional[str] = None) -> str:
        """Create data summary."""
        summary = f"البيانات: {len(df)} صف × {len(df.columns)} عمود\n\nالأعمدة:\n"
        
        for col in list(df.columns)[:10]:
            dtype = df[col].dtype
            n_unique = df[col].nunique()
            sample = df[col].dropna().head(2).tolist()
            summary += f"- {col}: {dtype}, فريد={n_unique}, عينة={sample}\n"
        
        if target_col:
            summary += f"\nالهدف: {target_col}"
        
        return summary
    
    def _parse_json_response(self, text: str) -> Dict:
        """Parse JSON from AI response."""
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"raw_response": text}
    
    def _rule_based_understanding(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback rule-based understanding."""
        return {
            'ai_powered': False,
            'column_meanings': {col: 'غير محدد' for col in df.columns},
            'cleaning_strategy': {},
            'suggested_features': ['إنشاء نسب بين الأعمدة الرقمية'],
            'potential_issues': [],
            'recommended_model': 'Random Forest'
        }


# Create singleton instance
_ensemble_instance = None

def get_ensemble() -> AIEnsemble:
    """Get or create the AI Ensemble instance."""
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = AIEnsemble()
    return _ensemble_instance
