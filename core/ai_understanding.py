"""
AI Data Understanding - Uses LLM to Truly Understand Data
Integrates Gemini/DeepSeek for intelligent data analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import AI libraries
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class AIDataExpert:
    """
    AI-Powered Data Expert - Uses LLM to truly understand data.
    Like having a senior data scientist looking at your data.
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: str = 'gemini'):
        """
        Initialize AI Expert.
        
        Args:
            api_key: API key (or set GEMINI_API_KEY env var)
            provider: 'gemini' or 'deepseek'
        """
        self.provider = provider
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
        self.is_configured = False
        self.model = None
        self.log_messages = []
        
        if self.api_key:
            self._setup_api()
    
    def _setup_api(self):
        """Setup the AI API."""
        try:
            if self.provider == 'gemini' and HAS_GEMINI:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.is_configured = True
                self.log("✅ Gemini API متصل")
            elif self.provider == 'deepseek' and HAS_REQUESTS:
                self.is_configured = True
                self.log("✅ DeepSeek API متصل")
        except Exception as e:
            self.log(f"⚠️ فشل الاتصال بالـ API: {e}")
            self.is_configured = False
    
    def log(self, message: str):
        """Add to log."""
        self.log_messages.append(message)
    
    def understand_data(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Use AI to understand the data deeply.
        
        Returns insights about:
        - What each column represents
        - Best cleaning strategy
        - Suggested features
        - Potential issues
        """
        # Create data summary
        summary = self._create_data_summary(df, target_col)
        
        if not self.is_configured:
            # Return rule-based understanding
            return self._rule_based_understanding(df, summary)
        
        # Use AI for understanding
        return self._ai_understanding(df, summary, target_col)
    
    def _create_data_summary(self, df: pd.DataFrame, target_col: Optional[str] = None) -> str:
        """Create a text summary of the data for the AI."""
        summary = f"""
البيانات تحتوي على {len(df)} صف و {len(df.columns)} عمود.

الأعمدة:
"""
        for col in df.columns:
            dtype = df[col].dtype
            n_unique = df[col].nunique()
            n_missing = df[col].isna().sum()
            sample = df[col].dropna().head(3).tolist()
            
            summary += f"- {col}: نوع={dtype}, قيم فريدة={n_unique}, مفقود={n_missing}, عينة={sample}\n"
        
        if target_col:
            summary += f"\nعمود الهدف: {target_col}"
        
        return summary
    
    def _ai_understanding(self, df: pd.DataFrame, summary: str, target_col: str) -> Dict[str, Any]:
        """Use AI to understand data."""
        prompt = f"""
أنت خبير بيانات محترف. حلل هذه البيانات وأعطني:

1. ماذا يمثل كل عمود على الأرجح؟
2. ما أفضل طريقة لتنظيف كل عمود؟
3. ما الميزات الجديدة التي يمكن إنشاؤها؟
4. ما المشاكل المحتملة في البيانات؟
5. ما نوع نموذج ML الأفضل؟

{summary}

أجب بصيغة JSON فقط:
{{
    "column_meanings": {{"col_name": "المعنى"}},
    "cleaning_strategy": {{"col_name": "الاستراتيجية"}},
    "suggested_features": ["ميزة1", "ميزة2"],
    "potential_issues": ["مشكلة1", "مشكلة2"],
    "recommended_model": "اسم النموذج",
    "confidence": 0.8
}}
"""
        try:
            if self.provider == 'gemini':
                response = self.model.generate_content(prompt)
                result_text = response.text
            else:
                result_text = self._call_deepseek(prompt)
            
            # Parse JSON from response
            result = self._parse_ai_response(result_text)
            result['ai_powered'] = True
            self.log("🤖 تم تحليل البيانات بالذكاء الاصطناعي")
            return result
            
        except Exception as e:
            self.log(f"⚠️ خطأ في AI: {e}")
            return self._rule_based_understanding(df, summary)
    
    def _call_deepseek(self, prompt: str) -> str:
        """Call DeepSeek API."""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']
    
    def _parse_ai_response(self, text: str) -> Dict:
        """Parse AI response to extract JSON."""
        try:
            # Try to find JSON in response
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {"raw_response": text, "parse_error": True}
    
    def _rule_based_understanding(self, df: pd.DataFrame, summary: str) -> Dict[str, Any]:
        """Rule-based data understanding (fallback)."""
        result = {
            'ai_powered': False,
            'column_meanings': {},
            'cleaning_strategy': {},
            'suggested_features': [],
            'potential_issues': [],
            'recommended_model': 'Random Forest'
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Guess column meaning
            if any(x in col_lower for x in ['name', 'اسم', 'الاسم']):
                result['column_meanings'][col] = 'اسم شخص أو كيان'
                result['cleaning_strategy'][col] = 'تنظيف النصوص وإزالة الأحرف الخاصة'
            elif any(x in col_lower for x in ['age', 'عمر', 'العمر']):
                result['column_meanings'][col] = 'عمر'
                result['cleaning_strategy'][col] = 'ملء القيم المفقودة بالوسيط'
            elif any(x in col_lower for x in ['salary', 'راتب', 'الراتب', 'price', 'سعر']):
                result['column_meanings'][col] = 'قيمة مالية'
                result['cleaning_strategy'][col] = 'إزالة رموز العملة وتحويل لأرقام'
            elif any(x in col_lower for x in ['date', 'تاريخ', 'التاريخ', 'time']):
                result['column_meanings'][col] = 'تاريخ أو وقت'
                result['cleaning_strategy'][col] = 'تحويل لتاريخ واستخراج السنة/الشهر/اليوم'
            elif any(x in col_lower for x in ['id', 'رقم', 'الرقم', 'code']):
                result['column_meanings'][col] = 'معرّف فريد'
                result['cleaning_strategy'][col] = 'حذف - لا فائدة في التدريب'
            elif any(x in col_lower for x in ['email', 'بريد', 'phone', 'هاتف']):
                result['column_meanings'][col] = 'معلومات تواصل'
                result['cleaning_strategy'][col] = 'استخراج الدومين أو كود المنطقة'
            else:
                if df[col].dtype in ['int64', 'float64']:
                    result['column_meanings'][col] = 'قيمة رقمية'
                    result['cleaning_strategy'][col] = 'ملء المفقود بالوسيط'
                else:
                    result['column_meanings'][col] = 'قيمة نصية/فئوية'
                    result['cleaning_strategy'][col] = 'ترميز الفئات'
        
        # Detect issues
        if df.isna().sum().sum() > len(df) * len(df.columns) * 0.3:
            result['potential_issues'].append('نسبة عالية من القيم المفقودة')
        
        if df.duplicated().sum() > len(df) * 0.1:
            result['potential_issues'].append('نسبة عالية من الصفوف المكررة')
        
        # Suggest features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            result['suggested_features'].append('إنشاء نسب بين الأعمدة الرقمية')
            result['suggested_features'].append('حساب المجموع والمتوسط للصفوف')
        
        self.log("📋 تم تحليل البيانات بالقواعد")
        return result
    
    def suggest_cleaning(self, df: pd.DataFrame, understanding: Dict) -> List[Dict]:
        """Suggest specific cleaning actions based on understanding."""
        actions = []
        
        for col, strategy in understanding.get('cleaning_strategy', {}).items():
            actions.append({
                'column': col,
                'action': strategy,
                'priority': 'high' if 'حذف' in strategy or 'مفقود' in strategy else 'medium'
            })
        
        return actions
    
    def explain_results(self, results: Dict, target_col: str, lang: str = 'ar') -> str:
        """Use AI to explain model results in human language."""
        if not self.is_configured:
            return self._rule_based_explanation(results, target_col, lang)
        
        prompt = f"""
أنت خبير بيانات تشرح للمدير التنفيذي.

نتائج التحليل:
- نوع المشكلة: {results.get('problem_type', 'غير محدد')}
- أفضل نموذج: {results.get('best_model', 'غير محدد')}
- الدقة: {results.get('metrics', {})}
- أهم المتغيرات: {results.get('feature_importance', pd.DataFrame()).head(5).to_dict() if isinstance(results.get('feature_importance'), pd.DataFrame) else {}}

اكتب شرحاً بسيطاً (3-5 جمل) يفهمه غير التقني.
"""
        try:
            if self.provider == 'gemini':
                response = self.model.generate_content(prompt)
                return response.text
            else:
                return self._call_deepseek(prompt)
        except:
            return self._rule_based_explanation(results, target_col, lang)
    
    def _rule_based_explanation(self, results: Dict, target_col: str, lang: str) -> str:
        """Generate explanation without AI."""
        problem_type = results.get('problem_type', 'classification')
        best_model = results.get('best_model', 'Random Forest')
        metrics = results.get('metrics', {})
        
        if lang == 'ar':
            if problem_type == 'classification':
                accuracy = metrics.get('accuracy', 0) * 100
                return f"""
📊 **ملخص التحليل:**

تم بناء نموذج {best_model} للتنبؤ بـ {target_col}.
دقة النموذج: {accuracy:.1f}%

{'✅ النموذج جيد ويمكن الاعتماد عليه.' if accuracy > 70 else '⚠️ النموذج يحتاج تحسين - جرب بيانات أكثر.'}
"""
            else:
                r2 = metrics.get('r2', 0) * 100
                return f"""
📊 **ملخص التحليل:**

تم بناء نموذج {best_model} للتنبؤ بقيمة {target_col}.
جودة التنبؤ (R²): {r2:.1f}%

{'✅ النموذج جيد للتنبؤات.' if r2 > 50 else '⚠️ النموذج يحتاج تحسين.'}
"""
        else:
            return f"Model {best_model} trained with {metrics}"
    
    def get_log(self) -> List[str]:
        """Get log messages."""
        return self.log_messages


def understand_data(df: pd.DataFrame, target_col: str, api_key: Optional[str] = None) -> Dict:
    """Quick function to understand data."""
    expert = AIDataExpert(api_key)
    return expert.understand_data(df, target_col)
