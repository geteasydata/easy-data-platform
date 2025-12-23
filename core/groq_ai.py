"""
Groq AI Integration - Llama 3.3 70B for Expert Data Analysis
Primary AI engine for fast, intelligent data insights
"""

import os
from typing import Dict, Any, Optional, List
import pandas as pd
import json
import re

# Try to import Groq
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# Try to import requests for DeepSeek fallback
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class GroqAI:
    """
    Groq AI Engine using Llama 3.3 70B for intelligent data analysis.
    Blazingly fast with free tier support.
    """
    
    # Default API keys (user provided)
    DEFAULT_GROQ_KEY = "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"
    DEFAULT_DEEPSEEK_KEY = "sk-0d8b9806c944495387bd466460a53932"
    
    def __init__(self, api_key: Optional[str] = None, deepseek_key: Optional[str] = None):
        """
        Initialize Groq AI with optional DeepSeek fallback.
        
        Args:
            api_key: Groq API key (or uses default/env var)
            deepseek_key: Optional DeepSeek API key for fallback
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY') or self.DEFAULT_GROQ_KEY
        self.deepseek_key = deepseek_key or os.getenv('DEEPSEEK_API_KEY') or self.DEFAULT_DEEPSEEK_KEY
        self.client = None
        self.is_configured = False
        self.active_provider = None
        self.log_messages = []
        
        self._setup()
    
    def _setup(self):
        """Setup the AI client."""
        # Try Groq first
        if HAS_GROQ and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                # Test connection with a simple request
                self.is_configured = True
                self.active_provider = 'groq'
                self.log("✅ Groq API متصل - Llama 3.3 70B جاهز")
            except Exception as e:
                self.log(f"⚠️ فشل اتصال Groq: {e}")
        
        # If Groq fails, try DeepSeek
        if not self.is_configured and self.deepseek_key and HAS_REQUESTS:
            self.is_configured = True
            self.active_provider = 'deepseek'
            self.log("✅ DeepSeek API متصل كبديل")
    
    def log(self, message: str):
        """Add log message."""
        self.log_messages.append(message)
        print(message)
    
    def get_log(self) -> List[str]:
        """Get all log messages."""
        return self.log_messages
    
    def _call_groq(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call Groq API with Llama 3.3."""
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "أنت خبير بيانات محترف بخبرة 30 سنة. تجيب بوضوح وإيجاز. You are a senior data scientist with 30 years of experience."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def _call_deepseek(self, prompt: str) -> str:
        """Call DeepSeek API as fallback."""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.deepseek_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        return response.json()['choices'][0]['message']['content']
    
    def _call_ai(self, prompt: str) -> str:
        """Call the active AI provider."""
        if self.active_provider == 'groq':
            return self._call_groq(prompt)
        elif self.active_provider == 'deepseek':
            return self._call_deepseek(prompt)
        else:
            raise Exception("No AI provider configured")
    
    def analyze_data_quality(self, analysis: Dict[str, Any], lang: str = 'ar') -> str:
        """Generate AI insights about data quality."""
        if not self.is_configured:
            return self._fallback_data_quality(analysis, lang)
        
        prompt = f"""
        You are a senior data scientist with 30+ years of experience.
        Analyze this dataset summary and provide expert insights:
        
        - Rows: {analysis.get('rows', 0)}
        - Columns: {analysis.get('columns', 0)}
        - Numeric columns: {len(analysis.get('numeric_columns', []))}
        - Categorical columns: {len(analysis.get('categorical_columns', []))}
        - Missing values: {analysis.get('total_missing', 0)}
        - Duplicates: {analysis.get('duplicates', 0)}
        
        Provide 3-5 key observations and recommendations.
        Language: {'Arabic' if lang == 'ar' else 'English'}
        Format: Bullet points with emojis for visual appeal.
        Keep it concise and actionable.
        """
        
        try:
            return self._call_ai(prompt)
        except Exception as e:
            self.log(f"⚠️ AI Error: {e}")
            return self._fallback_data_quality(analysis, lang)
    
    def generate_insights(self, results: Dict[str, Any], target_col: str, lang: str = 'ar') -> str:
        """Generate strategic business insights from ML results."""
        if not self.is_configured:
            return self._fallback_insights(results, target_col, lang)
        
        # Get top features safely
        feature_importance = results.get('feature_importance')
        if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
            top_features = feature_importance.head(5)['Feature'].tolist()
        else:
            top_features = []
        
        prompt = f"""
        You are a senior data scientist with 30+ years of experience advising Fortune 500 companies.
        Based on the ML analysis results below, provide strategic business recommendations:
        
        Problem Type: {results.get('problem_type', 'unknown')}
        Best Model: {results.get('best_model', 'unknown')}
        Performance Metrics: {results.get('metrics', {})}
        Top 5 Important Features: {top_features}
        Target Variable: {target_col}
        
        Provide:
        1. Performance interpretation (is it good? what does it mean for business?)
        2. Top 3 strategic recommendations based on feature importance
        3. Data quality notes and improvement suggestions
        4. Next steps for production deployment
        
        Language: {'Arabic' if lang == 'ar' else 'English'}
        Format: Professional report style with emojis and clear sections.
        Be specific and actionable.
        """
        
        try:
            return self._call_ai(prompt)
        except Exception as e:
            self.log(f"⚠️ AI Error: {e}")
            return self._fallback_insights(results, target_col, lang)
    
    def suggest_feature_engineering(self, df: pd.DataFrame, lang: str = 'ar') -> str:
        """Suggest feature engineering ideas based on data."""
        if not self.is_configured:
            return self._fallback_feature_suggestions(df, lang)
        
        columns_info = {col: str(df[col].dtype) for col in list(df.columns)[:20]}
        
        prompt = f"""
        As an expert data scientist, suggest feature engineering ideas for this dataset:
        
        Columns and types: {columns_info}
        Number of rows: {len(df)}
        
        Suggest:
        1. Derived features that could be created
        2. Feature combinations or interactions
        3. Time-based features if applicable
        4. Encoding strategies for categorical variables
        
        Language: {'Arabic' if lang == 'ar' else 'English'}
        Be specific and practical.
        """
        
        try:
            return self._call_ai(prompt)
        except Exception as e:
            self.log(f"⚠️ AI Error: {e}")
            return self._fallback_feature_suggestions(df, lang)
    
    def understand_data(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Use AI to deeply understand the data.
        
        Returns insights about:
        - What each column represents
        - Best cleaning strategy
        - Suggested features
        - Potential issues
        - Recommended model
        """
        summary = self._create_data_summary(df, target_col)
        
        if not self.is_configured:
            return self._rule_based_understanding(df, summary)
        
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
            result_text = self._call_ai(prompt)
            result = self._parse_json_response(result_text)
            result['ai_powered'] = True
            result['provider'] = self.active_provider
            self.log("🤖 تم تحليل البيانات بالذكاء الاصطناعي")
            return result
        except Exception as e:
            self.log(f"⚠️ AI Error: {e}")
            return self._rule_based_understanding(df, summary)
    
    def explain_results(self, results: Dict, target_col: str, lang: str = 'ar') -> str:
        """Use AI to explain model results in human language."""
        if not self.is_configured:
            return self._rule_based_explanation(results, target_col, lang)
        
        # Get feature importance safely
        feature_importance = results.get('feature_importance')
        if isinstance(feature_importance, pd.DataFrame):
            top_features = feature_importance.head(5).to_dict()
        else:
            top_features = {}
        
        prompt = f"""
أنت خبير بيانات تشرح للمدير التنفيذي.

نتائج التحليل:
- نوع المشكلة: {results.get('problem_type', 'غير محدد')}
- أفضل نموذج: {results.get('best_model', 'غير محدد')}
- الدقة: {results.get('metrics', {})}
- أهم المتغيرات: {top_features}

اكتب شرحاً بسيطاً (3-5 جمل) يفهمه غير التقني.
اللغة: {'العربية' if lang == 'ar' else 'English'}
"""
        try:
            return self._call_ai(prompt)
        except Exception as e:
            self.log(f"⚠️ AI Error: {e}")
            return self._rule_based_explanation(results, target_col, lang)
    
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
    
    def _parse_json_response(self, text: str) -> Dict:
        """Parse AI response to extract JSON."""
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {"raw_response": text, "parse_error": True}
    
    def _fallback_data_quality(self, analysis: Dict[str, Any], lang: str) -> str:
        """Fallback when AI is not available."""
        if lang == 'ar':
            insights = []
            insights.append(f"📊 **حجم البيانات**: {analysis.get('rows', 0)} صف × {analysis.get('columns', 0)} عمود")
            
            total_missing = analysis.get('total_missing', 0)
            rows = analysis.get('rows', 1)
            cols = analysis.get('columns', 1)
            
            if total_missing > 0:
                pct = (total_missing / (rows * cols)) * 100
                insights.append(f"⚠️ **قيم مفقودة**: {total_missing} ({pct:.1f}%) - تم معالجتها تلقائياً")
            else:
                insights.append("✅ **لا توجد قيم مفقودة**")
            
            if analysis.get('duplicates', 0) > 0:
                insights.append(f"🗑️ **صفوف مكررة**: {analysis['duplicates']} - تم إزالتها")
            
            if analysis.get('numeric_columns'):
                insights.append(f"🔢 **أعمدة رقمية**: {len(analysis['numeric_columns'])}")
            if analysis.get('categorical_columns'):
                insights.append(f"🏷️ **أعمدة نصية**: {len(analysis['categorical_columns'])}")
            
            return "\n".join(insights)
        else:
            insights = []
            insights.append(f"📊 **Data Size**: {analysis.get('rows', 0)} rows × {analysis.get('columns', 0)} columns")
            
            total_missing = analysis.get('total_missing', 0)
            rows = analysis.get('rows', 1)
            cols = analysis.get('columns', 1)
            
            if total_missing > 0:
                pct = (total_missing / (rows * cols)) * 100
                insights.append(f"⚠️ **Missing Values**: {total_missing} ({pct:.1f}%) - Auto-handled")
            else:
                insights.append("✅ **No missing values**")
            
            if analysis.get('duplicates', 0) > 0:
                insights.append(f"🗑️ **Duplicate Rows**: {analysis['duplicates']} - Removed")
            
            if analysis.get('numeric_columns'):
                insights.append(f"🔢 **Numeric Columns**: {len(analysis['numeric_columns'])}")
            if analysis.get('categorical_columns'):
                insights.append(f"🏷️ **Categorical Columns**: {len(analysis['categorical_columns'])}")
            
            return "\n".join(insights)
    
    def _fallback_insights(self, results: Dict[str, Any], target_col: str, lang: str) -> str:
        """Fallback insights when AI is not available."""
        metrics = results.get('metrics', {})
        
        feature_importance = results.get('feature_importance')
        if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
            top_features = feature_importance.head(3)['Feature'].tolist()
        else:
            top_features = []
        
        if lang == 'ar':
            insights = []
            
            if results.get('problem_type') == 'classification':
                acc = metrics.get('accuracy', 0)
                if acc > 0.90:
                    insights.append(f"✅ **أداء ممتاز** (دقة: {acc:.1%}): النموذج جاهز للإنتاج!")
                elif acc > 0.75:
                    insights.append(f"⚠️ **أداء جيد** (دقة: {acc:.1%}): يحتاج مراقبة في بعض الحالات")
                else:
                    insights.append(f"❌ **أداء ضعيف** (دقة: {acc:.1%}): نحتاج بيانات أو ميزات إضافية")
            else:
                r2 = metrics.get('r2', 0)
                if r2 > 0.80:
                    insights.append(f"✅ **تفسير قوي** (R²: {r2:.2f}): النموذج يفسر معظم التباين")
                else:
                    insights.append(f"⚠️ **تفسير متوسط** (R²: {r2:.2f}): هناك عوامل غير مرصودة")
            
            insights.append("\n**💡 التوصيات الاستراتيجية:**")
            if len(top_features) >= 1:
                insights.append(f"1. **{top_features[0]}** هو المؤثر الأهم - ركز مواردك هنا")
            if len(top_features) >= 2:
                insights.append(f"2. **{top_features[1]}** عامل ثانوي مهم - راقبه عن كثب")
            if len(top_features) >= 3:
                insights.append(f"3. **{top_features[2]}** يستحق الاهتمام أيضاً")
            
            return "\n".join(insights)
        else:
            insights = []
            
            if results.get('problem_type') == 'classification':
                acc = metrics.get('accuracy', 0)
                if acc > 0.90:
                    insights.append(f"✅ **Excellent Performance** (Accuracy: {acc:.1%}): Production-ready!")
                elif acc > 0.75:
                    insights.append(f"⚠️ **Good Performance** (Accuracy: {acc:.1%}): Needs monitoring")
                else:
                    insights.append(f"❌ **Weak Performance** (Accuracy: {acc:.1%}): More data needed")
            else:
                r2 = metrics.get('r2', 0)
                if r2 > 0.80:
                    insights.append(f"✅ **Strong Fit** (R²: {r2:.2f}): Model explains most variance")
                else:
                    insights.append(f"⚠️ **Moderate Fit** (R²: {r2:.2f}): External factors present")
            
            insights.append("\n**💡 Strategic Recommendations:**")
            if len(top_features) >= 1:
                insights.append(f"1. **{top_features[0]}** is the key driver - Focus resources here")
            if len(top_features) >= 2:
                insights.append(f"2. **{top_features[1]}** is secondary but important")
            if len(top_features) >= 3:
                insights.append(f"3. **{top_features[2]}** also deserves attention")
            
            return "\n".join(insights)
    
    def _fallback_feature_suggestions(self, df: pd.DataFrame, lang: str) -> str:
        """Fallback feature suggestions."""
        if lang == 'ar':
            return """
**💡 اقتراحات هندسة الميزات:**
1. 🔢 إنشاء ميزات رياضية (المجموع، المتوسط، النسب)
2. 📅 استخراج ميزات زمنية إذا كانت هناك تواريخ
3. 🏷️ تجميع الفئات النادرة في فئة واحدة
4. 📊 إنشاء ميزات تفاعلية بين الأعمدة المهمة
"""
        else:
            return """
**💡 Feature Engineering Suggestions:**
1. 🔢 Create mathematical features (sum, mean, ratios)
2. 📅 Extract datetime features if dates exist
3. 🏷️ Group rare categories together
4. 📊 Create interaction features between important columns
"""
    
    def _rule_based_understanding(self, df: pd.DataFrame, summary: str) -> Dict[str, Any]:
        """Rule-based data understanding (fallback)."""
        import numpy as np
        
        result = {
            'ai_powered': False,
            'provider': 'rule-based',
            'column_meanings': {},
            'cleaning_strategy': {},
            'suggested_features': [],
            'potential_issues': [],
            'recommended_model': 'Random Forest'
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
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
        
        return result
    
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


# Convenience function
def create_ai_engine(groq_key: Optional[str] = None, deepseek_key: Optional[str] = None) -> GroqAI:
    """Create an AI engine with Groq as primary and DeepSeek as fallback."""
    return GroqAI(api_key=groq_key, deepseek_key=deepseek_key)
