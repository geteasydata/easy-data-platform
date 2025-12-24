"""
AI Engine - Multi-Provider AI for Expert Analysis
Primary: Groq (Llama 3.3) | Backup: DeepSeek | Fallback: Gemini
Provides intelligent insights and recommendations
"""

import os
from typing import Dict, Any, Optional
import pandas as pd

# Try to import Groq AI (Primary)
try:
    from core.groq_ai import GroqAI, create_ai_engine
    HAS_GROQ_MODULE = True
except ImportError:
    HAS_GROQ_MODULE = False

# Try to import Google Generative AI (Fallback)
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class AIEngine:
    """
    AI Engine with multi-provider support.
    Priority: Groq (Llama 3.3) > DeepSeek > Gemini > Rule-based
    Acts as a 30+ year expert data scientist.
    """
    
    def __init__(self, api_key: Optional[str] = None, deepseek_key: Optional[str] = None):
        self.api_key = api_key
        self.deepseek_key = deepseek_key
        self.model = None
        self.groq_engine = None
        self.is_configured = False
        self.active_provider = None
        
        # Try Groq first (Primary - with embedded key)
        if HAS_GROQ_MODULE:
            try:
                self.groq_engine = create_ai_engine(groq_key=api_key, deepseek_key=deepseek_key)
                if self.groq_engine.is_configured:
                    self.is_configured = True
                    self.active_provider = self.groq_engine.active_provider
                    print(f"✅ AI Engine: Using {self.active_provider}")
            except Exception as e:
                print(f"⚠️ Groq setup error: {e}")
        
        # Try Gemini as fallback
        if not self.is_configured and HAS_GEMINI:
            # Get Gemini key from environment or st.secrets - NEVER hardcode!
            gemini_key = api_key or os.getenv('GEMINI_API_KEY')
            if gemini_key:
                try:
                    genai.configure(api_key=gemini_key)
                    self.model = genai.GenerativeModel('gemini-pro')
                    self.is_configured = True
                    self.active_provider = 'gemini'
                    print("✅ AI Engine: Using Gemini")
                except Exception as e:
                    print(f"⚠️ Gemini setup error: {e}")
    
    def analyze_data_quality(self, analysis: Dict[str, Any], lang: str = 'ar') -> str:
        """Generate AI insights about data quality."""
        if not self.is_configured:
            return self._fallback_data_quality(analysis, lang)
        
        # Use Groq engine if available (primary)
        if self.groq_engine and self.groq_engine.is_configured:
            return self.groq_engine.analyze_data_quality(analysis, lang)
        
        # Fallback to Gemini
        prompt = f"""
        You are a senior data scientist with 30+ years of experience.
        Analyze this dataset summary and provide expert insights:
        
        - Rows: {analysis['rows']}
        - Columns: {analysis['columns']}
        - Numeric columns: {len(analysis['numeric_columns'])}
        - Categorical columns: {len(analysis['categorical_columns'])}
        - Missing values: {analysis['total_missing']}
        - Duplicates: {analysis['duplicates']}
        
        Provide 3-5 key observations and recommendations.
        Language: {'Arabic' if lang == 'ar' else 'English'}
        Format: Bullet points with emojis for visual appeal.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return self._fallback_data_quality(analysis, lang)
    
    def generate_insights(self, results: Dict[str, Any], target_col: str, lang: str = 'ar') -> str:
        """Generate strategic business insights from ML results."""
        if not self.is_configured:
            return self._fallback_insights(results, target_col, lang)
        
        # Use Groq engine if available (primary)
        if self.groq_engine and self.groq_engine.is_configured:
            return self.groq_engine.generate_insights(results, target_col, lang)
        
        # Fallback to Gemini
        top_features = results['feature_importance'].head(5)['Feature'].tolist()
        
        prompt = f"""
        You are a senior data scientist with 30+ years of experience advising Fortune 500 companies.
        Based on the ML analysis results below, provide strategic business recommendations:
        
        Problem Type: {results['problem_type']}
        Best Model: {results['best_model']}
        Performance Metrics: {results['metrics']}
        Top 5 Important Features: {top_features}
        Target Variable: {target_col}
        
        Provide:
        1. Performance interpretation (is it good? what does it mean for business?)
        2. Top 3 strategic recommendations based on feature importance
        3. Data quality notes and improvement suggestions
        4. Next steps for production deployment
        
        Language: {'Arabic' if lang == 'ar' else 'English'}
        Format: Professional report style with emojis and clear sections.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return self._fallback_insights(results, target_col, lang)
    
    def suggest_feature_engineering(self, df: pd.DataFrame, lang: str = 'ar') -> str:
        """Suggest feature engineering ideas."""
        if not self.is_configured:
            return self._fallback_feature_suggestions(df, lang)
        
        # Use Groq engine if available (primary)
        if self.groq_engine and self.groq_engine.is_configured:
            return self.groq_engine.suggest_feature_engineering(df, lang)
        
        # Fallback to Gemini
        columns_info = {col: str(df[col].dtype) for col in df.columns[:20]}
        
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
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return self._fallback_feature_suggestions(df, lang)
    
    def _fallback_data_quality(self, analysis: Dict[str, Any], lang: str) -> str:
        """Fallback when AI is not available."""
        if lang == 'ar':
            insights = []
            insights.append(f"📊 **حجم البيانات**: {analysis['rows']} صف × {analysis['columns']} عمود")
            
            if analysis['total_missing'] > 0:
                pct = (analysis['total_missing'] / (analysis['rows'] * analysis['columns'])) * 100
                insights.append(f"⚠️ **قيم مفقودة**: {analysis['total_missing']} ({pct:.1f}%) - تم معالجتها تلقائياً")
            else:
                insights.append("✅ **لا توجد قيم مفقودة**")
            
            if analysis['duplicates'] > 0:
                insights.append(f"🗑️ **صفوف مكررة**: {analysis['duplicates']} - تم إزالتها")
            
            if analysis['numeric_columns']:
                insights.append(f"🔢 **أعمدة رقمية**: {len(analysis['numeric_columns'])}")
            if analysis['categorical_columns']:
                insights.append(f"🏷️ **أعمدة نصية**: {len(analysis['categorical_columns'])}")
            
            return "\n".join(insights)
        else:
            insights = []
            insights.append(f"📊 **Data Size**: {analysis['rows']} rows × {analysis['columns']} columns")
            
            if analysis['total_missing'] > 0:
                pct = (analysis['total_missing'] / (analysis['rows'] * analysis['columns'])) * 100
                insights.append(f"⚠️ **Missing Values**: {analysis['total_missing']} ({pct:.1f}%) - Auto-handled")
            else:
                insights.append("✅ **No missing values**")
            
            if analysis['duplicates'] > 0:
                insights.append(f"🗑️ **Duplicate Rows**: {analysis['duplicates']} - Removed")
            
            if analysis['numeric_columns']:
                insights.append(f"🔢 **Numeric Columns**: {len(analysis['numeric_columns'])}")
            if analysis['categorical_columns']:
                insights.append(f"🏷️ **Categorical Columns**: {len(analysis['categorical_columns'])}")
            
            return "\n".join(insights)
    
    def _fallback_insights(self, results: Dict[str, Any], target_col: str, lang: str) -> str:
        """Fallback insights when AI is not available."""
        metrics = results['metrics']
        top_features = results['feature_importance'].head(3)['Feature'].tolist()
        
        if lang == 'ar':
            insights = []
            
            # Performance
            if results['problem_type'] == 'classification':
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
            
            # Features
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
            
            if results['problem_type'] == 'classification':
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
