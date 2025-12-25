"""
AI Assistant Module
Integration with LLMs for intelligent data analysis
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd

# Try to import AI libraries
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class AIResponse:
    """AI assistant response"""
    answer: str
    suggestions: List[str]
    confidence: float
    model_used: str


class AIAssistant:
    """
    AI-Powered Data Analysis Assistant
    
    Features:
    - Natural language data queries
    - Intelligent insights generation
    - Recommendations based on data patterns
    - Explanation of analysis results
    - Arabic/English support
    """
    
    # Default API key for Gemini (shared with AI_Expert_App)
    DEFAULT_GEMINI_KEY = "AIzaSyC_0gsd0E7_Xf3g64ReTlVCvrgM7m2spwE"
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = 'gemini',
                 lang: str = 'en'):
        
        self.lang = lang
        self.model = model
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY') or os.environ.get('OPENAI_API_KEY') or self.DEFAULT_GEMINI_KEY
        
        self._setup_model()
    
    def _setup_model(self):
        """Setup AI model"""
        if self.model == 'gemini' and HAS_GEMINI and self.api_key:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel('gemini-2.5-flash')
            self.available = True
        elif self.model == 'openai' and HAS_OPENAI and self.api_key:
            openai.api_key = self.api_key
            self.client = openai
            self.available = True
        else:
            self.available = False
    
    def analyze_data(self, df: pd.DataFrame, question: str = None) -> AIResponse:
        """
        Analyze data using AI
        
        Args:
            df: DataFrame to analyze
            question: Specific question about the data
            
        Returns:
            AIResponse with insights
        """
        if not self.available:
            return self._fallback_analysis(df)
        
        # Prepare data summary
        data_summary = self._prepare_data_summary(df)
        
        # Create prompt
        prompt = self._create_prompt(data_summary, question)
        
        try:
            if self.model == 'gemini':
                response = self.client.generate_content(prompt)
                answer = response.text
            else:
                response = self.client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message.content
            
            return AIResponse(
                answer=answer,
                suggestions=self._extract_suggestions(answer),
                confidence=0.85,
                model_used=self.model
            )
        except Exception as e:
            return self._fallback_analysis(df, str(e))
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> str:
        """Prepare data summary for AI"""
        summary = f"""
Dataset Overview:
- Rows: {len(df):,}
- Columns: {len(df.columns)}
- Column names: {', '.join(df.columns.tolist()[:20])}
- Data types: {dict(df.dtypes.value_counts())}

Missing Values:
{df.isnull().sum().to_dict()}

Numeric Columns Statistics:
{df.describe().to_dict()}

Sample Data (first 5 rows):
{df.head().to_string()}
"""
        return summary
    
    def _create_prompt(self, data_summary: str, question: str = None) -> str:
        """Create prompt for AI"""
        if self.lang == 'ar':
            base_prompt = f"""
أنت محلل بيانات خبير. قم بتحليل البيانات التالية وقدم رؤى قيمة.

{data_summary}

"""
            if question:
                base_prompt += f"السؤال المحدد: {question}\n"
            
            base_prompt += """
قدم:
1. ملخص تنفيذي (3-5 نقاط)
2. الرؤى الرئيسية
3. التوصيات العملية
4. المخاطر المحتملة

أجب باللغة العربية.
"""
        else:
            base_prompt = f"""
You are an expert data analyst. Analyze the following data and provide valuable insights.

{data_summary}

"""
            if question:
                base_prompt += f"Specific question: {question}\n"
            
            base_prompt += """
Provide:
1. Executive summary (3-5 points)
2. Key insights
3. Actionable recommendations
4. Potential risks

Be concise and practical.
"""
        return base_prompt
    
    def _extract_suggestions(self, answer: str) -> List[str]:
        """Extract actionable suggestions from AI response"""
        suggestions = []
        
        # Simple extraction - look for numbered items or bullet points
        lines = answer.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Clean up the line
                clean = line.lstrip('0123456789.-•) ')
                if len(clean) > 10:
                    suggestions.append(clean)
        
        return suggestions[:5]  # Return top 5
    
    def _fallback_analysis(self, df: pd.DataFrame, error: str = None) -> AIResponse:
        """Fallback analysis when AI is not available"""
        insights = []
        
        # Basic statistics-based insights
        if self.lang == 'ar':
            insights.append(f"مجموعة البيانات تحتوي على {len(df):,} صف و {len(df.columns)} عمود")
            
            missing_pct = (df.isnull().sum().sum() / df.size) * 100
            if missing_pct > 10:
                insights.append(f"تحذير: نسبة البيانات المفقودة {missing_pct:.1f}%")
            
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                insights.append(f"يوجد {duplicates:,} صف مكرر")
        else:
            insights.append(f"Dataset contains {len(df):,} rows and {len(df.columns)} columns")
            
            missing_pct = (df.isnull().sum().sum() / df.size) * 100
            if missing_pct > 10:
                insights.append(f"Warning: {missing_pct:.1f}% missing data")
            
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                insights.append(f"Found {duplicates:,} duplicate rows")
        
        answer = '\n'.join([f"• {i}" for i in insights])
        
        if error:
            if self.lang == 'ar':
                answer = f"⚠️ الذكاء الاصطناعي غير متوفر. تحليل أساسي:\n\n{answer}"
            else:
                answer = f"⚠️ AI not available. Basic analysis:\n\n{answer}"
        
        return AIResponse(
            answer=answer,
            suggestions=insights,
            confidence=0.5,
            model_used='fallback'
        )
    
    def chat(self, message: str, context: Dict = None) -> str:
        """
        Chat with AI about data
        
        Args:
            message: User message
            context: Additional context (data summary, previous insights, etc.)
            
        Returns:
            AI response string
        """
        if not self.available:
            if self.lang == 'ar':
                return "عذراً، الذكاء الاصطناعي غير متوفر. يرجى التحقق من مفتاح API."
            return "Sorry, AI is not available. Please check your API key."
        
        prompt = message
        if context:
            prompt = f"Context:\n{json.dumps(context, ensure_ascii=False)}\n\nQuestion: {message}"
        
        try:
            if self.model == 'gemini':
                response = self.client.generate_content(prompt)
                return response.text
            else:
                response = self.client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def explain_results(self, results: Dict, audience: str = 'business') -> str:
        """
        Explain analysis results for different audiences
        
        Args:
            results: Analysis results to explain
            audience: 'business', 'technical', or 'executive'
            
        Returns:
            Explanation text
        """
        if self.lang == 'ar':
            audience_prompts = {
                'business': "اشرح هذه النتائج لمدير أعمال بأسلوب بسيط وعملي",
                'technical': "اشرح هذه النتائج بالتفصيل التقني",
                'executive': "قدم ملخص تنفيذي موجز (3 نقاط فقط)"
            }
        else:
            audience_prompts = {
                'business': "Explain these results for a business manager in simple, practical terms",
                'technical': "Provide a detailed technical explanation",
                'executive': "Give a brief executive summary (3 points only)"
            }
        
        prompt = f"{audience_prompts.get(audience, audience_prompts['business'])}\n\nResults:\n{json.dumps(results, ensure_ascii=False, indent=2)}"
        
        if not self.available:
            return str(results)
        
        try:
            if self.model == 'gemini':
                response = self.client.generate_content(prompt)
                return response.text
            else:
                response = self.client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
        except Exception as e:
            return str(results)
