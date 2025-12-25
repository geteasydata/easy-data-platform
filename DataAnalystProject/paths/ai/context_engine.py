"""
Context Engine - Human-AI Bridge Module
=======================================
Bridges the gap between automated ML and expert human insight.
Handles:
1. Business Context Injection
2. Creative Feature Ideation via LLM
3. Safe Code Generation for Features
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .ai_assistant import AIAssistant

@dataclass
class CreativeFeature:
    name: str
    description: str
    justification: str
    code_snippet: Optional[str] = None
    complexity: str = "Medium"

class ContextEngine:
    def __init__(self, ai_assistant: AIAssistant):
        self.ai = ai_assistant
    
    def brainstorm_features(self, df: pd.DataFrame, context: str, lang: str = 'en') -> List[CreativeFeature]:
        """
        Ask LLM to brainstorm feature ideas based on data columns and user context.
        """
        if not self.ai.available:
            return []
            
        # Summary for prompt
        columns = df.columns.tolist()
        dtypes = df.dtypes.to_dict()
        sample = df.head(3).to_dict(orient='records')
        
        prompt = f"""
        You are a World-Class Data Scientist.
        I have a dataset with these columns: {columns}
        
        Data Types: {dtypes}
        Sample Data: {sample}
        
        USER CONTEXT (Very Important):
        "{context}"
        
        Task:
        Suggest 5 creative, non-obvious feature engineering ideas that would improve a machine learning model.
        Focus on:
        1. Domain-specific insights (based on context).
        2. Interaction features (Column A * Column B).
        3. Temporal features (if dates exist).
        4. Psychological/Behavioral proxies (if applicable).

        Return ONLY a JSON list of objects with keys: "name", "description", "justification", "complexity".
        Do not write code yet.
        """
        
        if lang == 'ar':
            prompt += "\nRespond in Arabic but keep JSON keys in English."
            
        try:
            response_text = self.ai.chat(prompt)
            # Clean response to ensure it's valid JSON
            import json
             # Find JSON block
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                features = []
                for item in data:
                    features.append(CreativeFeature(
                        name=item.get('name', 'Feature'),
                        description=item.get('description', ''),
                        justification=item.get('justification', ''),
                        complexity=item.get('complexity', 'Medium')
                    ))
                return features
            else:
                 # Fallback: simple parsing if JSON fails
                 return [CreativeFeature("Error", "Could not parse AI response", "Check API")]
                 
        except Exception as e:
            return [CreativeFeature("Error", f"AI generation failed: {str(e)}", "Check logs")]

    def generate_feature_code(self, df: pd.DataFrame, feature: CreativeFeature) -> str:
        """
        Ask LLM to generate the Pandas code for a specific feature.
        """
        if not self.ai.available:
            return ""
            
        columns = df.columns.tolist()
        dtypes = {k: str(v) for k, v in df.dtypes.items()}
        
        prompt = f"""
        Write a valid Python Pandas code snippet to create this feature:
        Feature Name: {feature.name}
        Description: {feature.description}
        
        Dataset Columns: {columns}
        Column Types: {dtypes}
        
        Requirements:
        1. Assume dataframe is named 'df'.
        2. RETURN ONLY THE CODE. No markdown, no comments, no explanations.
        3. Handle potential division by zero or missing values if needed.
        4. The result must be assigned to 'df["{feature.name}"]'.
        
        Example Output:
        df["NewFeature"] = df["ColA"] / df["ColB"].replace(0, 1)
        """
        
        try:
            code = self.ai.chat(prompt)
            # Clean code formatting
            code = code.replace("```python", "").replace("```", "").strip()
            return code
        except Exception as e:
            return f"# Error generating code: {e}"

    def safe_execute_feature(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        Execute the generated code safely (relatively).
        """
        local_scope = {'df': df.copy(), 'pd': pd, 'np': np}
        
        try:
            exec(code, {}, local_scope)
            return local_scope['df']
        except Exception as e:
            raise ValueError(f"Code execution failed: {e}")
