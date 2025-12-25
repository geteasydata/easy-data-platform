"""
Data Storyteller Engine
=======================
Transforms raw data insights into compelling business narratives.
Uses GenAI to weave facts into a coherent story with chapters and strategic recommendations.
"""

from typing import List, Dict, Any
from dataclasses import asdict
from .insights import Insight
from paths.ai.ai_assistant import AIAssistant

class StorytellerEngine:
    def __init__(self, ai_assistant: AIAssistant):
        self.ai = ai_assistant

    def generate_narrative(self, df_summary: str, insights: List[Insight], context: str = "", lang: str = 'en') -> str:
        """
        Generate a narrative story based on insights and context.
        """
        if not self.ai.available:
            return "AI Storyteller unavailable. Please check API keys."

        # Prepare inputs for the LLM
        insights_text = "\n".join([f"- [{i.severity.upper()}] {i.title}: {i.description}" for i in insights])
        
        prompt = f"""
        You are an expert Data Storyteller and Strategy Consultant.
        
        TASK:
        Transform the following data insights into a professional "Executive Data Story".
        Do NOT just list the facts. Weave them into a narrative.
        
        DATA CONTEXT:
        {df_summary}
        
        USER CONTEXT (Business Background):
        "{context}"
        
        KEY INSIGHTS FOUND:
        {insights_text}
        
        STRUCTURE YOUR STORY AS FOLLOWS:
        
        # 1. The Executive Summary (The "Hook")
        A 3-sentence summary of the most critical findings. What is the headline?
        
        # 2. Chapter 1: The State of Affairs
        Describe the current situation based on descriptive stats (Data quality, volumes, general distribution).
        
        # 3. Chapter 2: The Hidden Patterns
        Discuss correlations, trends, and deeper findings. Connect the dots between different insights.
        
        # 4. Chapter 3: The Strategic Path Forward
        Actionable recommendations based on the findings. What should the business DO?
        
        TONE:
        Professional, insightful, and engaging. Avoid robotic language.
        Write in {lang.upper()}.
        """
        
        try:
            story = self.ai.chat(prompt)
            return story
        except Exception as e:
            return f"Error generating story: {str(e)}"
