"""
NLP Analysis Module
Text analysis and sentiment detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


@dataclass
class NLPResult:
    """NLP analysis results"""
    text_columns: List[str]
    total_texts: int
    avg_length: float
    word_frequency: Dict[str, int]
    sentiment_summary: Dict[str, int]
    language_detected: str
    topics: List[str] = field(default_factory=list)


class TextAnalyzer:
    """
    NLP Text Analysis
    
    Features:
    - Automatic text column detection
    - Sentiment analysis (basic)
    - Word frequency analysis
    - Language detection
    - Topic extraction
    """
    
    def __init__(self, lang: str = 'en'):
        self.lang = lang
        self.results: NLPResult = None
        
        # Sentiment words (basic)
        self.positive_words_en = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'best', 'happy', 'perfect', 'beautiful', 'awesome',
            'nice', 'like', 'recommend', 'satisfied', 'helpful', 'quality'
        }
        
        self.negative_words_en = {
            'bad', 'terrible', 'horrible', 'awful', 'worst', 'hate',
            'poor', 'disappointed', 'waste', 'useless', 'broken', 'problem',
            'issue', 'fail', 'never', 'wrong', 'slow', 'expensive'
        }
        
        self.positive_words_ar = {
            'ممتاز', 'رائع', 'جيد', 'حلو', 'جميل', 'سعيد', 'مبسوط',
            'أفضل', 'أحسن', 'عظيم', 'راضي', 'سريع', 'نظيف'
        }
        
        self.negative_words_ar = {
            'سيء', 'مشكلة', 'بطيء', 'غالي', 'زعلان', 'خايب',
            'ضعيف', 'سبب', 'تأخير', 'إلغاء', 'فشل', 'خطأ'
        }
        
        # Stop words
        self.stop_words_en = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'
        }
        
        self.stop_words_ar = {
            'في', 'من', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'ذلك',
            'التي', 'الذي', 'هو', 'هي', 'نحن', 'أنت', 'هم', 'كان', 'كانت',
            'يكون', 'أن', 'لا', 'ما', 'قد', 'و', 'أو', 'ثم', 'لكن'
        }
    
    def analyze(self, df: pd.DataFrame, text_columns: List[str] = None) -> NLPResult:
        """
        Analyze text data
        
        Args:
            df: DataFrame with text data
            text_columns: List of text columns to analyze (auto-detect if None)
            
        Returns:
            NLPResult with analysis
        """
        # Detect text columns if not provided
        if text_columns is None:
            text_columns = self._detect_text_columns(df)
        
        if not text_columns:
            return NLPResult(
                text_columns=[],
                total_texts=0,
                avg_length=0,
                word_frequency={},
                sentiment_summary={},
                language_detected='unknown'
            )
        
        # Combine all text
        all_texts = []
        for col in text_columns:
            texts = df[col].dropna().astype(str).tolist()
            all_texts.extend(texts)
        
        # Detect language
        language = self._detect_language(all_texts)
        
        # Calculate statistics
        avg_length = np.mean([len(t) for t in all_texts]) if all_texts else 0
        
        # Word frequency
        word_freq = self._calculate_word_frequency(all_texts, language)
        
        # Sentiment analysis
        sentiment = self._analyze_sentiment(all_texts, language)
        
        # Extract topics
        topics = self._extract_topics(word_freq)
        
        self.results = NLPResult(
            text_columns=text_columns,
            total_texts=len(all_texts),
            avg_length=round(avg_length, 2),
            word_frequency=word_freq,
            sentiment_summary=sentiment,
            language_detected=language,
            topics=topics
        )
        
        return self.results
    
    def _detect_text_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns containing text data"""
        text_columns = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    avg_len = sample.astype(str).str.len().mean()
                    # Consider as text if average length > 20 chars
                    if avg_len > 20:
                        text_columns.append(col)
        
        return text_columns
    
    def _detect_language(self, texts: List[str]) -> str:
        """Detect predominant language"""
        if not texts:
            return 'unknown'
        
        # Sample texts
        sample = texts[:100]
        sample_text = ' '.join(sample)
        
        # Check for Arabic characters
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', sample_text))
        total_chars = len(sample_text)
        
        if total_chars > 0 and arabic_chars / total_chars > 0.3:
            return 'ar'
        return 'en'
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split by whitespace
        tokens = text.split()
        return tokens
    
    def _calculate_word_frequency(self, texts: List[str], language: str) -> Dict[str, int]:
        """Calculate word frequency"""
        all_words = []
        
        stop_words = self.stop_words_ar if language == 'ar' else self.stop_words_en
        
        for text in texts:
            tokens = self._tokenize(text)
            # Filter stop words and short words
            words = [w for w in tokens if w not in stop_words and len(w) > 2]
            all_words.extend(words)
        
        # Count frequencies
        freq = Counter(all_words)
        
        # Return top 50
        return dict(freq.most_common(50))
    
    def _analyze_sentiment(self, texts: List[str], language: str) -> Dict[str, int]:
        """Basic sentiment analysis"""
        positive = 0
        negative = 0
        neutral = 0
        
        if language == 'ar':
            pos_words = self.positive_words_ar
            neg_words = self.negative_words_ar
        else:
            pos_words = self.positive_words_en
            neg_words = self.negative_words_en
        
        for text in texts:
            tokens = set(self._tokenize(text))
            
            pos_count = len(tokens & pos_words)
            neg_count = len(tokens & neg_words)
            
            if pos_count > neg_count:
                positive += 1
            elif neg_count > pos_count:
                negative += 1
            else:
                neutral += 1
        
        total = positive + negative + neutral
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'positive_pct': round((positive / total) * 100, 1) if total > 0 else 0,
            'negative_pct': round((negative / total) * 100, 1) if total > 0 else 0
        }
    
    def _extract_topics(self, word_freq: Dict[str, int]) -> List[str]:
        """Extract main topics from word frequency"""
        # Get top 10 words as topics
        topics = list(word_freq.keys())[:10]
        return topics
    
    def get_word_cloud_data(self) -> List[Dict]:
        """Get data for word cloud visualization"""
        if not self.results:
            return []
        
        return [
            {'text': word, 'value': count}
            for word, count in self.results.word_frequency.items()
        ]
