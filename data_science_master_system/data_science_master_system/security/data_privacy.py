"""
Privacy and Security Module.

Implements data privacy, differential privacy, and PII detection.
"""

from typing import Any, Dict, List, Optional, Set
import re
import hashlib
import numpy as np
import pandas as pd

from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class PIIDetector:
    """Detect and handle Personally Identifiable Information."""
    
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        'date_of_birth': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    }
    
    def __init__(self, custom_patterns: Dict[str, str] = None):
        self.patterns = {**self.PATTERNS, **(custom_patterns or {})}
        self.compiled_patterns = {name: re.compile(pattern) for name, pattern in self.patterns.items()}
    
    def detect(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text."""
        found = {}
        for name, pattern in self.compiled_patterns.items():
            matches = pattern.findall(str(text))
            if matches:
                found[name] = matches
        return found
    
    def scan_dataframe(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Scan DataFrame for PII."""
        results = {}
        for col in df.columns:
            col_results = {}
            for name in self.patterns:
                count = df[col].astype(str).str.contains(self.compiled_patterns[name], na=False).sum()
                if count > 0:
                    col_results[name] = count
            if col_results:
                results[col] = col_results
        return results
    
    def mask(self, text: str, pii_type: str = None) -> str:
        """Mask PII in text."""
        result = str(text)
        patterns_to_use = {pii_type: self.compiled_patterns[pii_type]} if pii_type else self.compiled_patterns
        for name, pattern in patterns_to_use.items():
            result = pattern.sub(f'[{name.upper()}_MASKED]', result)
        return result


class DataAnonymizer:
    """Anonymize datasets for privacy compliance."""
    
    def __init__(self, salt: str = 'default_salt'):
        self.salt = salt
        self.pii_detector = PIIDetector()
    
    def hash_column(self, series: pd.Series) -> pd.Series:
        """Hash values in a column."""
        return series.apply(lambda x: hashlib.sha256(f"{self.salt}{x}".encode()).hexdigest()[:16] if pd.notna(x) else x)
    
    def generalize_age(self, series: pd.Series, bins: List[int] = [0, 18, 30, 50, 70, 100]) -> pd.Series:
        """Generalize age to ranges."""
        labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
        return pd.cut(series, bins=bins, labels=labels)
    
    def generalize_location(self, series: pd.Series, level: str = 'city') -> pd.Series:
        """Generalize location (placeholder)."""
        if level == 'region':
            return series.str[:3] + '***'
        return series
    
    def k_anonymize(self, df: pd.DataFrame, quasi_identifiers: List[str], k: int = 5) -> pd.DataFrame:
        """Basic k-anonymization by suppressing small groups."""
        group_sizes = df.groupby(quasi_identifiers).size()
        small_groups = group_sizes[group_sizes < k].index
        
        result = df.copy()
        for qi in quasi_identifiers:
            result.loc[result.set_index(quasi_identifiers).index.isin(small_groups), qi] = '*'
        
        return result
    
    def anonymize(self, df: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
        """
        Anonymize DataFrame based on config.
        
        Config example:
            {'email': 'hash', 'age': 'generalize', 'name': 'drop'}
        """
        result = df.copy()
        
        for col, method in config.items():
            if col not in result.columns:
                continue
            
            if method == 'hash':
                result[col] = self.hash_column(result[col])
            elif method == 'drop':
                result = result.drop(columns=[col])
            elif method == 'mask':
                result[col] = result[col].apply(lambda x: self.pii_detector.mask(str(x)))
            elif method == 'generalize' and 'age' in col.lower():
                result[col] = self.generalize_age(result[col])
        
        return result


class DifferentialPrivacy:
    """Implement differential privacy mechanisms."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def laplace_mechanism(self, value: float, sensitivity: float) -> float:
        """Add Laplace noise."""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def gaussian_mechanism(self, value: float, sensitivity: float) -> float:
        """Add Gaussian noise."""
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma)
        return value + noise
    
    def randomized_response(self, value: bool, p: float = None) -> bool:
        """Randomized response for binary data."""
        if p is None:
            p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        
        if np.random.random() < p:
            return value
        return np.random.choice([True, False])
    
    def private_mean(self, values: np.ndarray, bounds: Tuple[float, float]) -> float:
        """Compute differentially private mean."""
        clipped = np.clip(values, bounds[0], bounds[1])
        sensitivity = (bounds[1] - bounds[0]) / len(values)
        true_mean = np.mean(clipped)
        return self.laplace_mechanism(true_mean, sensitivity)
    
    def private_count(self, count: int) -> int:
        """Compute differentially private count."""
        return max(0, int(self.laplace_mechanism(count, 1)))


from typing import Tuple


class ComplianceChecker:
    """Check for GDPR/CCPA compliance."""
    
    def __init__(self):
        self.pii_detector = PIIDetector()
    
    def check_gdpr_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check GDPR compliance requirements."""
        issues = []
        
        # Check for PII
        pii_found = self.pii_detector.scan_dataframe(df)
        if pii_found:
            issues.append(f"PII detected in columns: {list(pii_found.keys())}")
        
        # Check for consent column
        consent_cols = [c for c in df.columns if 'consent' in c.lower()]
        if not consent_cols:
            issues.append("No consent tracking column found")
        
        return {
            'compliant': len(issues) == 0,
            'issues': issues,
            'pii_detected': pii_found,
            'recommendations': self._get_recommendations(issues)
        }
    
    def _get_recommendations(self, issues: List[str]) -> List[str]:
        recommendations = []
        for issue in issues:
            if 'PII' in issue:
                recommendations.append("Consider anonymizing or pseudonymizing PII data")
            if 'consent' in issue:
                recommendations.append("Add consent tracking mechanism")
        return recommendations
