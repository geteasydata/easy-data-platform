"""
Evaluation Module - Model Evaluation and Metrics.

Provides comprehensive model evaluation:
    - metrics: Classification, regression, ranking metrics
    - comparison: Model comparison tools
    - reports: Automated evaluation reports
"""

from data_science_master_system.evaluation.metrics import (
    Evaluator,
    ClassificationMetrics,
    RegressionMetrics,
    calculate_metrics,
)
from data_science_master_system.evaluation.comparison import (
    ModelComparison,
    StatisticalTests,
)

__all__ = [
    "Evaluator",
    "ClassificationMetrics",
    "RegressionMetrics",
    "calculate_metrics",
    "ModelComparison",
    "StatisticalTests",
]
