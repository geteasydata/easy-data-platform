"""
Visualization Module.

Provides comprehensive visualization tools:
    - Plotter: Static visualization engine
    - DashboardBuilder: Interactive dashboards
    - AutoEDA: Automated exploratory data analysis
"""

from data_science_master_system.visualization.plotter import (
    Plotter,
    plot_distribution,
    plot_correlation,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_learning_curve,
)

__all__ = [
    "Plotter",
    "plot_distribution",
    "plot_correlation",
    "plot_feature_importance",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_learning_curve",
]
