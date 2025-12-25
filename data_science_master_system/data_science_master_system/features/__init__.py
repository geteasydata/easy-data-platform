"""
Features Module - Feature Engineering, Selection, and Transformation.

Provides comprehensive feature engineering:
    - engineering: Automated feature generation
    - selection: Feature selection methods
    - transformation: Feature transformers
"""

from data_science_master_system.features.engineering.feature_factory import (
    FeatureFactory,
    AutoFeatureGenerator,
)
from data_science_master_system.features.selection.feature_selection import (
    FeatureSelector,
    FilterSelector,
    WrapperSelector,
    EmbeddedSelector,
)
from data_science_master_system.features.transformation.transformers import (
    FeatureTransformer,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
    TargetEncoder,
)

__all__ = [
    # Engineering
    "FeatureFactory",
    "AutoFeatureGenerator",
    # Selection
    "FeatureSelector",
    "FilterSelector",
    "WrapperSelector",
    "EmbeddedSelector",
    # Transformation
    "FeatureTransformer",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "LabelEncoder",
    "OneHotEncoder",
    "TargetEncoder",
]
