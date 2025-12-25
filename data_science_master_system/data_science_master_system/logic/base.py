from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)

class BaseLogic(ABC):
    """
    Abstract Base Class for all Human Logic Layers.
    """
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"logic.{name}")

    @abstractmethod
    def execute(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the logic layer's reasoning.
        :param data: The primary data to analyze (DataFrame, Model, etc.)
        :param context: Shared memory/context from other logic layers.
        :return: Result dictionary containing insights, transformed data, or decisions.
        """
        pass
