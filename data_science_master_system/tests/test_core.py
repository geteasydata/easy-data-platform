"""
Test Suite for Data Science Master System.

Provides unit tests for all core components.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


class TestConfigManager:
    """Tests for ConfigManager."""
    
    def test_load_yaml(self, tmp_path):
        """Test loading YAML config."""
        from data_science_master_system.core.config_manager import ConfigManager
        
        # Create test config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
database:
  host: localhost
  port: 5432
app:
  debug: true
  name: test
""")
        
        config = ConfigManager()
        config.load(str(config_path))
        
        assert config.get("database.host") == "localhost"
        assert config.get("database.port") == 5432
        assert config.get("app.debug") is True
    
    def test_get_default(self):
        """Test get with default value."""
        from data_science_master_system.core.config_manager import ConfigManager
        
        config = ConfigManager()
        assert config.get("nonexistent", default="default") == "default"
    
    def test_env_override(self, monkeypatch):
        """Test environment variable override."""
        from data_science_master_system.core.config_manager import ConfigManager
        
        monkeypatch.setenv("DSMS_DATABASE_HOST", "production-db")
        
        config = ConfigManager()
        config.load_env(prefix="DSMS")
        
        assert config.get("database.host") == "production-db"


class TestExceptions:
    """Tests for custom exceptions."""
    
    def test_base_exception(self):
        """Test DSMSError base class."""
        from data_science_master_system.core.exceptions import DSMSError
        
        error = DSMSError("Test error", error_code="TEST001", context={"key": "value"})
        
        assert "Test error" in str(error)
        assert error.error_code == "TEST001"
        assert error.context["key"] == "value"
    
    def test_validation_error(self):
        """Test ValidationError."""
        from data_science_master_system.core.exceptions import ValidationError
        
        with pytest.raises(Exception) as exc_info:
            raise ValidationError("Invalid data")
        
        assert "Invalid data" in str(exc_info.value)


class TestLogger:
    """Tests for Logger."""
    
    def test_basic_logging(self):
        """Test basic logging functionality."""
        from data_science_master_system.core.logger import get_logger
        
        logger = get_logger("test")
        
        # Should not raise
        logger.info("Test message")
        logger.warning("Warning message")
        logger.debug("Debug message")
    
    def test_structured_logging(self):
        """Test structured logging with kwargs."""
        from data_science_master_system.core.logger import get_logger
        
        logger = get_logger("test")
        
        # Should not raise
        logger.info("Test message", key1="value1", key2=123)


class TestFileHandler:
    """Tests for FileHandler."""
    
    def test_read_csv(self, tmp_path):
        """Test reading CSV file."""
        from data_science_master_system.data.ingestion.file_handlers import FileHandler
        
        # Create test CSV
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n")
        
        handler = FileHandler(backend="pandas")
        df = handler.read(str(csv_path))
        
        assert len(df) == 2
        assert list(df.columns) == ["a", "b", "c"]
    
    def test_write_parquet(self, tmp_path):
        """Test writing Parquet file."""
        from data_science_master_system.data.ingestion.file_handlers import FileHandler
        
        handler = FileHandler(backend="pandas")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        parquet_path = tmp_path / "test.parquet"
        handler.write(df, str(parquet_path))
        
        assert parquet_path.exists()
        
        # Read back
        df_read = handler.read(str(parquet_path))
        assert len(df_read) == 3


class TestProcessingEngine:
    """Tests for ProcessingEngine."""
    
    def test_filter(self):
        """Test DataFrame filtering."""
        from data_science_master_system.data.processing import ProcessingEngine
        
        engine = ProcessingEngine(backend="pandas")
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        
        filtered = engine.filter(df, "a > 2")
        
        assert len(filtered) == 3
        assert filtered["a"].min() == 3
    
    def test_group_by(self):
        """Test group by aggregation."""
        from data_science_master_system.data.processing import ProcessingEngine
        
        engine = ProcessingEngine(backend="pandas")
        df = pd.DataFrame({
            "category": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40],
        })
        
        result = engine.group_by(df, "category").sum(["value"])
        
        assert len(result) == 2


class TestFeatureFactory:
    """Tests for FeatureFactory."""
    
    def test_numeric_features(self):
        """Test numeric feature generation."""
        from data_science_master_system.features.engineering.feature_factory import FeatureFactory
        
        factory = FeatureFactory()
        df = pd.DataFrame({"a": [1, 4, 9, 16], "b": [2, 4, 6, 8]})
        
        features = factory.generate_numeric_features(df)
        
        assert "a_sqrt" in features.columns
        assert "a_log" in features.columns
    
    def test_datetime_features(self):
        """Test datetime feature generation."""
        from data_science_master_system.features.engineering.feature_factory import FeatureFactory
        
        factory = FeatureFactory()
        dates = pd.Series(pd.date_range("2023-01-01", periods=10))
        
        features = factory.generate_datetime_features(dates, prefix="date")
        
        assert "date_year" in features.columns
        assert "date_month" in features.columns
        assert "date_dayofweek" in features.columns


class TestFeatureSelector:
    """Tests for FeatureSelector."""
    
    def test_filter_selection(self):
        """Test filter-based feature selection."""
        from data_science_master_system.features.selection.feature_selection import FeatureSelector
        
        np.random.seed(42)
        X = pd.DataFrame({
            "important": np.random.randn(100),
            "noise": np.random.randn(100) * 0.01,
            "target_related": np.linspace(0, 1, 100) + np.random.randn(100) * 0.1,
        })
        y = X["target_related"] + np.random.randn(100) * 0.1
        
        selector = FeatureSelector(method="filter", n_features=2)
        X_selected = selector.select(X, y)
        
        assert X_selected.shape[1] == 2


class TestTransformers:
    """Tests for Feature Transformers."""
    
    def test_standard_scaler(self):
        """Test StandardScaler."""
        from data_science_master_system.features.transformation.transformers import StandardScaler
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        assert np.abs(X_scaled.mean(axis=0)).max() < 1e-10
        assert np.abs(X_scaled.std(axis=0) - 1).max() < 1e-10
    
    def test_label_encoder(self):
        """Test LabelEncoder."""
        from data_science_master_system.features.transformation.transformers import LabelEncoder
        
        y = pd.Series(["cat", "dog", "cat", "bird"])
        
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        assert len(np.unique(y_encoded)) == 3
        
        y_decoded = encoder.inverse_transform(y_encoded)
        assert list(y_decoded) == list(y)


class TestModelFactory:
    """Tests for ModelFactory."""
    
    def test_create_random_forest(self):
        """Test creating RandomForest model."""
        from data_science_master_system.models.model_factory import ModelFactory
        
        model = ModelFactory.create("random_forest", problem_type="classification")
        
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
    
    def test_list_available(self):
        """Test listing available models."""
        from data_science_master_system.models.model_factory import ModelFactory
        
        models = ModelFactory.list_available("classification")
        
        assert "random_forest" in models
        assert "gradient_boosting" in models


class TestTraditionalMLModel:
    """Tests for TraditionalMLModel."""
    
    def test_fit_predict(self):
        """Test model fitting and prediction."""
        from data_science_master_system.models.traditional.traditional_ml import ClassificationModel
        
        np.random.seed(42)
        X = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        })
        y = (X["feature1"] + X["feature2"] > 0).astype(int)
        
        model = ClassificationModel("random_forest", n_estimators=10)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        from data_science_master_system.models.traditional.traditional_ml import ClassificationModel
        
        np.random.seed(42)
        X = pd.DataFrame({
            "important": np.linspace(0, 1, 100),
            "noise": np.random.randn(100) * 0.01,
        })
        y = (X["important"] > 0.5).astype(int)
        
        model = ClassificationModel("random_forest", n_estimators=10)
        model.fit(X, y)
        
        importance = model.feature_importance()
        
        assert "important" in importance["feature"].values
        assert importance.iloc[0]["feature"] == "important"  # Should be most important


class TestPipeline:
    """Tests for Pipeline."""
    
    def test_auto_detect(self):
        """Test pipeline auto-detection."""
        from data_science_master_system.pipeline import Pipeline
        
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "label": np.random.choice([0, 1], 100),
        })
        
        pipeline = Pipeline.auto_detect(df, target="label")
        
        assert pipeline.problem_type == "classification"
    
    def test_fit_predict(self):
        """Test pipeline fit and predict."""
        from data_science_master_system.pipeline import Pipeline
        
        np.random.seed(42)
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "label": np.random.choice([0, 1], 100),
        })
        
        pipeline = Pipeline.auto_detect(df, target="label")
        pipeline.fit()
        
        test_data = pd.DataFrame({
            "feature1": np.random.randn(10),
            "feature2": np.random.randn(10),
        })
        
        predictions = pipeline.predict(test_data)
        
        assert len(predictions) == 10


class TestEvaluator:
    """Tests for Evaluator."""
    
    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        from data_science_master_system.evaluation.metrics import Evaluator
        
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        evaluator = Evaluator(problem_type="classification")
        metrics = evaluator.evaluate(y_true, y_pred)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_regression_metrics(self):
        """Test regression metrics calculation."""
        from data_science_master_system.evaluation.metrics import Evaluator
        
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])
        
        evaluator = Evaluator(problem_type="regression")
        metrics = evaluator.evaluate(y_true, y_pred)
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics


class TestPlotter:
    """Tests for Plotter."""
    
    def test_distribution_plot(self):
        """Test distribution plot generation."""
        from data_science_master_system.visualization.plotter import Plotter
        
        plotter = Plotter()
        data = pd.Series(np.random.randn(100))
        
        fig = plotter.distribution(data, title="Test Distribution")
        
        assert fig is not None
    
    def test_correlation_matrix(self):
        """Test correlation matrix plot."""
        from data_science_master_system.visualization.plotter import Plotter
        
        plotter = Plotter()
        df = pd.DataFrame({
            "a": np.random.randn(50),
            "b": np.random.randn(50),
            "c": np.random.randn(50),
        })
        
        fig = plotter.correlation_matrix(df)
        
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
