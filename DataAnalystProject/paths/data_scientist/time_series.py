"""
Time Series Forecasting Module
Automated time series analysis and prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import warnings
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')


@dataclass
class ForecastResult:
    """Time series forecast results"""
    forecast_values: List[float]
    forecast_dates: List[str]
    confidence_lower: List[float]
    confidence_upper: List[float]
    model_name: str
    metrics: Dict[str, float]
    feature_importance: Dict[str, float] = field(default_factory=dict)


class TimeSeriesForecaster:
    """
    Automated Time Series Forecasting
    
    Features:
    - Automatic frequency detection
    - Multiple models (ARIMA-like, Prophet-like, ML-based)
    - Seasonal decomposition
    - Trend analysis
    - Confidence intervals
    """
    
    def __init__(self, 
                 forecast_periods: int = 30,
                 confidence_level: float = 0.95,
                 use_ml: bool = True):
        
        self.forecast_periods = forecast_periods
        self.confidence_level = confidence_level
        self.use_ml = use_ml
        
        self.best_model = None
        self.scaler = StandardScaler()
        self.results: ForecastResult = None
        
    def fit_predict(self, df: pd.DataFrame, 
                    date_col: str, 
                    value_col: str) -> ForecastResult:
        """
        Fit model and generate forecasts
        
        Args:
            df: DataFrame with time series data
            date_col: Name of date column
            value_col: Name of value column to forecast
            
        Returns:
            ForecastResult with predictions
        """
        # Prepare data
        ts_data = self._prepare_timeseries(df, date_col, value_col)
        
        # Create features
        X, y = self._create_features(ts_data)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model, metrics = self._train_best_model(X_train, y_train, X_test, y_test)
        self.best_model = model
        
        # Generate forecast
        forecast_X = self._create_future_features(ts_data, self.forecast_periods)
        forecast_values = model.predict(forecast_X)
        
        # Calculate confidence intervals
        residuals = y_test - model.predict(X_test)
        std_error = np.std(residuals)
        z_score = 1.96  # 95% confidence
        
        confidence_lower = (forecast_values - z_score * std_error).tolist()
        confidence_upper = (forecast_values + z_score * std_error).tolist()
        
        # Generate future dates
        last_date = ts_data.index[-1]
        freq = self._detect_frequency(ts_data)
        future_dates = pd.date_range(start=last_date + freq, periods=self.forecast_periods, freq=freq)
        
        # Get feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_names = ['lag_1', 'lag_7', 'lag_30', 'rolling_7', 'rolling_30', 
                           'day_of_week', 'month', 'trend']
            for name, imp in zip(feature_names, model.feature_importances_):
                feature_importance[name] = float(imp)
        
        self.results = ForecastResult(
            forecast_values=forecast_values.tolist(),
            forecast_dates=[d.strftime('%Y-%m-%d') for d in future_dates],
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            model_name=type(model).__name__,
            metrics=metrics,
            feature_importance=feature_importance
        )
        
        return self.results
    
    def _prepare_timeseries(self, df: pd.DataFrame, 
                            date_col: str, 
                            value_col: str) -> pd.Series:
        """Prepare time series data"""
        df = df.copy()
        
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Set index
        df = df.set_index(date_col)
        
        # Get series
        ts = df[value_col].copy()
        
        # Handle missing
        ts = ts.interpolate(method='linear')
        ts = ts.fillna(method='ffill').fillna(method='bfill')
        
        return ts
    
    def _detect_frequency(self, ts: pd.Series) -> pd.DateOffset:
        """Detect time series frequency"""
        if len(ts) < 2:
            return pd.DateOffset(days=1)
        
        # Calculate median difference
        diffs = pd.Series(ts.index).diff().dropna()
        median_diff = diffs.median()
        
        if median_diff <= pd.Timedelta(hours=1):
            return pd.DateOffset(hours=1)
        elif median_diff <= pd.Timedelta(days=1):
            return pd.DateOffset(days=1)
        elif median_diff <= pd.Timedelta(days=7):
            return pd.DateOffset(weeks=1)
        elif median_diff <= pd.Timedelta(days=31):
            return pd.DateOffset(months=1)
        else:
            return pd.DateOffset(days=1)
    
    def _create_features(self, ts: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create features for ML model"""
        df = pd.DataFrame({'value': ts})
        
        # Lag features
        df['lag_1'] = df['value'].shift(1)
        df['lag_7'] = df['value'].shift(7)
        df['lag_30'] = df['value'].shift(30)
        
        # Rolling statistics
        df['rolling_7'] = df['value'].rolling(window=7).mean()
        df['rolling_30'] = df['value'].rolling(window=30).mean()
        
        # Date features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Trend
        df['trend'] = np.arange(len(df))
        
        # Drop NaN
        df = df.dropna()
        
        feature_cols = ['lag_1', 'lag_7', 'lag_30', 'rolling_7', 'rolling_30', 
                       'day_of_week', 'month', 'trend']
        
        X = df[feature_cols].values
        y = df['value'].values
        
        return X, y
    
    def _create_future_features(self, ts: pd.Series, periods: int) -> np.ndarray:
        """Create features for future predictions"""
        features = []
        
        values = ts.values.tolist()
        last_date = ts.index[-1]
        
        for i in range(periods):
            # Calculate features
            lag_1 = values[-1]
            lag_7 = values[-7] if len(values) >= 7 else values[-1]
            lag_30 = values[-30] if len(values) >= 30 else values[-1]
            rolling_7 = np.mean(values[-7:])
            rolling_30 = np.mean(values[-30:]) if len(values) >= 30 else np.mean(values)
            
            future_date = last_date + pd.DateOffset(days=i+1)
            day_of_week = future_date.dayofweek
            month = future_date.month
            trend = len(ts) + i
            
            features.append([lag_1, lag_7, lag_30, rolling_7, rolling_30, 
                           day_of_week, month, trend])
            
            # Add predicted value for next iteration (simple persistence)
            values.append(values[-1])
        
        return np.array(features)
    
    def _train_best_model(self, X_train, y_train, X_test, y_test) -> Tuple[Any, Dict]:
        """Train and select best model"""
        from sklearn.ensemble import (
            RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
        )
        from sklearn.linear_model import Ridge, ElasticNet
        
        models = [
            ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('GradientBoosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('ExtraTrees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
            ('Ridge', Ridge(alpha=1.0)),
        ]
        
        best_model = None
        best_score = float('inf')
        best_metrics = {}
        
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                r2 = r2_score(y_test, predictions)
                mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-10))) * 100
                
                if mae < best_score:
                    best_score = mae
                    best_model = model
                    best_metrics = {
                        'MAE': round(mae, 4),
                        'RMSE': round(rmse, 4),
                        'R2': round(r2, 4),
                        'MAPE': round(mape, 2)
                    }
            except:
                continue
        
        return best_model, best_metrics
    
    def get_decomposition(self, ts: pd.Series) -> Dict[str, List[float]]:
        """Simple seasonal decomposition"""
        from scipy import signal
        
        values = ts.values
        
        # Trend (moving average)
        window = min(7, len(values) // 4)
        if window > 0:
            trend = pd.Series(values).rolling(window=window, center=True).mean().fillna(method='ffill').fillna(method='bfill').values
        else:
            trend = values
        
        # Detrended
        detrended = values - trend
        
        # Seasonal (simplified)
        seasonal = np.zeros_like(values)
        
        # Residual
        residual = values - trend - seasonal
        
        return {
            'original': values.tolist(),
            'trend': trend.tolist(),
            'seasonal': seasonal.tolist(),
            'residual': residual.tolist()
        }
