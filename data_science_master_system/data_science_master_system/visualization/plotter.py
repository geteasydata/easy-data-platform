"""
Visualization Engine for Data Science Master System.

Provides beautiful, publication-ready visualizations:
- Distribution plots
- Correlation matrices
- Feature importance
- Model performance (ROC, confusion matrix, etc.)
- Learning curves

Example:
    >>> plotter = Plotter(style="publication")
    >>> plotter.distribution(df["age"])
    >>> plotter.correlation_matrix(df)
    >>> plotter.save("analysis.png")
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path

from data_science_master_system.core.base_classes import BaseVisualizer
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None


class Plotter(BaseVisualizer):
    """
    Publication-quality visualization engine.
    
    Example:
        >>> plotter = Plotter(style="publication", figsize=(10, 6))
        >>> 
        >>> # Distribution plot
        >>> fig = plotter.distribution(df["age"], title="Age Distribution")
        >>> 
        >>> # Correlation matrix
        >>> fig = plotter.correlation_matrix(df)
        >>> 
        >>> # Save figure
        >>> plotter.save(fig, "analysis.png", dpi=300)
    """
    
    STYLES = {
        "default": {
            "figure.figsize": (10, 6),
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        },
        "publication": {
            "figure.figsize": (8, 6),
            "font.size": 11,
            "font.family": "serif",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
        },
        "presentation": {
            "figure.figsize": (12, 8),
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
        },
        "dark": {
            "figure.figsize": (10, 6),
            "figure.facecolor": "#1a1a2e",
            "axes.facecolor": "#16213e",
            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
        },
    }
    
    PALETTES = {
        "default": ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12", "#1abc9c"],
        "categorical": ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f", "#edc949"],
        "sequential": "viridis",
        "diverging": "RdBu_r",
    }
    
    def __init__(
        self,
        style: str = "default",
        palette: str = "default",
        figsize: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Initialize plotter.
        
        Args:
            style: Style preset ("default", "publication", "presentation", "dark")
            palette: Color palette
            figsize: Default figure size
        """
        super().__init__()
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib not installed. Install with: pip install matplotlib")
        
        self.style = style
        self.palette = palette
        self.figsize = figsize or self.STYLES.get(style, {}).get("figure.figsize", (10, 6))
        
        self._apply_style()
    
    def _apply_style(self) -> None:
        """Apply the selected style."""
        if self.style in self.STYLES:
            plt.rcParams.update(self.STYLES[self.style])
        
        if SEABORN_AVAILABLE:
            if self.palette in self.PALETTES:
                colors = self.PALETTES[self.palette]
                if isinstance(colors, list):
                    sns.set_palette(colors)
    
    def visualize(self, data: Any, **kwargs: Any) -> Any:
        """Generic visualization method."""
        if isinstance(data, pd.DataFrame):
            return self.pairplot(data, **kwargs)
        elif isinstance(data, pd.Series):
            return self.distribution(data, **kwargs)
        else:
            return self.distribution(pd.Series(data), **kwargs)
    
    def distribution(
        self,
        data: Union[pd.Series, np.ndarray],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        bins: int = 30,
        kde: bool = True,
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot distribution of a variable.
        
        Args:
            data: Data to plot
            title: Plot title
            xlabel: X-axis label
            bins: Number of bins
            kde: Show kernel density estimate
            ax: Matplotlib axes
            
        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()
        
        if SEABORN_AVAILABLE:
            sns.histplot(data, bins=bins, kde=kde, ax=ax)
        else:
            ax.hist(data, bins=bins, density=kde, alpha=0.7)
        
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        
        plt.tight_layout()
        return fig
    
    def correlation_matrix(
        self,
        data: pd.DataFrame,
        method: str = "pearson",
        annot: bool = True,
        cmap: str = "RdBu_r",
        title: Optional[str] = None,
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot correlation matrix heatmap.
        
        Args:
            data: DataFrame with numeric columns
            method: Correlation method
            annot: Show annotations
            cmap: Colormap
            title: Plot title
            ax: Matplotlib axes
            
        Returns:
            Figure object
        """
        corr = data.corr(method=method)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()
        
        if SEABORN_AVAILABLE:
            sns.heatmap(
                corr, annot=annot, cmap=cmap, center=0,
                vmin=-1, vmax=1, ax=ax, fmt=".2f",
                square=True, linewidths=0.5,
            )
        else:
            im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def feature_importance(
        self,
        importance: Union[pd.DataFrame, Dict[str, float]],
        top_n: int = 20,
        title: str = "Feature Importance",
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot feature importance.
        
        Args:
            importance: Feature importance (DataFrame with 'feature' and 'importance' columns, or dict)
            top_n: Number of top features to show
            title: Plot title
            ax: Matplotlib axes
            
        Returns:
            Figure object
        """
        if isinstance(importance, dict):
            importance = pd.DataFrame({
                "feature": list(importance.keys()),
                "importance": list(importance.values()),
            })
        
        importance = importance.nlargest(top_n, "importance")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()
        
        if SEABORN_AVAILABLE:
            sns.barplot(
                data=importance, x="importance", y="feature",
                palette="viridis", ax=ax,
            )
        else:
            ax.barh(importance["feature"], importance["importance"])
        
        ax.set_title(title)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        
        plt.tight_layout()
        return fig
    
    def confusion_matrix(
        self,
        cm: np.ndarray,
        labels: Optional[List[str]] = None,
        normalize: bool = False,
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            normalize: Normalize values
            title: Plot title
            cmap: Colormap
            ax: Matplotlib axes
            
        Returns:
            Figure object
        """
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()
        
        if SEABORN_AVAILABLE:
            sns.heatmap(
                cm, annot=True, cmap=cmap, ax=ax,
                fmt=".2f" if normalize else "d",
                xticklabels=labels, yticklabels=labels,
            )
        else:
            im = ax.imshow(cm, cmap=cmap)
            plt.colorbar(im, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
        plt.tight_layout()
        return fig
    
    def roc_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        title: str = "ROC Curve",
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_score: Predicted scores
            title: Plot title
            ax: Matplotlib axes
            
        Returns:
            Figure object
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()
        
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        ax.fill_between(fpr, tpr, alpha=0.3)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def learning_curve(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        title: str = "Learning Curve",
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot learning curve.
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores
            val_scores: Validation scores
            title: Plot title
            ax: Matplotlib axes
            
        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        
        ax.plot(train_sizes, val_mean, "o-", color="g", label="Validation score")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")
        
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def pairplot(
        self,
        data: pd.DataFrame,
        hue: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Any:
        """
        Create pairwise scatter plot matrix.
        
        Args:
            data: DataFrame
            hue: Column for color coding
            title: Plot title
            
        Returns:
            Figure object
        """
        if not SEABORN_AVAILABLE:
            raise ImportError("Seaborn required for pairplot")
        
        g = sns.pairplot(data, hue=hue, diag_kind="kde")
        if title:
            g.fig.suptitle(title, y=1.02)
        
        return g.fig
    
    def save(
        self,
        fig: Any,
        path: str,
        dpi: int = 150,
        bbox_inches: str = "tight",
    ) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Figure object
            path: Output path
            dpi: Resolution
            bbox_inches: Bounding box setting
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"Figure saved to {path}")


# Convenience functions
def plot_distribution(data: Union[pd.Series, np.ndarray], **kwargs: Any) -> Any:
    """Quick distribution plot."""
    return Plotter().distribution(data, **kwargs)


def plot_correlation(data: pd.DataFrame, **kwargs: Any) -> Any:
    """Quick correlation matrix plot."""
    return Plotter().correlation_matrix(data, **kwargs)


def plot_feature_importance(importance: Union[pd.DataFrame, Dict], **kwargs: Any) -> Any:
    """Quick feature importance plot."""
    return Plotter().feature_importance(importance, **kwargs)


def plot_confusion_matrix(cm: np.ndarray, **kwargs: Any) -> Any:
    """Quick confusion matrix plot."""
    return Plotter().confusion_matrix(cm, **kwargs)


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, **kwargs: Any) -> Any:
    """Quick ROC curve plot."""
    return Plotter().roc_curve(y_true, y_score, **kwargs)


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    **kwargs: Any,
) -> Any:
    """Quick learning curve plot."""
    return Plotter().learning_curve(train_sizes, train_scores, val_scores, **kwargs)
