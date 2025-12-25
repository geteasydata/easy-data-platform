"""
Advanced Visualizer Module
Complex interactive charts and geographic analysis
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import Dict, List, Any, Optional

class AdvancedVisualizer:
    """
    Advanced Visualization Tools
    
    Features:
    - Sunburst Charts (Hierarchical data)
    - Sankey Diagrams (Flow data)
    - 3D Scatter Plots
    - Geographic Maps (Choropleth, Scatter Geo)
    - Animation support
    """
    
    def __init__(self, lang: str = 'en'):
        self.lang = lang
        
    def plot_sunburst(self, df: pd.DataFrame, 
                      path_cols: List[str], 
                      value_col: str,
                      title: str = "Sunburst Chart") -> go.Figure:
        """Create interactive Sunburst chart"""
        if not path_cols or not value_col:
            return None
            
        fig = px.sunburst(
            df,
            path=path_cols,
            values=value_col,
            title=title,
            color=value_col,
            color_continuous_scale='RdBu'
        )
        self._update_layout(fig)
        return fig
    
    def plot_sankey(self, df: pd.DataFrame, 
                    source_col: str, 
                    target_col: str, 
                    value_col: str,
                    title: str = "Sankey Diagram") -> go.Figure:
        """Create Sankey diagram for flow analysis"""
        if not all([source_col, target_col, value_col]):
            return None
            
        # Create labels
        sources = df[source_col].unique().tolist()
        targets = df[target_col].unique().tolist()
        label_list = list(set(sources + targets))
        
        # Create indices
        source_indices = [label_list.index(s) for s in df[source_col]]
        target_indices = [label_list.index(t) for t in df[target_col]]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=label_list,
                color="blue"
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=df[value_col]
            )
        )])
        
        fig.update_layout(title_text=title, font_size=10)
        self._update_layout(fig)
        return fig
    
    def plot_3d_scatter(self, df: pd.DataFrame,
                        x: str, y: str, z: str,
                        color: str = None,
                        title: str = "3D Scatter Plot") -> go.Figure:
        """Create 3D scatter plot"""
        fig = px.scatter_3d(
            df, x=x, y=y, z=z,
            color=color,
            title=title,
            opacity=0.7
        )
        self._update_layout(fig)
        return fig
        
    def plot_geo_map(self, df: pd.DataFrame,
                     loc_col: str,
                     value_col: str,
                     location_mode: str = 'country names',
                     title: str = "Geographic Distribution") -> go.Figure:
        """
        Create geographic map
        
        location_mode options: 'ISO-3', 'USA-states', 'country names'
        """
        fig = px.choropleth(
            df,
            locations=loc_col,
            locationmode=location_mode,
            color=value_col,
            hover_name=loc_col,
            title=title,
            color_continuous_scale=px.colors.sequential.Plasma
        )
        self._update_layout(fig)
        return fig
    
    def plot_geo_scatter(self, df: pd.DataFrame,
                         lat_col: str, lon_col: str,
                         size_col: str = None,
                         color_col: str = None,
                         title: str = "Geographic Scatter") -> go.Figure:
        """Create scatter map using coordinates"""
        fig = px.scatter_geo(
            df,
            lat=lat_col,
            lon=lon_col,
            size=size_col,
            color=color_col,
            title=title,
            projection="natural earth"
        )
        self._update_layout(fig)
        return fig
    
    def _update_layout(self, fig):
        """Apply common layout settings"""
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
