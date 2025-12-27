"""
Data Science Dual-Path Application
Main Entry Point with Streamlit UI - Bilingual Support
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import io
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import translations using explicit path to avoid conflicts with other projects
import importlib.util
_translations_path = PROJECT_ROOT / "translations.py"
_spec = importlib.util.spec_from_file_location("dap_translations", _translations_path)
_translations_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_translations_module)
t = _translations_module.t
TRANSLATIONS = _translations_module.TRANSLATIONS

# Import modules
from config import DOMAINS, TOOLS, OUTPUT_FORMATS, BASE_DIR
from scripts.load_data import DataLoader
from paths.data_analyst.analyzer import DataAnalyzer
from paths.data_analyst.cleaner import DataCleaner, CleaningTool
from paths.data_analyst.insights import InsightsGenerator
from paths.data_analyst.dashboard_gen import DashboardGenerator
from paths.data_scientist.ml_pipeline import MLPipeline
from paths.data_scientist.feature_engineer import FeatureEngineer
from paths.data_scientist.model_evaluator import ModelEvaluator
from paths.data_scientist.model_explainer import ModelExplainer

# Try to import advanced modules
try:
    from paths.data_scientist.automl_engine import AdvancedAutoML, AutoMLResult
    HAS_AUTOML = True
except ImportError:
    HAS_AUTOML = False

try:
    from paths.data_scientist.advanced_features import AdvancedFeatureEngineer
    HAS_ADVANCED_FE = True
except ImportError:
    HAS_ADVANCED_FE = False

# Import new advanced modules
try:
    from paths.data_analyst.auto_eda import AutoEDA
    from paths.data_scientist.time_series import TimeSeriesForecaster
    from paths.deployment.api_generator import APIGenerator
    from paths.reports.pdf_generator import generate_quick_pdf
    from paths.data_analyst.nlp_analysis import TextAnalyzer
    from paths.ai.ai_assistant import AIAssistant
    from paths.ai.chief_data_scientist import ChiefDataScientist # NEW
    from paths.data_analyst.advanced_visualizer import AdvancedVisualizer
    from paths.data_scientist.what_if_analysis import WhatIfSimulator
    from paths.reports.excel_generator import create_excel_report
    from paths.reports.powerbi_generator import create_powerbi_package
    from core.sentinel import get_sentinel
    HAS_NEW_FEATURES = True
except ImportError:
    HAS_NEW_FEATURES = False
    print("Warning: Could not import some new advanced features")

# Add Omni-Logic path
BRAIN_PATH = Path(__file__).parent.parent / "data_science_master_system"
if str(BRAIN_PATH) not in sys.path:
    sys.path.append(str(BRAIN_PATH))

# Import Omni-Logic Layers
try:
    from data_science_master_system.logic import (
        AnalyticalLogic, EthicalLogic, CausalLogic
    )
    HAS_OMNI_LOGIC = True
except ImportError:
    HAS_OMNI_LOGIC = False
    print("Warning: Could not import Omni-Logic layers")

# Page Configuration
st.set_page_config(
    page_title="Easy Data | البيانات السهلة",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# Custom CSS - Premium Dark Theme
st.markdown("""
<style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Cairo', sans-serif;
    }

    /* Main Background */
    .stApp {
        background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
        color: #e2e8f0;
    }

    /* Hide Streamlit Header */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    footer {
        display: none !important;
    }

    /* Card Styling (Glassmorphism) */
    .path-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem !important; /* Reduced padding */
        border-radius: 20px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.5rem 0 !important; /* Reduced margin */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }

    .path-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: 0.5s;
    }

    .path-card:hover::before {
        left: 100%;
    }

    .path-card:hover {
        transform: translateY(-8px) scale(1.02);
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.3);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.5);
    }

    /* Scientist Card Special Style */
    .path-card.scientist {
        background: rgba(30, 41, 59, 0.4); /* Unified look */
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .path-card.scientist:hover {
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.2);
    }

    /* Typography */
    h1 {
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    
    h2 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    
    p {
        color: #94a3b8 !important;
        line-height: 1.6;
    }

    /* Metric Boxes */
    .metric-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(71, 85, 105, 0.5);
        border-radius: 16px;
        padding: 1.5rem;
        transition: transform 0.2s;
    }
    
    .metric-box:hover {
        transform: translateY(-3px);
        border-color: #6366f1;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.3);
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
    }

    div.stButton > button:active {
        transform: translateY(0);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Inputs */
    .stTextInput > div > div > input {
        background-color: rgba(30, 41, 59, 0.5);
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    .stSelectbox > div > div > div {
        background-color: rgba(30, 41, 59, 0.5) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }

</style>
""", unsafe_allow_html=True)


def get_lang():
    """Get current language"""
    return st.session_state.get('lang', 'en')


def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'path': None,
        'df': None,
        'domain': 'custom',
        'tool': 'python',
        'output_format': 'jupyter',
        'analysis_complete': False,
        'analysis_report': None,
        'insights': None,
        'ml_results': None,
        'lang': 'en'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value





def show_path_selection():
    """Show path selection screen"""
    lang = get_lang()
    
    st.markdown(f"<h1 style='text-align: center;'>{t('app_title', lang)}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #aaa;'>{t('app_subtitle', lang)}</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="path-card">
            <h2>{t('data_analyst', lang)}</h2>
            <p>{t('data_analyst_desc', lang)}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button(t('start_analyst', lang), key="analyst_btn", use_container_width=True):
            st.session_state.path = 'analyst'
            st.rerun()
    
    with col2:
        st.markdown(f"""
        <div class="path-card scientist">
            <h2>{t('data_scientist', lang)}</h2>
            <p>{t('data_scientist_desc', lang)}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button(t('start_scientist', lang), key="scientist_btn", use_container_width=True):
            st.session_state.path = 'scientist'
            st.rerun()
    
    # Sample data section
    st.markdown("---")
    st.subheader(t('sample_data_title', lang))
    
    sample_cols = st.columns(3)
    samples = [
        ('hr', t('sample_hr', lang)), 
        ('finance', t('sample_finance', lang)), 
        ('healthcare', t('sample_healthcare', lang)),
        ('retail', t('sample_retail', lang)), 
        ('marketing', t('sample_marketing', lang)), 
        ('education', t('sample_education', lang))
    ]
    
    for i, (domain, label) in enumerate(samples):
        with sample_cols[i % 3]:
            if st.button(label, key=f"sample_{domain}", use_container_width=True):
                loader = DataLoader()
                st.session_state.df = loader._generate_sample_data(domain)
                st.session_state.domain = domain
                st.success(t('loaded_sample', lang).format(domain))


def show_sidebar():
    """Show configuration sidebar"""
    lang = get_lang()
    
    with st.sidebar:
        # --- Language Switcher ---
        # st.markdown("### 🌐 Language")
        lang_options = {"🇺🇸 English": "en", "🇸🇦 العربية": "ar"}
        
        # Determine current index
        current_lang = st.session_state.get('lang', 'en')
        index = 0 if current_lang == 'en' else 1
        
        sel_lang = st.radio(
            "Language / اللغة",
            options=list(lang_options.keys()),
            index=index,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        new_lang = lang_options[sel_lang]
        if new_lang != current_lang:
            st.session_state.lang = new_lang
            st.rerun()
        
        # Back to home button
        if st.button("🏠 " + ("Back to Home" if lang == 'en' else "العودة للرئيسية"), key="back_to_home_analyst", use_container_width=True):
            st.session_state.app_mode = None
            st.session_state.path = None
            st.rerun()
            
        st.markdown("---")

        st.markdown(f"## {t('configuration', lang)}")
        
        # Path switcher
        current_path = t('data_analyst', lang) if st.session_state.path == 'analyst' else t('data_scientist', lang)
        st.markdown(f"**{t('current_path', lang)}:** {current_path}")
        if st.button(t('switch_path', lang)):
            st.session_state.path = None
            st.session_state.app_mode = None  # Also reset app_mode for main app compatibility
            st.session_state.analysis_complete = False
            st.rerun()
        
        st.markdown("---")
        
        # Domain selection
        st.markdown(f"### {t('domain', lang)}")
        domain_options = {k: f"{v['icon']} {v['name'] if lang == 'en' else v.get('name_ar', v['name'])}" 
                        for k, v in DOMAINS.items()}
        st.session_state.domain = st.selectbox(
            t('select_domain', lang),
            options=list(domain_options.keys()),
            format_func=lambda x: domain_options[x],
            index=list(domain_options.keys()).index(st.session_state.domain)
        )
        
        # Tool selection (for analyst path)
        if st.session_state.path == 'analyst':
            st.markdown(f"### {t('processing_tool', lang)}")
            tool_options = {k: f"{v['icon']} {v['name'] if lang == 'en' else v.get('name_ar', v['name'])}" 
                          for k, v in TOOLS.items()}
            st.session_state.tool = st.selectbox(
                t('select_tool', lang),
                options=list(tool_options.keys()),
                format_func=lambda x: tool_options[x]
            )
        
        # Output format
        st.markdown(f"### {t('output_format', lang)}")
        output_options = {k: f"{v['icon']} {v['name'] if lang == 'en' else v.get('name_ar', v['name'])}" 
                        for k, v in OUTPUT_FORMATS.items()}
        st.session_state.output_format = st.selectbox(
            t('select_format', lang),
            options=list(output_options.keys()),
            format_func=lambda x: output_options[x]
        )
        
        st.markdown("---")
        
        # Data info
        if st.session_state.df is not None:
            st.markdown(f"### {t('data_info', lang)}")
            df = st.session_state.df
            st.markdown(f"**{t('rows', lang)}:** {len(df):,}")
            st.markdown(f"**{t('columns', lang)}:** {len(df.columns)}")
            st.markdown(f"**{t('memory', lang)}:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def show_data_upload():
    """Show data upload section with SQL support"""
    lang = get_lang()
    
    st.markdown(f"## {t('upload_title', lang)}")
    
    # Data Source Selection
    tab_upload, tab_sql = st.tabs([
        t('upload_files', lang) if lang else "Upload Files", 
        "🗄️ " + ("قاعدة بيانات SQL" if lang == 'ar' else "SQL Database")
    ])
    
    with tab_upload:
        uploaded_file = st.file_uploader(
            t('upload_hint', lang),
            type=['csv', 'xlsx', 'xls', 'json', 'parquet']
        )
        
        if uploaded_file:
            try:
                # Save file to temp for DuckDB processing
                import os
                temp_dir = PROJECT_ROOT / "temp_data"
                temp_dir.mkdir(exist_ok=True)
                
                # Create a uniquely named temp file to avoid conflicts
                safe_name = f"{datetime.now().strftime('%H%M%S')}_{uploaded_file.name}"
                temp_path = temp_dir / safe_name
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Register with DuckDB
                if 'duck_engine' not in st.session_state:
                    from core.duck_engine import DuckDBEngine
                    st.session_state.duck_engine = DuckDBEngine()
                
                # Check file size/lines quickly without loading everything
                # Register the file content as a view
                success, error_msg = st.session_state.duck_engine.register_file(str(temp_path), table_name='current_data')
                
                if success:
                    # Get robust stats and SAMPLE
                    st.session_state.file_path = str(temp_path)
                    
                    # Log info
                    stats = st.session_state.duck_engine.get_summary_stats('current_data')
                    total_rows = stats.get('total_rows', 0)
                    
                    # Load Safe Sample (Max 100k for UI)
                    if total_rows > 100000:
                        st.info(f"🚀 Large dataset detected ({total_rows:,} rows). Optimized mode enabled using DuckDB.")
                        st.session_state.df = st.session_state.duck_engine.get_sample('current_data', limit=100000)
                    else:
                        # Small enough to load fully
                        st.session_state.df = st.session_state.duck_engine.query("SELECT * FROM current_data")
                    
                    st.success(t('loaded_success', lang).format(len(st.session_state.df), len(st.session_state.df.columns)))
                    
                    # Preview
                    with st.expander(t('data_preview', lang), expanded=True):
                        st.dataframe(st.session_state.df.head(10), use_container_width=True)
                    
                    # Auto-transition to analysis tabs
                    st.rerun()
                else:
                    st.error(f"Failed to register data with analytics engine: {error_msg}")
            
            except Exception as e:
                st.error(f"{t('error_loading', lang)}: {e}")

    with tab_sql:
        col_sql1, col_sql2 = st.columns(2)
        with col_sql1:
            db_type = st.selectbox(
                "نوع قاعدة البيانات" if lang == 'ar' else "Database Type",
                ["SQLite", "PostgreSQL", "MySQL", "SQL Server"]
            )
        with col_sql2:
            conn_str = st.text_input(
                "سلسلة الاتصال (Connection String)" if lang == 'ar' else "Connection String",
                placeholder="sqlite:///database.db",
                type="password"
            )
            
        sql_query = st.text_area(
            "استعلام SQL" if lang == 'ar' else "SQL Query",
            value="SELECT * FROM table_name LIMIT 1000",
            height=100
        )
        
        if st.button("🔌 " + ("اتصال وتحميل" if lang == 'ar' else "Connect & Load"), key="btn_sql_main"):
            if conn_str and sql_query:
                try:
                    loader = DataLoader()
                    with st.spinner("Connecting..."):
                        df = loader.load_sql(conn_str, sql_query)
                        if df is not None:
                            st.session_state.df = df
                            st.success(t('loaded_success', lang).format(len(df), len(df.columns)))
                            st.rerun()
                except Exception as e:
                    st.error(str(e))
    
    # --- Sample Data Section (Quick Start) ---
    st.markdown("---")
    col_sample_title, _ = st.columns([0.4, 0.6])
    with col_sample_title:
        st.caption(t('or_use_sample', lang))
    
    sample_cols = st.columns(6)
    samples = [
        ('hr', t('sample_hr', lang), '👥'), 
        ('finance', t('sample_finance', lang), '💰'), 
        ('healthcare', t('sample_healthcare', lang), '🏥'),
        ('retail', t('sample_retail', lang), '🛒'), 
        ('marketing', t('sample_marketing', lang), '📈'), 
        ('education', t('sample_education', lang), '🎓')
    ]
    
    for i, (domain, label, emoji) in enumerate(samples):
        with sample_cols[i]:
            if st.button(f"{emoji}\n{label}", key=f"upload_sample_{domain}", use_container_width=True):
                loader = DataLoader() # Ensure loader is available
                with st.spinner(t('loading', lang)):
                    st.session_state.df = loader._generate_sample_data(domain)
                    st.session_state.domain = domain
                    st.rerun()




# ==========================================
# Missing Functions (Added to prevent errors)
# ==========================================
def show_advanced_viz_tab(df):
    """Placeholder for Advanced Visualization"""
    st.info("🚧 Advanced Visualization: Coming Soon")
    st.caption("This module is under development.")

def show_maintenance_tab():
    """Display the AI Sentinel UI."""
    lang = get_lang()
    sentinel = get_sentinel()
    sentinel.show_maintenance_ui(lang)


def show_knowledge_hub(lang):
    """Placeholder for Knowledge Hub"""
    st.info("🚧 Knowledge Hub: Coming Soon" if lang == 'en' else "🚧 مركز المعرفة: قريباً")
    st.caption("This module is under development.")

def show_data_analyst_path():
    """Show Data Analyst path interface"""
    lang = get_lang()
    
    st.markdown(f"# {t('data_analyst', lang)}")
    
    show_sidebar()
    
    if st.session_state.df is None:
        show_data_upload()
        return
    
    df = st.session_state.df
    
    # Analysis tabs - Enhanced
    tabs = [
        t('tab_analysis', lang), 
        "🔍 Auto EDA" if lang == 'en' else "🔍 تحليل تلقائي",
        "📝 NLP" if lang == 'en' else "📝 تحليل نصوص",
        "🎨 Advanced Viz" if lang == 'en' else "🎨 رسوم متقدمة",
        t('tab_cleaning', lang), 
        t('tab_insights', lang), 
        t('tab_dashboard', lang), 
        t('tab_export', lang),
        "📚 Knowledge Hub" if lang == 'en' else "📚 مركز المعرفة"
    ]
    
    # Add Maintenance tab ONLY for Admins
    is_admin = False
    if 'user_data' in st.session_state:
        # Check role in user_data OR explicit username
        is_admin = st.session_state.user_data.get('role') == 'admin' or st.session_state.get('username') == 'admin' or st.session_state.get('username') == 'Admin'
        
    if HAS_NEW_FEATURES and is_admin:
        tabs.append("🛡️ Maintenance" if lang == 'en' else "🛡️ الصيانة الذكية")
        
    tab_list = st.tabs(tabs)
    
    with tab_list[0]:
        show_analysis_tab(df)
    
    with tab_list[1]:
        show_auto_eda_tab(df)
        
    with tab_list[2]:
        show_nlp_tab(df)
        
    with tab_list[3]:
        show_advanced_viz_tab(df)
        
    with tab_list[4]:
        show_cleaning_tab(df)
    
    with tab_list[5]:
        show_insights_tab(df)
    
    with tab_list[6]:
        show_dashboard_tab(df)
    
    with tab_list[7]:
        show_export_tab(df)

    with tab_list[8]:
        show_knowledge_hub(lang) # New function
        
    if HAS_NEW_FEATURES and is_admin:
        with tab_list[9]:
            show_maintenance_tab()


def show_analysis_tab(df):
    """Show analysis results"""
    lang = get_lang()
    
    st.markdown(f"### {t('data_quality', lang)}")
    
    if st.button(t('run_analysis', lang), key="run_analysis"):
        with st.spinner(t('analyzing', lang)):
            analyzer = DataAnalyzer()
            # Pass file path if available for DuckDB optimization
            file_path = st.session_state.get('file_path')
            report = analyzer.analyze(df, file_path=file_path)
            st.session_state.analysis_report = report
            st.session_state.analysis_complete = True
    
    if st.session_state.analysis_complete and st.session_state.analysis_report:
        report = st.session_state.analysis_report
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"📊 {t('rows', lang)}", f"{report.total_rows:,}")
        with col2:
            st.metric(f"📋 {t('columns', lang)}", report.total_columns)
        with col3:
            st.metric(f"❓ {t('missing', lang)}", f"{report.missing_percent}%")
        with col4:
            st.metric(f"🔄 {t('duplicates', lang)}", f"{report.duplicate_rows:,}")
        
        # Column analysis
        st.markdown(f"#### {t('column_stats', lang)}")
        col_data = []
        for col, analysis in report.columns_analysis.items():
            col_data.append({
                t('columns', lang): col,
                t('type', lang): analysis.dtype,
                f"{t('missing', lang)} %": f"{analysis.missing_percent}%",
                t('unique', lang): analysis.unique_count,
                t('mean', lang): analysis.mean if analysis.mean else '-'
            })
        st.dataframe(pd.DataFrame(col_data), use_container_width=True)
        
        # Correlation heatmap
        if len(report.numeric_columns) > 1:
            st.markdown(f"#### {t('correlation_matrix', lang)}")
            numeric_df = df[report.numeric_columns]
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
        
        # Issues
        if report.data_issues:
            st.markdown(f"#### {t('data_issues', lang)}")
            for issue in report.data_issues:
                severity_color = {'critical': '🔴', 'warning': '🟡', 'info': 'ℹ️'}
                st.markdown(f"{severity_color.get(issue['severity'], 'ℹ️')} **{issue['message']}**")
                if issue.get('recommendation'):
                    st.caption(f"💡 {issue['recommendation']}")


def show_cleaning_tab(df):
    """Show data cleaning interface"""
    lang = get_lang()
    
    st.markdown(f"### {t('data_cleaning', lang)}")
    
    # Tool selection
    tool_map = {'python': CleaningTool.PYTHON, 'excel': CleaningTool.EXCEL, 'powerbi': CleaningTool.POWERBI}
    current_tool = tool_map.get(st.session_state.tool, CleaningTool.PYTHON)
    
    tool_name = TOOLS[st.session_state.tool]['name'] if lang == 'en' else TOOLS[st.session_state.tool].get('name_ar', TOOLS[st.session_state.tool]['name'])
    st.info(f"{t('using_tool', lang)}: **{tool_name}**")
    
    # Operation selection
    operations = st.multiselect(
        t('select_operations', lang),
        ['missing_values', 'duplicates', 'outliers', 'normalize', 'standardize', 'encode'],
        default=['missing_values', 'duplicates']
    )
    
    if st.button(t('clean_data', lang)):
        with st.spinner(t('analyzing', lang)):
            cleaner = DataCleaner(tool=current_tool)
            cleaned_df = cleaner.clean(df, operations)
            
            st.success(t('cleaning_complete', lang))
            
            # Show summary
            summary = cleaner.get_summary()
            col1, col2 = st.columns(2)
            with col1:
                st.metric(t('rows', lang), f"{summary['rows_before']} → {summary['rows_after']}")
            with col2:
                st.metric(t('columns', lang), f"{summary['columns_before']} → {summary['columns_after']}")
            
            # Show generated code
            st.markdown(f"#### {t('generated_code', lang)}")
            st.code(cleaner.get_code(), language='python' if current_tool == CleaningTool.PYTHON else 'sql')
            
            # Download cleaned data
            if current_tool == CleaningTool.PYTHON:
                csv = cleaned_df.to_csv(index=False)
                st.download_button(t('download_cleaned', lang), csv, "cleaned_data.csv", "text/csv")


def show_insights_tab(df):
    """Show domain-specific insights"""
    lang = get_lang()
    
    st.markdown(f"### {t('domain_insights', lang)}")
    
    domain = st.session_state.domain
    domain_name = DOMAINS[domain]['name'] if lang == 'en' else DOMAINS[domain].get('name_ar', DOMAINS[domain]['name'])
    st.info(f"{t('domain', lang)}: **{domain_name}** {DOMAINS[domain]['icon']}")
    
    if st.button(t('generate_insights', lang)):
        with st.spinner(t('generating_insights', lang)):
            analyzer = DataAnalyzer()
            # Pass file path if available
            file_path = st.session_state.get('file_path')
            report = analyzer.analyze(df, file_path=file_path)
            
            insights_gen = InsightsGenerator(domain=domain, lang=lang)
            insights = insights_gen.generate_insights(df, analyzer.get_summary(), lang=lang)
            st.session_state.insights = insights
    
    if st.session_state.insights:
        for insight in st.session_state.insights[:10]:
            icon = "✅" if insight.severity == 'success' else "⚠️" if insight.severity == 'warning' else "🔴" if insight.severity == 'critical' else "ℹ️"
            
            with st.container():
                st.markdown(f"**{icon} {insight.title}**")
                st.caption(insight.description)
                if insight.recommendation:
                    st.success(f"💡 {insight.recommendation}")
                st.markdown("---")
        
        # STORYTELLER ENGINE INTEGRATION
        if HAS_NEW_FEATURES:
            st.markdown("### 📜 " + ("Intelligent Data Story" if lang == 'en' else "قصة البيانات الذكية"))
            
            with st.expander("✨ " + ("Generate Executive Narrative" if lang == 'en' else "توليد قصة تنفيذية"), expanded=True):
                 story_context = st.text_area(
                    "Context (Optional)" if lang == 'en' else "سياق إضافي (اختياري)",
                    placeholder="E.g., Prepare this for the CEO.",
                    height=70
                 )
                 
                 if st.button("🖋️ " + ("Write Data Story" if lang == 'en' else "كتابة قصة البيانات"), type="primary"):
                     from paths.data_analyst.storyteller import StorytellerEngine
                     
                     with st.spinner("🤖 Weaving the story..." if lang == 'en' else "جاري صياغة القصة..."):
                        # Get data summary string
                        data_summary = f"Rows: {len(df)}, Cols: {len(df.columns)}. " + \
                                     str(df.describe().to_string())
                                     
                        ai_assistant = AIAssistant(lang=lang)
                        storyteller = StorytellerEngine(ai_assistant)
                        
                        story = storyteller.generate_narrative(
                            data_summary, 
                            st.session_state.insights, 
                            story_context, 
                            lang=lang
                        )
                        
                        st.session_state['data_story'] = story
            
            if 'data_story' in st.session_state:
                st.markdown("#### 📖 " + ("Your Data Story" if lang == 'en' else "قصة بياناتك"))
                st.markdown(st.session_state['data_story'])
                
                # Export Button
                st.download_button(
                    "📥 " + ("Download Story" if lang == 'en' else "تحميل القصة"),
                    st.session_state['data_story'],
                    "data_story.md",
                    "text/markdown"
                )


def show_dashboard_tab(df):
    """Show interactive dashboard with charts and filters"""
    lang = get_lang()
    
    st.markdown(f"### {t('dashboard_gen', lang)}")
    
    # Dashboard mode selection
    dashboard_mode = st.radio(
        "📊 " + ("Dashboard Mode" if lang == 'en' else "وضع لوحة المعلومات"),
        ["interactive", "export"],
        format_func=lambda x: ("🎯 Interactive Dashboard" if x == "interactive" else "📥 Export Dashboard") if lang == 'en' else ("🎯 لوحة تفاعلية" if x == "interactive" else "📥 تصدير لوحة المعلومات"),
        horizontal=True
    )
    
    if dashboard_mode == "interactive":
        show_interactive_dashboard(df, lang)
    else:
        show_export_dashboard(df, lang)


def show_interactive_dashboard(df, lang):
    """Interactive dashboard with live charts and filters"""
    
    st.markdown("---")
    
    # === SIDEBAR FILTERS ===
    st.sidebar.markdown(f"### 🎛️ " + ("Filters" if lang == 'en' else "الفلاتر"))
    
    # Get column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Categorical filters
    filters = {}
    for col in categorical_cols[:4]:  # Limit to 4 filters
        unique_vals = df[col].dropna().unique().tolist()
        if len(unique_vals) <= 20:  # Only show filter if manageable number of options
            selected = st.sidebar.multiselect(
                f"📌 {col}",
                options=unique_vals,
                default=unique_vals[:5] if len(unique_vals) > 5 else unique_vals
            )
            if selected:
                filters[col] = selected
    
    # Apply filters
    filtered_df = df.copy()
    for col, values in filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(values)]
    
    # === KPI CARDS ===
    st.markdown("#### 📈 " + ("Key Metrics" if lang == 'en' else "المقاييس الرئيسية"))
    
    kpi_cols = st.columns(4)
    
    with kpi_cols[0]:
        st.metric(
            "📊 " + ("Records" if lang == 'en' else "السجلات"),
            f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
        )
    
    with kpi_cols[1]:
        if numeric_cols:
            total = filtered_df[numeric_cols[0]].sum()
            st.metric(
                f"💰 {numeric_cols[0][:15]}",
                f"{total:,.0f}"
            )
    
    with kpi_cols[2]:
        if len(numeric_cols) > 1:
            avg = filtered_df[numeric_cols[1]].mean()
            st.metric(
                f"📈 " + ("Avg" if lang == 'en' else "متوسط") + f" {numeric_cols[1][:10]}",
                f"{avg:,.2f}"
            )
    
    with kpi_cols[3]:
        if categorical_cols:
            unique = filtered_df[categorical_cols[0]].nunique()
            st.metric(
                f"🏷️ {categorical_cols[0][:15]}",
                f"{unique:,}"
            )
    
    st.markdown("---")
    
    # === CHARTS ===
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### 📊 " + ("Distribution Chart" if lang == 'en' else "مخطط التوزيع"))
        
        if categorical_cols and numeric_cols:
            chart_cat = st.selectbox(
                "Category" if lang == 'en' else "الفئة",
                categorical_cols,
                key="chart_cat_1"
            )
            chart_num = st.selectbox(
                "Value" if lang == 'en' else "القيمة",
                numeric_cols,
                key="chart_num_1"
            )
            
            # Aggregate data
            agg_data = filtered_df.groupby(chart_cat)[chart_num].sum().reset_index()
            agg_data = agg_data.sort_values(chart_num, ascending=False).head(10)
            
            fig = px.bar(
                agg_data, 
                x=chart_cat, 
                y=chart_num,
                color=chart_num,
                color_continuous_scale='Viridis',
                title=f"{chart_num} by {chart_cat}"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.markdown("#### 🥧 " + ("Pie Chart" if lang == 'en' else "مخطط دائري"))
        
        if categorical_cols:
            pie_col = st.selectbox(
                "Category" if lang == 'en' else "الفئة",
                categorical_cols,
                key="pie_cat"
            )
            
            pie_data = filtered_df[pie_col].value_counts().head(8)
            
            fig = px.pie(
                values=pie_data.values,
                names=pie_data.index,
                title=f"{pie_col} " + ("Distribution" if lang == 'en' else "التوزيع"),
                hole=0.4
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # === LINE CHART / TREND ===
    st.markdown("#### 📈 " + ("Trend Analysis" if lang == 'en' else "تحليل الاتجاه"))
    
    if len(numeric_cols) >= 2:
        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            x_axis = st.selectbox("X-Axis" if lang == 'en' else "المحور الأفقي", numeric_cols, key="x_trend")
        with trend_col2:
            y_axis = st.selectbox("Y-Axis" if lang == 'en' else "المحور الرأسي", 
                                 [c for c in numeric_cols if c != x_axis], key="y_trend")
        
        color_by = st.selectbox(
            "Color by" if lang == 'en' else "تلوين حسب",
            ["None"] + categorical_cols,
            key="color_trend"
        )
        
        fig = px.scatter(
            filtered_df.head(1000),  # Limit points for performance
            x=x_axis,
            y=y_axis,
            color=color_by if color_by != "None" else None,
            title=f"{y_axis} vs {x_axis}",
            trendline="ols" if len(filtered_df) > 10 else None
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # === DATA TABLE ===
    st.markdown("#### 📋 " + ("Filtered Data" if lang == 'en' else "البيانات المفلترة"))
    
    with st.expander("🔍 " + ("View Data" if lang == 'en' else "عرض البيانات"), expanded=False):
        st.dataframe(filtered_df.head(100), use_container_width=True)
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 " + ("Download Filtered Data" if lang == 'en' else "تحميل البيانات المفلترة"),
            csv,
            "filtered_data.csv",
            "text/csv"
        )


def show_export_dashboard(df, lang):
    """Export dashboard to files"""
    
    format_choice = st.radio(
        t('select_format_dash', lang),
        ['jupyter', 'excel', 'powerbi'],
        format_func=lambda x: OUTPUT_FORMATS[x]['name'] if lang == 'en' else OUTPUT_FORMATS[x].get('name_ar', OUTPUT_FORMATS[x]['name'])
    )
    
    if st.button(t('generate_dashboard', lang)):
        with st.spinner(t('generating_dashboard', lang)):
            output_dir = BASE_DIR / "outputs" / "dashboards"
            generator = DashboardGenerator(output_dir=output_dir)
            
            analyzer = DataAnalyzer()
            analyzer.analyze(df)
            analysis = analyzer.get_summary()
            
            if format_choice == 'jupyter':
                path = generator.generate_jupyter_notebook(df, analysis, st.session_state.domain)
            elif format_choice == 'excel':
                path = generator.generate_excel_dashboard_template(df, analysis, st.session_state.domain)
            else:
                path = generator.generate_powerbi_template(df, analysis, st.session_state.domain)
            
            st.success(f"{t('dashboard_generated', lang)}: {path}")
            
            # Provide download for Jupyter
            if format_choice == 'jupyter' and Path(path).exists():
                with open(path, 'r') as f:
                    st.download_button(t('download_notebook', lang), f.read(), Path(path).name, "application/json")


def show_export_tab(df):
    """Show export options with full insights"""
    lang = get_lang()
    
    st.markdown(f"### {t('export_reports', lang)}")
    
    # Generate analysis first
    analyzer = DataAnalyzer()
    report = analyzer.analyze(df)
    analysis = analyzer.get_summary()
    
    insights_gen = InsightsGenerator(domain=st.session_state.domain, lang=lang)
    insights = insights_gen.generate_insights(df, analysis, lang=lang)
    
    # Display report preview
    st.markdown("#### 📋 " + ("Report Preview" if lang == 'en' else "معاينة التقرير"))
    
    with st.expander("📊 " + ("Statistics & Insights" if lang == 'en' else "الإحصائيات والرؤى"), expanded=True):
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 " + ("Rows" if lang == 'en' else "صفوف"), f"{report.total_rows:,}")
        with col2:
            st.metric("📋 " + ("Columns" if lang == 'en' else "أعمدة"), report.total_columns)
        with col3:
            st.metric("❓ " + ("Missing" if lang == 'en' else "مفقود"), f"{report.missing_percent}%")
        with col4:
            st.metric("🔄 " + ("Duplicates" if lang == 'en' else "مكرر"), f"{report.duplicate_rows:,}")
        
        st.markdown("---")
        
        # Insights summary
        st.markdown("##### 💡 " + ("Key Insights" if lang == 'en' else "الرؤى الرئيسية"))
        for insight in insights[:5]:
            icon = "✅" if insight.severity == 'success' else "⚠️" if insight.severity == 'warning' else "🔴" if insight.severity == 'critical' else "ℹ️"
            st.markdown(f"**{icon} {insight.title}**")
            st.caption(insight.description)
            if insight.recommendation:
                st.info(f"💡 {insight.recommendation}")
    
    st.markdown("---")
    
    # Export options
    st.markdown("#### 📥 " + ("Export Options" if lang == 'en' else "خيارات التصدير"))
    
    export_cols = st.columns(3)
    
    # 1. Full Report (Markdown)
    with export_cols[0]:
        st.markdown("##### 📝 " + ("Full Report" if lang == 'en' else "تقرير كامل"))
        
        # Generate markdown report
        report_md = generate_full_report_md(df, report, insights, lang)
        
        st.download_button(
            "📥 " + ("Download Report (MD)" if lang == 'en' else "تحميل التقرير"),
            report_md,
            "analysis_report.md",
            "text/markdown",
            use_container_width=True
        )
    
    # 2. Insights Only
    with export_cols[1]:
        st.markdown("##### 💡 " + ("Insights Only" if lang == 'en' else "الرؤى فقط"))
        
        insights_text = generate_insights_text(insights, lang)
        
        st.download_button(
            "📥 " + ("Download Insights" if lang == 'en' else "تحميل الرؤى"),
            insights_text,
            "insights.txt",
            "text/plain",
            use_container_width=True
        )
    
    # 3. Data + Summary
    with export_cols[2]:
        st.markdown("##### 📊 " + ("Data + Summary" if lang == 'en' else "البيانات + ملخص"))
        
        # Create Excel with multiple sheets
        csv_with_summary = generate_csv_with_summary(df, analysis, insights, lang)
        
        st.download_button(
            "📥 " + ("Download CSV" if lang == 'en' else "تحميل CSV"),
            csv_with_summary,
            "data_with_insights.csv",
            "text/csv",
            use_container_width=True
        )


def generate_full_report_md(df, report, insights, lang):
    """Generate full markdown report"""
    title = "تقرير تحليل البيانات" if lang == 'ar' else "Data Analysis Report"
    summary_title = "الملخص التنفيذي" if lang == 'ar' else "Executive Summary"
    stats_title = "الإحصائيات" if lang == 'ar' else "Statistics"
    insights_title = "الرؤى والتوصيات" if lang == 'ar' else "Insights & Recommendations"
    
    report_content = f"""# {title}

## {summary_title}

| {"المقياس" if lang == 'ar' else "Metric"} | {"القيمة" if lang == 'ar' else "Value"} |
|--------|-------|
| {"إجمالي الصفوف" if lang == 'ar' else "Total Rows"} | {report.total_rows:,} |
| {"إجمالي الأعمدة" if lang == 'ar' else "Total Columns"} | {report.total_columns} |
| {"نسبة المفقود" if lang == 'ar' else "Missing %"} | {report.missing_percent}% |
| {"الصفوف المكررة" if lang == 'ar' else "Duplicate Rows"} | {report.duplicate_rows:,} |
| {"الأعمدة الرقمية" if lang == 'ar' else "Numeric Columns"} | {len(report.numeric_columns)} |
| {"الأعمدة النصية" if lang == 'ar' else "Categorical Columns"} | {len(report.categorical_columns)} |

---

## {insights_title}

"""
    
    for i, insight in enumerate(insights, 1):
        icon = "✅" if insight.severity == 'success' else "⚠️" if insight.severity == 'warning' else "🔴" if insight.severity == 'critical' else "ℹ️"
        report_content += f"""
### {i}. {icon} {insight.title}

{insight.description}

"""
        if insight.recommendation:
            rec_label = "التوصية" if lang == 'ar' else "Recommendation"
            report_content += f"**💡 {rec_label}:** {insight.recommendation}\n\n"
    
    # Column statistics
    report_content += f"""
---

## {stats_title}

| {"العمود" if lang == 'ar' else "Column"} | {"النوع" if lang == 'ar' else "Type"} | {"مفقود %" if lang == 'ar' else "Missing %"} | {"فريد" if lang == 'ar' else "Unique"} |
|--------|------|---------|--------|
"""
    
    for col, analysis in report.columns_analysis.items():
        report_content += f"| {col} | {analysis.dtype} | {analysis.missing_percent}% | {analysis.unique_count} |\n"
    
    return report_content


def generate_insights_text(insights, lang):
    """Generate insights as plain text"""
    title = "الرؤى والتوصيات" if lang == 'ar' else "Insights & Recommendations"
    
    text = f"{'='*50}\n{title}\n{'='*50}\n\n"
    
    for i, insight in enumerate(insights, 1):
        icon = "[SUCCESS]" if insight.severity == 'success' else "[WARNING]" if insight.severity == 'warning' else "[CRITICAL]" if insight.severity == 'critical' else "[INFO]"
        text += f"{i}. {icon} {insight.title}\n"
        text += f"   {insight.description}\n"
        if insight.recommendation:
            rec = "التوصية" if lang == 'ar' else "Recommendation"
            text += f"   >> {rec}: {insight.recommendation}\n"
        text += "\n"
    
    return text


def show_auto_eda_tab(df):
    """Show Auto EDA interface"""
    lang = get_lang()
    
    st.markdown("### 🔍 " + ("Auto EDA" if lang == 'en' else "التحليل الاستكشافي الآلي"))
    
    if HAS_NEW_FEATURES:
        if st.button("🚀 " + ("Run Auto EDA" if lang == 'en' else "تشغيل التحليل الآلي"), type="primary"):
            with st.spinner(t('analyzing', lang)):
                eda = AutoEDA(lang=lang)
                report = eda.analyze(df)
                st.session_state.eda_report = report
        
        if hasattr(st.session_state, 'eda_report'):
            report = st.session_state.eda_report
            
            # 1. Summary Metrics
            st.markdown("#### 📊 " + ("Dataset Overview" if lang == 'en' else "نظرة عامة"))
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", f"{report.n_rows:,}")
            col2.metric("Columns", report.n_cols)
            col3.metric("Memory", f"{report.memory_mb:.1f} MB")
            col4.metric("Duplicates", f"{report.duplicate_percent}%")
            
            # 2. Recommendations
            if report.recommendations:
                st.info("💡 " + "\n\n".join(report.recommendations))
            
            # 3. Numeric Stats
            if report.numeric_stats:
                st.markdown("#### 🔢 " + ("Numeric Statistics" if lang == 'en' else "الإحصائيات الرقمية"))
                stats_df = pd.DataFrame(report.numeric_stats).T
                st.dataframe(stats_df, use_container_width=True)
            
            # 4. Outliers
            if report.outlier_summary:
                st.markdown("#### 📉 " + ("Outliers" if lang == 'en' else "القيم الشاذة"))
                st.bar_chart(report.outlier_summary)
            
            # 5. Correlations
            if report.high_correlations:
                st.markdown("#### 🔗 " + ("High Correlations" if lang == 'en' else "ارتباطات قوية"))
                corr_df = pd.DataFrame(report.high_correlations)
                st.dataframe(corr_df[['col1', 'col2', 'correlation']], use_container_width=True)
    else:
        st.warning("Auto EDA module not loaded")


def show_nlp_tab(df):
    """Show NLP Analysis interface"""
    lang = get_lang()
    
    st.markdown("### 📝 " + ("NLP Analysis" if lang == 'en' else "تحليل النصوص"))
    
    if HAS_NEW_FEATURES:
        target_cols = [c for c in df.columns if df[c].dtype == 'object']
        
        if not target_cols:
            st.info("No text columns detected")
            return
            
        selected_cols = st.multiselect(
            "Select Text Columns" if lang == 'en' else "اختر أعمدة النصوص",
            target_cols
        )
        
        if selected_cols and st.button("🚀 " + ("Analyze Text" if lang == 'en' else "تحليل النص")):
            with st.spinner(t('analyzing', lang)):
                analyzer = TextAnalyzer(lang=lang)
                results = analyzer.analyze(df, selected_cols)
                st.session_state.nlp_results = results
        
        if hasattr(st.session_state, 'nlp_results'):
            res = st.session_state.nlp_results
            
            # Stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Texts", f"{res.total_texts:,}")
            col2.metric("Avg Length", f"{res.avg_length:.1f}")
            col3.metric("Language", res.language_detected.upper())
            
            # Sentiment
            st.markdown("#### ❤️ " + ("Sentiment" if lang == 'en' else "المشاعر"))
            sent = res.sentiment_summary
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Positive", f"{sent.get('positive_pct', 0)}%")
            c2.metric("Negative", f"{sent.get('negative_pct', 0)}%")
            c3.metric("Neutral", f"{sent.get('neutral', 0)}")
            
            # Topics
            if res.topics:
                st.markdown("#### 🏷️ " + ("Top Topics" if lang == 'en' else "أهم المواضيع"))
                st.write(", ".join(res.topics))
                
            # Word Cloud data
            if res.word_frequency:
                st.markdown("#### ☁️ " + ("Top Words" if lang == 'en' else "أكثر الكلمات تكراراً"))
                wf_df = pd.DataFrame(list(res.word_frequency.items()), columns=['Word', 'Count'])
                st.bar_chart(wf_df.set_index('Word').head(20))
    else:
        st.warning("NLP module not loaded")


def generate_csv_with_summary(df, analysis, insights, lang):
    """Generate CSV with summary header"""
    import io
    
    # Create summary section
    summary_lines = [
        f"# {'تقرير تحليل البيانات' if lang == 'ar' else 'Data Analysis Report'}",
        f"# {'الصفوف' if lang == 'ar' else 'Rows'}: {analysis['rows']:,}",
        f"# {'الأعمدة' if lang == 'ar' else 'Columns'}: {analysis['columns']}",
        f"# {'نسبة المفقود' if lang == 'ar' else 'Missing %'}: {analysis['missing_percent']}%",
        "#",
        f"# {'الرؤى الرئيسية' if lang == 'ar' else 'Key Insights'}:",
    ]
    
    for insight in insights[:3]:
        summary_lines.append(f"# - {insight.title}")
    
    summary_lines.append("#")
    summary_lines.append(f"# {'البيانات' if lang == 'ar' else 'Data'}:")
    
    # Combine summary with data
    csv_buffer = io.StringIO()
    csv_buffer.write("\n".join(summary_lines) + "\n")
    df.to_csv(csv_buffer, index=False)
    
    return csv_buffer.getvalue()


def show_data_scientist_path():
    """Show Data Scientist path interface"""
    lang = get_lang()
    
    st.markdown(f"# {t('data_scientist', lang)}")
    
    show_sidebar()
    
    if st.session_state.df is None:
        show_data_upload()
        return
    
    df = st.session_state.df
    
    # ML tabs - Enhanced
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        t('tab_target', lang), 
        t('tab_features', lang), 
        "🚀 AutoML" if lang == 'en' else "🚀 التعلم الآلي التلقائي",
        "📈 Time Series" if lang == 'en' else "📈 سلاسل زمنية", 
        "🔮 What-If" if lang == 'en' else "🔮 ماذا لو",
        t('tab_training', lang), 
        t('tab_results', lang),
        "🚀 Deployment" if lang == 'en' else "🚀 النشر"
    ])
    
    with tab1:
        show_target_setup(df)
    
    with tab2:
        show_feature_engineering(df)
    
    with tab3:
        show_automl_tab(df)
        
    with tab4:
        show_time_series_tab(df)

    with tab5:
        show_what_if_tab(df)
    
    with tab6:
        show_model_training(df)
    
    with tab7:
        show_ml_results()
        
    with tab8:
        model_results = st.session_state.get('ml_results')
        automl_results = st.session_state.get('automl_results')
        
        # Use best available model
        best_result = automl_results if automl_results else model_results
        
        if best_result:
            # Get features list
            target = st.session_state.get('target')
            features = [c for c in df.columns if c != target]
            show_deployment_tab(df, best_result, features)
        else:
            st.info("Train a model first to enable this tab")


def show_target_setup(df):
    """Target variable selection"""
    lang = get_lang()
    
    st.markdown(f"### {t('select_target', lang)}")
    
    target = st.selectbox(t('target_column', lang), df.columns.tolist())
    
    if target:
        st.markdown(f"**{t('selected', lang)}:** `{target}`")
        
        # Show target distribution
        if df[target].dtype == 'object' or df[target].nunique() <= 10:
            st.markdown(f"**{t('classification_task', lang)}**")
            fig = px.histogram(df, x=target, title=f'Distribution of {target}')
        else:
            st.markdown(f"**{t('regression_task', lang)}**")
            fig = px.histogram(df, x=target, title=f'Distribution of {target}')
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.session_state.target = target


def show_feature_engineering(df):
    """Feature engineering interface"""
    lang = get_lang()
    
    st.markdown(f"### {t('feature_engineering', lang)}")
    
    operations = st.multiselect(
        t('select_operations_fe', lang),
        ['basic_stats', 'interactions', 'bins', 'datetime', 'polynomial'],
        default=['basic_stats']
    )
    
    if st.button(t('engineer_features', lang)):
        with st.spinner(t('engineering_features', lang)):
            engineer = FeatureEngineer()
            target = st.session_state.get('target')
            enhanced_df = engineer.engineer_features(df, target, operations)
            
            st.success(t('created_features', lang).format(len(engineer.created_features)))
            
            # Show new features
            summary = engineer.get_feature_summary()
            if not summary.empty:
                st.dataframe(summary, use_container_width=True)
            
            st.session_state.df = enhanced_df


def show_model_training(df):
    """Model training interface"""
    lang = get_lang()
    
    st.markdown(f"### {t('train_models', lang)}")
    
    target = st.session_state.get('target')
    
    if not target:
        st.warning(t('select_target_first', lang))
        return
    
    # Feature selection
    all_features = [col for col in df.columns if col != target]
    features = st.multiselect(t('select_features', lang), all_features, default=all_features[:10])
    
    if st.button(t('train_all', lang)):
        with st.spinner(t('training_models', lang)):
            pipeline = MLPipeline()
            results = pipeline.train(df, target, features)
            st.session_state.ml_results = results
            st.session_state.ml_pipeline = pipeline
            
            st.success(f"{t('training_complete', lang)}: **{results.best_model.name}**")


def show_automl_tab(df):
    """Advanced AutoML interface with Bayesian Optimization"""
    lang = get_lang()
    
    st.markdown("### 🚀 " + ("Advanced AutoML" if lang == 'en' else "التعلم الآلي المتقدم"))
    
    if not HAS_AUTOML:
        st.warning("⚠️ " + ("AutoML requires optuna package. Install with: pip install optuna" 
                           if lang == 'en' 
                           else "AutoML يتطلب حزمة optuna. ثبتها بـ: pip install optuna"))
        return
    
    target = st.session_state.get('target')
    
    if not target:
        st.warning("⚠️ " + ("Please select a target variable first" 
                           if lang == 'en' 
                           else "يرجى اختيار المتغير الهدف أولاً"))
        return
    
    st.info("💡 " + ("AutoML will automatically find the best model using Bayesian Optimization" 
                    if lang == 'en' 
                    else "سيجد AutoML أفضل نموذج تلقائياً باستخدام التحسين البايزي"))
    
    # Configuration
    st.markdown("#### ⚙️ " + ("Configuration" if lang == 'en' else "الإعدادات"))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_trials = st.slider(
            "🔄 " + ("Optimization Trials" if lang == 'en' else "محاولات التحسين"),
            min_value=10, max_value=200, value=50, step=10
        )
    
    with col2:
        timeout = st.slider(
            "⏱️ " + ("Timeout (minutes)" if lang == 'en' else "المهلة (دقائق)"),
            min_value=5, max_value=60, value=15, step=5
        )
    
    with col3:
        use_ensemble = st.checkbox(
            "🎭 " + ("Use Ensemble" if lang == 'en' else "استخدام التجميع"),
            value=True
        )
    
    # Advanced Feature Engineering option
    use_advanced_fe = st.checkbox(
        "🔧 " + ("Advanced Feature Engineering" if lang == 'en' else "هندسة ميزات متقدمة"),
        value=True
    )
    
    st.markdown("---")

    # =========================================================================
    # PHASE 6: HUMAN-AI BRIDGE (CONTEXT & CREATIVITY)
    # =========================================================================
    if HAS_NEW_FEATURES:
        from paths.ai.context_engine import ContextEngine
        
        st.markdown("### 🧠 " + ("Human-AI Bridge (Context & Creativity)" if lang == 'en' else "الجسر بين البشر والذكاء الاصطناعي"))
        
        with st.expander("✨ " + ("Inject Context & Brainstorm Features" if lang == 'en' else "إدخال السياق وتوليد أفكار"), expanded=True):
            
            # 1. Context Input
            context_input = st.text_area(
                "📝 " + ("Business Context / Domain Knowledge" if lang == 'en' else "سياق العمل / معرفة المجال"),
                placeholder="Ex: These are sales data for a retail store. Seasonality matters. We want to predict churn...",
                help="Tell the AI what this data is about to get smarter features."
            )
            
            # 2. Brainstorm Button
            if st.button("✨ " + ("Brainstorm Creative Features" if lang == 'en' else "عصف ذهني للميزات"), type="secondary"):
                if not context_input:
                    st.warning("⚠️ Please provide context first.")
                else:
                    with st.spinner("🤖 asking Gemini for ideas..."):
                        ai_assistant = AIAssistant(lang=lang)
                        context_engine = ContextEngine(ai_assistant)
                        ideas = context_engine.brainstorm_features(df, context_input, lang)
                        
                        if ideas:
                           st.session_state['feature_ideas'] = ideas
                           st.success(f"Generated {len(ideas)} ideas!")
            
            # 3. Display Ideas & Apply
            if 'feature_ideas' in st.session_state:
                ideas = st.session_state['feature_ideas']
                selected_indices = []
                
                st.markdown("#### 💡 " + ("Proposed Features" if lang == 'en' else "الميزات المقترحة"))
                
                for i, idea in enumerate(ideas):
                    col1, col2 = st.columns([0.1, 0.9])
                    with col1:
                        if st.checkbox("", key=f"idea_{i}"):
                            selected_indices.append(i)
                    with col2:
                        st.markdown(f"**{idea.name}**: {idea.description}")
                        st.caption(f"Reason: {idea.justification} | Complexity: {idea.complexity}")
                
                # 4. Generate & Apply Code
                if selected_indices:
                     if st.button("🚀 " + ("Generate & Apply Code" if lang == 'en' else "توليد وتطبيق الكود")):
                        with st.spinner("Writing Python code..."):
                             ai_assistant = AIAssistant(lang=lang)
                             context_engine = ContextEngine(ai_assistant)
                             
                             applied_count = 0
                             for idx in selected_indices:
                                 idea = ideas[idx]
                                 code = context_engine.generate_feature_code(df, idea)
                                 
                                 if code:
                                     try:
                                         # Execute on SESSION STATE DF
                                         df_new = context_engine.safe_execute_feature(st.session_state.df, code)
                                         st.session_state.df = df_new # Update main DF
                                         applied_count += 1
                                         st.toast(f"✅ Applied: {idea.name}")
                                     except Exception as e:
                                         st.error(f"Failed to apply {idea.name}: {e}")
                             
                             if applied_count > 0:
                                 st.success(f"Successfully added {applied_count} new features!")
                                 st.rerun() # Rerun to update df reference

    
    # Run AutoML
    start_automl = st.button("🚀 " + ("Start AutoML Training" if lang == 'en' else "بدء تدريب AutoML"), 
                 type="primary", use_container_width=True)

    if start_automl:
        
        from paths.ai.ai_ensemble import get_ensemble
        
        # =========================================================
        # CHIEF DATA SCIENTIST: EXPERT THINKING LAYER
        # =========================================================
        ai_ensemble = get_ensemble()
        chief_ds = ChiefDataScientist(ai_ensemble)
        
        # Initialize Omni-Logic if available
        analytical = AnalyticalLogic(ai_ensemble) if HAS_OMNI_LOGIC else None
        causal = CausalLogic(ai_ensemble) if HAS_OMNI_LOGIC else None
        ethical = EthicalLogic(ai_ensemble) if HAS_OMNI_LOGIC else None
        
        progress_container = st.container()
        
        with progress_container:
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            # --- THINKING STAGE 1: PROBLEM REFRAMING ---
            status_text.text("🧠 " + ("Stage 1: Problem Reframing..." if lang == 'en' else "المرحلة 1: إعادة صياغة المشكلة..."))
            progress_bar.progress(5)
            
            stage1 = chief_ds.stage1_problem_reframing(df, target, lang)
            
            if stage1.status.value == "rejected": 
                 st.error(stage1.reasoning)
                 
                 # EXPERT RECOVERY MODE
                 if st.checkbox("🔧 " + ("Show Expert Recovery Plan" if lang == 'en' else "عرض خطة التعافي الخبيرة"), value=True):
                     recovery_plan = chief_ds.generate_recovery_plan(df, target, lang)
                     
                     st.warning("🚑 " + ("Expert Recovery Mode Activated" if lang == 'en' else "تفعيل وضع التعافي الخبير"))
                     
                     # 1. Root Cause
                     st.markdown("#### 🔍 " + ("Diagnosis" if lang == 'en' else "التشخيص"))
                     for diag in recovery_plan['root_cause_diagnosis']:
                         st.error(f"{diag['severity_icon']} {diag['concern']}")
                         st.caption(f"Reason: {diag['statistical_reason']}")
                     
                     # 2. Repair Plan
                     st.markdown("#### 🛠️ " + ("Repair Options" if lang == 'en' else "خيارات الإصلاح"))
                     for repair in recovery_plan['repair_plan']:
                         with st.expander(f"Fix: {repair['issue']}"):
                             st.markdown(f"**Conservative:** {repair['fix_conservative']}")
                             st.markdown(f"**Aggressive:** {repair['fix_aggressive']}")
                             st.info(f"🚫 When NOT to fix: {repair['when_not_to_fix']}")
                     
                     # 3. Auto Fixes
                     st.markdown("#### 🤖 " + ("Auto-Fix Candidates" if lang == 'en' else "إصلاحات تلقائية مقترحة"))
                     safe_fixes = recovery_plan['auto_fix_candidates']['safe']
                     if safe_fixes:
                         st.success(f"Found {len(safe_fixes)} safe fixes.")
                         if st.button("🚀 " + ("Apply Safe Fixes & Retry" if lang == 'en' else "تطبيق الإصلاحات وإعادة المحاولة")):
                             df_fixed, changes = chief_ds.apply_safe_fixes(df, target)
                             st.session_state.df = df_fixed
                             st.success("\n".join(changes))
                             st.rerun()
                     else:
                         st.info("No safe auto-fixes available. Manual intervention required.")
                 
                 st.stop()
                 
            # --- THINKING STAGE 2: DATA SKEPTICISM ---
            status_text.text("🔍 " + ("Stage 2: Data Skepticism..." if lang == 'en' else "المرحلة 2: الشك في البيانات..."))
            progress_bar.progress(10)
            stage2 = chief_ds.stage2_data_skepticism(df, target, lang)
            
            if stage2.status.value == "rejected":
                 st.error(stage2.reasoning)
                 st.stop()
            
            # --- THINKING STAGE 3: STRATEGY ---
            status_text.text("📋 " + ("Stage 3: Strategic Planning..." if lang == 'en' else "المرحلة 3: التخطيط الاستراتيجي..."))
            progress_bar.progress(15)
            stage3 = chief_ds.stage3_analysis_strategy(df, target, lang)
            
            if stage3.status.value == "rejected":
                st.error(stage3.reasoning)
                st.stop()

            # --- OMNI-LOGIC ANALYTICAL LAYER ---
            if analytical:
                status_text.text("🧠 " + ("Analytical Deep Dive..." if lang == 'en' else "تحليل عميق..."))
                analytical_result = analytical.analyze(df, target, lang)
                with st.expander("📊 Analytical Insights", expanded=False):
                    st.write(analytical_result)

        # If approved, continue to original logic...
        with progress_container:

            progress_bar = st.progress(20)
            status_text = st.empty()
            
            # Step 1: Feature Engineering
            if use_advanced_fe and HAS_ADVANCED_FE:
                status_text.text("🔧 " + ("Engineering features..." if lang == 'en' else "هندسة الميزات..."))
                progress_bar.progress(25)
                
                fe = AdvancedFeatureEngineer(max_features=100)
                enhanced_df = fe.fit_transform(df.copy(), target)
                
                st.success("✅ " + (f"Created {len(fe.created_features)} new features" 
                                   if lang == 'en' 
                                   else f"تم إنشاء {len(fe.created_features)} ميزة جديدة"))
            else:
                enhanced_df = df.copy()
            
            progress_bar.progress(30)

            # --- OMNI-LOGIC CAUSAL LAYER ---
            if causal:
                 status_text.text("🔗 " + ("Causal Analysis..." if lang == 'en' else "تحليل السببية..."))
                 causal_result = causal.analyze(enhanced_df, target, lang)
            
            # Step 2: AutoML Training
            status_text.text("🤖 " + ("Running Bayesian Optimization..." if lang == 'en' else "تشغيل التحسين البايزي..."))
            
            automl = AdvancedAutoML(
                n_trials=n_trials,
                timeout_minutes=timeout,
                use_ensemble=use_ensemble,
                random_state=42
            )
            
            # Get features
            features = [c for c in enhanced_df.columns if c != target]
            
            # Run AutoML
            results = automl.fit(enhanced_df, target, features)
            
            progress_bar.progress(90)
            
            # --- EXPERT SELF-CRITIQUE ---
            status_text.text("⚖️ " + ("Performing Self-Critique..." if lang == 'en' else "إجراء نقد ذاتي..."))
            critique = chief_ds.generate_self_critique(results.__dict__, lang)
            results.critique = critique # Attach critique to results
            
            # --- OMNI-LOGIC ETHICAL LAYER ---
            if ethical:
                status_text.text("🛡️ " + ("Ethical Audit..." if lang == 'en' else "تدقيق أخلاقي..."))
                ethical_audit = ethical.audit(enhanced_df, target, results.__dict__, lang)
                results.ethical_audit = ethical_audit

            # Store results
            st.session_state.automl_results = results
            st.session_state.automl_model = automl
            
            progress_bar.progress(100)
            status_text.text("✅ " + ("Training complete!" if lang == 'en' else "اكتمل التدريب!"))
        
        # Show results
        st.markdown("---")
        
        # EXPERT INTERPRETATION SECTION
        if hasattr(results, 'critique') and results.critique:
            st.markdown("### 🧙‍♂️ " + ("Chief Data Scientist Opinion" if lang == 'en' else "رأي كبير علماء البيانات"))
            
            critique = results.critique
            conf_level = critique.get('confidence_level', 'medium').upper()
            
            # Confidence Banner
            if conf_level == 'HIGH':
                st.success(f"✅ CONFIDENCE: HIGH")
            elif conf_level == 'MEDIUM':
                st.warning(f"📊 CONFIDENCE: MEDIUM")
            else:
                st.error(f"⚠️ CONFIDENCE: LOW")
                
            # Warnings
            if critique['expert_warnings']:
                 st.markdown("#### ⚠️ " + ("Expert Warnings" if lang == 'en' else "تحذيرات الخبير"))
                 for warn in critique['expert_warnings']:
                     st.warning(warn)

        st.markdown("### 🏆 " + ("Results" if lang == 'en' else "النتائج"))
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 " + ("Best Score" if lang == 'en' else "أفضل نتيجة"),
                f"{results.best_score:.4f}"
            )
        
        with col2:
            st.metric(
                "📊 " + ("CV Mean" if lang == 'en' else "متوسط CV"),
                f"{np.mean(results.cv_scores):.4f}"
            )
        
        with col3:
            st.metric(
                "⏱️ " + ("Time" if lang == 'en' else "الوقت"),
                f"{results.optimization_time:.1f}s"
            )
        
        with col4:
            st.metric(
                "🔄 " + ("Trials" if lang == 'en' else "المحاولات"),
                len(results.all_trials)
            )
        
        # Best model info
        st.markdown("#### 🥇 " + ("Best Model" if lang == 'en' else "أفضل نموذج"))
        st.json(results.best_params)
        
        # Feature importance
        if results.feature_importance:
            st.markdown("#### 📊 " + ("Feature Importance" if lang == 'en' else "أهمية الميزات"))
            
            imp_df = pd.DataFrame([
                {'Feature': k, 'Importance': v}
                for k, v in list(results.feature_importance.items())[:15]
            ]).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                imp_df, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Trial history
        if results.all_trials:
            st.markdown("#### 📈 " + ("Optimization History" if lang == 'en' else "تاريخ التحسين"))
            
            trial_df = pd.DataFrame([
                {'Trial': i+1, 'Score': t.get('cv_score', t.get('score', 0)), 'Model': t.get('model_type', 'unknown')}
                for i, t in enumerate(results.all_trials)
            ])
            
            fig = px.line(
                trial_df, 
                x='Trial', 
                y='Score',
                color='Model',
                title="Score by Trial"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)


def show_ml_results():
    """Show ML results"""
    lang = get_lang()
    
    st.markdown(f"### {t('model_results', lang)}")
    
    results = st.session_state.get('ml_results')
    pipeline = st.session_state.get('ml_pipeline')
    
    if not results:
        st.info(t('train_first', lang))
        return
    
    # Best model
    st.markdown(f"#### {t('best_model', lang)}: {results.best_model.name}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(t('cv_score', lang), f"{results.best_model.cv_mean:.4f}")
    with col2:
        st.metric(t('test_score', lang), f"{results.best_model.test_score:.4f}")
    with col3:
        st.metric(t('train_score', lang), f"{results.best_model.train_score:.4f}")
    
    # Model comparison
    st.markdown(f"#### {t('model_comparison', lang)}")
    comparison_df = pipeline.get_model_comparison()
    st.dataframe(comparison_df, use_container_width=True)
    
    # Feature importance
    if results.best_model.feature_importance:
        st.markdown(f"#### {t('feature_importance', lang)}")
        imp_df = pd.DataFrame([
            {'Feature': k, 'Importance': v}
            for k, v in results.best_model.feature_importance.items()
        ]).sort_values('Importance', ascending=False).head(10)
        
        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                    title=t('feature_importance', lang))
        st.plotly_chart(fig, use_container_width=True)

    # --- Export Section ---
    st.markdown("---")
    st.markdown(f"### 📥 {t('export_results', lang) if t('export_results', lang) != 'export_results' else 'Export Reports'}")
    
    col_exp1, col_exp2 = st.columns(2)
    
    # Prepare data for export
    # We use current session state DF as 'clean data'. 
    # If we have predictions (Deployment tab logic), we might want them here but 
    # results object usually just has metrics. 
    # Let's check if we can get predictions on current df.
    
    df_clean = st.session_state.df
    
    # 1. Excel Dashboard
    with col_exp1:
        if st.button("📊 " + ("Download Excel Dashboard" if lang == 'en' else "تحميل لوحة Excel")):
            with st.spinner("Generating Dashboard..."):
                # Basic analysis summary
                analysis_summary = {
                    'overview': {
                        'rows': len(df_clean),
                        'columns': len(df_clean.columns)
                    }
                }
                
                excel_bytes = create_excel_report(
                    df_clean=df_clean,
                    df_predictions=None, # Training phase doesn't necessarily produce predictions on new data
                    feature_importance=pd.DataFrame(list(results.best_model.feature_importance.items()), columns=['Feature', 'Importance']) if results.best_model.feature_importance else None,
                    metrics={
                        'CV Score': results.best_model.cv_mean,
                        'Test Score': results.best_model.test_score,
                        'Train Score': results.best_model.train_score
                    },
                    analysis=analysis_summary,
                    lang=lang
                )
                
                st.download_button(
                    label="📥 " + ("Save Excel File" if lang == 'en' else "حفظ ملف Excel"),
                    data=excel_bytes,
                    file_name="AI_Expert_Dashboard.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="btn_dl_excel_res"
                )

    # 2. Power BI Package
    with col_exp2:
        if st.button("📊 " + ("Download Power BI Package" if lang == 'en' else "تحميل حزمة Power BI")):
            with st.spinner("Generating Package..."):
                pbi_bytes = create_powerbi_package(
                    df_clean=df_clean,
                    df_predictions=None,
                    analysis={'overview': {'generated_by': 'Data Science Hub'}},
                    lang=lang
                )
                
                st.download_button(
                    label="📥 " + ("Save Power BI ZIP" if lang == 'en' else "حفظ حزمة ZIP"),
                    data=pbi_bytes,
                    file_name="PowerBI_Ready_Data.zip",
                    mime="application/zip",
                    key="btn_dl_pbi_res"
                )


def show_knowledge_hub(lang):
    """Show knowledge resources from data_science_master_system"""
    import os
    
    st.markdown("### 📚 " + ("Knowledge Hub & Cheatsheets" if lang == 'en' else "مركز المعرفة والملخصات"))
    
    # Path to master system: Go up one level from DataAnalystProject to 'Data Scientist' root, then into master system
    # PROJECT_ROOT is .../DataAnalystProject
    master_path = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'data_science_master_system', 'notebooks'))
    
    if not os.path.exists(master_path):
        st.error(f"Cannot find data_science_master_system at: {master_path}")
        st.info("Expected structure: \n- Data Scientist/\n  - DataAnalystProject/\n  - data_science_master_system/")
        return

    # Recursive search for markdown files
    cheatsheets = []
    for root, dirs, files in os.walk(master_path):
        for file in files:
            if file.endswith(".md") and "cheatsheet" in file.lower():
                full_path = os.path.join(root, file)
                # Create readable name
                rel_path = os.path.relpath(full_path, master_path)
                cheatsheets.append((rel_path, full_path))
    
    if cheatsheets:
        selected_cs = st.selectbox(
            "Select Cheatsheet" if lang == 'en' else "اختر ملخص",
            cheatsheets,
            format_func=lambda x: x[0]
        )
        
        if selected_cs:
            with open(selected_cs[1], 'r', encoding='utf-8') as f:
                content = f.read()
            st.markdown(content)
    else:
        st.info("No cheatsheets found.")


def show_time_series_tab(df):
    """Show Time Series Forecasting interface"""
    lang = get_lang()
    
    st.markdown("### 📈 " + ("Time Series Forecasting" if lang == 'en' else "التنبؤ بالسلاسل الزمنية"))
    
    if HAS_NEW_FEATURES:
        # Date column selection
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        
        # Try to detect date columns if not typed correctly
        if not datetime_cols:
            for c in df.columns:
                if df[c].dtype == 'object':
                    try:
                        pd.to_datetime(df[c].head(100))
                        datetime_cols.append(c)
                    except:
                        pass
        
        if datetime_cols:
            date_col = st.selectbox(
                "Select Date Column" if lang == 'en' else "اختر عمود التاريخ",
                datetime_cols
            )
            steps = st.slider(
                "Forecast Horizon (Steps)" if lang == 'en' else "أفق التنبؤ (خطوات)",
                min_value=1, max_value=365, value=30
            )
            
            if st.button("📈 " + ("Generate Forecast" if lang == 'en' else "توليد التنبؤ")):
                with st.spinner("Forecasting..."):
                    forecaster = TimeSeriesForecaster("prophet")
                    forecast_df, metrics = forecaster.forecast(df, date_col, target_col, steps)
                    
                    st.success("✅ Forecast Generated")
                    
                    # Store result in session state to persist
                    st.session_state.ts_result = {
                        'forecast_df': forecast_df,
                        'metrics': metrics
                    }

            if 'ts_result' in st.session_state:
                result = st.session_state.ts_result
                forecast_df = result['forecast_df']
                metrics = result['metrics']
                
                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{metrics['mae']:.2f}")
                c2.metric("RMSE", f"{metrics['rmse']:.2f}")
                c3.metric("R2", f"{metrics['r2']:.2f}")
                
                # Plot
                fig = px.line(forecast_df, x='ds', y='yhat', title="Forecast")
                fig.add_scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', line=dict(width=0), showlegend=False)
                fig.add_scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,80,0.2)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.warning("⚠️ No date column detected. Please ensure your data has a valid datetime column." if lang == 'en' else "⚠️ لم يتم اكتشاف عمود تاريخ. يرجى التأكد من وجود عمود تاريخ صالح.")


def show_deployment_tab(df, model, features):
    """Show Deployment interface"""
    lang = get_lang()
    
    st.markdown("### 🚀 " + ("Deployment & API" if lang == 'en' else "النشر و API"))
    
    if HAS_NEW_FEATURES:
        if not model:
            st.warning("⚠️ " + ("Please train a model first" if lang == 'en' else "يرجى تدريب نموذج أولاً"))
            return
            
        st.info("💡 " + ("Generate a REST API for your model using FastAPI and Docker" 
                        if lang == 'en' 
                        else "إنشاء REST API لنموذجك باستخدام FastAPI و Docker"))
        
        col1, col2 = st.columns(2)
        with col1:
            api_title = st.text_input("API Title", "My Model API")
        with col2:
            port = st.number_input("Port", value=8000)
            
        if st.button("📦 " + ("Generate API Package" if lang == 'en' else "إنشاء حزمة API")):
            with st.spinner("Generating..."):
                from paths.deployment.api_generator import APIGenerator, APIConfig
                
                generator = APIGenerator(output_dir="deployment_output")
                config = APIConfig(title=api_title, port=port)
                
                # Use model name from trained model
                files = generator.generate_api(
                    model=model.best_model if hasattr(model, 'best_model') else model,
                    feature_names=features,
                    config=config
                )
                
                st.success("✅ " + ("API generated successfully!" if lang == 'en' else "تم إنشاء API بنجاح!"))
                
                st.markdown("#### 📂 " + ("Generated Files" if lang == 'en' else "الملفات المنشأة"))
                for name, path in files.items():
                    st.code(f"{name}: {path}")
                    
                st.info("To run the API:\n1. Open terminal\n2. cd deployment_output\n3. python main.py")

        st.markdown("---")
        st.markdown("### 🔄 " + ("Automated Pipeline" if lang == 'en' else "Pipeline آلي"))
        
        if st.button("🔌 " + ("Generate Pipeline Script" if lang == 'en' else "إنشاء سكربت Pipeline")):
             with st.spinner("Generating Pipeline..."):
                from paths.deployment.pipeline_gen import PipelineGenerator
                
                pipe_gen = PipelineGenerator(output_dir="deployment_output")
                
                # Basic config derivation
                target = st.session_state.get('target', 'target')
                
                # Mock config as we don't have full history tracking implemented yet
                cleaning_config = {}
                fe_config = ['basic_stats'] 
                
                # Save model first for pipeline to pick up
                import joblib
                import os
                if not os.path.exists("deployment_output"): os.makedirs("deployment_output")
                model_path = "deployment_output/model.joblib"
                
                actual_model = model.best_model if hasattr(model, 'best_model') else model
                joblib.dump(actual_model, model_path)
                
                script_path = pipe_gen.generate_pipeline(
                    cleaning_config=cleaning_config,
                    fe_config=fe_config,
                    model_path="model.joblib",
                    target=target,
                    features=features
                )
                
                st.success("✅ " + ("Pipeline script generated!" if lang == 'en' else "تم إنشاء Pipeline!"))
                st.code(f"Location: {script_path}")
                st.info("Run with: python deployment_output/pipeline.py --input data.csv")


def show_ai_chat(df=None):
    """Show AI Assistant Chat"""
    lang = get_lang()
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 " + ("AI Assistant" if lang == 'en' else "المساعد الذكي"))
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Chat interface
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        if prompt := st.chat_input("Ask about data..." if lang == 'en' else "اسأل عن البيانات..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if HAS_NEW_FEATURES:
                        assistant = AIAssistant(lang=lang)
                        # Basic context
                        context = {"rows": len(df)} if df is not None else {}
                        response = assistant.chat(prompt, context)
                    else:
                        response = "AI module not loaded."
                    
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})


def show_advanced_viz_tab(df):
    """Show Advanced Visualization interface"""
    lang = get_lang()
    
    st.markdown("### 🎨 " + ("Advanced Visualizations" if lang == 'en' else "الرسوم البيانية المتقدمة"))
    
    if HAS_NEW_FEATURES:
        viz = AdvancedVisualizer(lang=lang)
        
        chart_type = st.selectbox(
            "Select Chart Type" if lang == 'en' else "اختر نوع الرسم البياني",
            ["Sunburst", "Sankey", "3D Scatter", "Geo Map", "Geo Scatter"]
        )
        
        st.markdown("---")
        
        if chart_type == "Sunburst":
            # Broaden categorical detection: any column with <= 25 unique values
            cat_cols = [c for c in df.columns if df[c].nunique() <= 25]
            
            cols = st.multiselect(
                "Select Hierarchy Columns (Order matters)" if lang == 'en' else "اختر أعمدة التسلسل الهرمي (الترتيب مهم)",
                cat_cols
            )
            value_col = st.selectbox(
                "Value Column" if lang == 'en' else "عمود القيمة",
                df.select_dtypes(include=[np.number]).columns
            )
            
            if cols and value_col:
                fig = viz.plot_sunburst(df, cols, value_col)
                st.plotly_chart(fig, use_container_width=True)
                
        elif chart_type == "3D Scatter":
            c1, c2, c3 = st.columns(3)
            num_cols = df.select_dtypes(include=[np.number]).columns
            with c1: x = st.selectbox("X Axis", num_cols, index=0)
            with c2: y = st.selectbox("Y Axis", num_cols, index=1 if len(num_cols)>1 else 0)
            with c3: z = st.selectbox("Z Axis", num_cols, index=2 if len(num_cols)>2 else 0)
            color = st.selectbox("Color", df.columns)
            
            if x and y and z:
                fig = viz.plot_3d_scatter(df, x, y, z, color)
                st.plotly_chart(fig, use_container_width=True)
                
        elif chart_type == "Sankey":
            # Broaden categorical detection: any column with <= 25 unique values
            cat_cols = [c for c in df.columns if df[c].nunique() <= 25]
            
            num_cols = df.select_dtypes(include=[np.number]).columns
            
            c1, c2, c3 = st.columns(3)
            with c1: source = st.selectbox("Source", cat_cols, index=0 if len(cat_cols) > 0 else None)
            with c2: target = st.selectbox("Target", cat_cols, index=1 if len(cat_cols)>1 else 0 if len(cat_cols) > 0 else None)
            with c3: value = st.selectbox("Value", num_cols)
            
            if source and target and value:
                fig = viz.plot_sankey(df, source, target, value)
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Geo Map" or chart_type == "Geo Scatter":
            st.info("Requires country names or ISO codes" if lang == 'en' else "يتطلب أسماء دول أو رموز ISO")
            loc_col = st.selectbox("Location Column" if lang == 'en' else "عمود الموقع", df.columns)
            val_col = st.selectbox("Value Column" if lang == 'en' else "عمود القيمة", 
                                 df.select_dtypes(include=[np.number]).columns)
                                 
            if chart_type == "Geo Map":
                fig = viz.plot_geo_map(df, loc_col, val_col)
            else:
                fig = viz.plot_geo_scatter(df, loc_col, loc_col, size_col=val_col)
                
            st.plotly_chart(fig, use_container_width=True)


def show_what_if_tab(df):
    """Show What-If Analysis interface"""
    lang = get_lang()
    
    st.markdown("### 🔮 " + ("What-If Analysis" if lang == 'en' else "تحليل ماذا لو"))
    
    results = st.session_state.get('ml_results')
    automl_results = st.session_state.get('automl_results')
    
    # Get model 
    model = None
    if automl_results:
        model = st.session_state.get('automl_model') # This might need adjustment based on how automl stores model
        # Actually automl_results.best_model might be better but it lacks predict method sometimes if it's just result object
        # Let's assume user has trained a model via standard pipeline for now if automl isn't fully adaptable
    
    pipeline = st.session_state.get('ml_pipeline')
    if pipeline and hasattr(pipeline, 'best_model'):
        model = pipeline.best_model
    
    if not model:
        st.warning("⚠️ " + ("Please train a model first (Standard Training)" if lang == 'en' else "يرجى تدريب نموذج أولاً (التدريب القياسي)"))
        return

    # Use features from session state or detect
    target = st.session_state.get('target')
    if not target: return
    
    features = [c for c in df.columns if c != target] # Simplification
    
    if HAS_NEW_FEATURES:
        simulator = WhatIfSimulator(model, features)
        
        tabs = st.tabs(["Scenario Simulation", "Sensitivity Analysis"])
        
        with tabs[0]:
            st.markdown("#### " + ("Simulate Changes" if lang == 'en' else "محاكاة التغييرات"))
            
            col1, col2 = st.columns(2)
            with col1:
                selected_feature = st.selectbox("Select Feature" if lang == 'en' else "اختر الميزة", features)
            
            with col2:
                change_type = st.radio("Change Type" if lang == 'en' else "نوع التغيير", 
                                     ["Fixed Value", "Percentage Increase"], horizontal=True)
            
            if change_type == "Fixed Value":
                val = st.number_input("New Value" if lang == 'en' else "قيمة جديدة")
                change = val
            else:
                pct = st.slider("Increase %" if lang == 'en' else "زيادة %", -50, 50, 10)
                change = lambda x: x * (1 + pct/100)
                
            if st.button("🔮 " + ("Simulate" if lang == 'en' else "محاكاة")):
                with st.spinner("Simulating..."):
                    res_df = simulator.simulate_scenario(df, {selected_feature: change})
                    
                    if res_df is not None:
                        # Show aggregate impact
                        avg_diff = res_df['Difference'].mean()
                        avg_pct = res_df['Percent Change'].mean()
                        
                        c1, c2 = st.columns(2)
                        c1.metric("Avg Impact" if lang == 'en' else "متوسط التأثير", f"{avg_diff:.4f}")
                        c2.metric("Avg % Change", f"{avg_pct:.2f}%")
                        
                        st.markdown("#### Detailed View")
                        st.dataframe(res_df.head(50), use_container_width=True)
                        
                        # Plot distribution comparison
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=res_df['Original Prediction'], name='Original', opacity=0.7))
                        fig.add_trace(go.Histogram(x=res_df['New Prediction'], name='New Scenario', opacity=0.7))
                        fig.update_layout(barmode='overlay', title="Prediction Distribution Shift")
                        st.plotly_chart(fig, use_container_width=True)
                        
        with tabs[1]:
            st.markdown("#### " + ("Sensitivity Analysis" if lang == 'en' else "تحليل الحساسية"))
            
            sens_feat = st.selectbox("Feature to Analyze" if lang == 'en' else "ميزة للتحليل", features)
            
            if st.button("📉 " + ("Analyze Sensitivity" if lang == 'en' else "تحليل الحساسية")):
                # Check numeric
                if np.issubdtype(df[sens_feat].dtype, np.number):
                    min_v = float(df[sens_feat].min())
                    max_v = float(df[sens_feat].max())
                    
                    sens_df = simulator.sensitivity_analysis(df, sens_feat, min_v, max_v, steps=20)
                    
                    if sens_df is not None:
                        fig = px.line(sens_df, x=sens_feat, y='Prediction', 
                                    title=f"Impact of {sens_feat} on Prediction")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Select numeric feature for sensitivity analysis")


def main():
    """Main application entry point"""
    init_session_state()
    
    try:
        # Direct routing
        if st.session_state.path == 'analyst':
            show_data_analyst_path()
        elif st.session_state.path == 'scientist':
            show_data_scientist_path()
        else:
            show_path_selection()
    except Exception as e:
        if HAS_NEW_FEATURES:
            sentinel = get_sentinel()
            log_id = sentinel.log_error(e, context={
                "path": st.session_state.get('path'),
                "domain": st.session_state.get('domain'),
                "step": "main_routing"
            })
            st.error(f"⚠️ A system error occurred. AI Sentinel has recorded this (ID: {log_id}) for immediate maintenance.")
            if st.button("Go to Maintenance Center" if get_lang() == 'en' else "الذهاب لمركز الصيانة"):
                st.session_state.path = 'analyst'
                st.rerun()
        else:
            st.error(f"Critical Error: {e}")

    # Show AI Assistant if data is loaded
    if st.session_state.df is not None:
        if HAS_NEW_FEATURES:
            show_knowledge_hub(get_lang())
        show_ai_chat(st.session_state.df)


if __name__ == "__main__":
    main()
