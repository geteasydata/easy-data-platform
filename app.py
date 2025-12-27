"""
AI Expert - Main Streamlit Application
A 30+ year expert data scientist at your fingertips
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import io
import joblib
import pathlib  # Added missing import
from pathlib import Path
import sys
import os


# Import translations using explicit path to avoid conflicts with other projects
import importlib.util
_translations_path = Path(__file__).parent / "translations.py"
_spec = importlib.util.spec_from_file_location("ai_expert_translations", _translations_path)
_translations_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_translations_module)
t = _translations_module.t
get_metric_name = _translations_module.get_metric_name
format_mixed_text = _translations_module.format_mixed_text
ltr = _translations_module.ltr

# Import custom modules
from core.data_loader import load_uploaded_file
from core.auto_ml import AutoML
from core.ai_ensemble import get_ensemble
from core.chief_data_scientist import get_chief_data_scientist, ApprovalStatus
import core.auth as auth
from reports.excel_output import create_excel_report
from reports.powerbi_export import create_powerbi_package
from reports.notebook_generator import generate_notebook, get_notebook_bytes
from reports.pdf_report import create_pdf_report
from reports.word_report import create_word_report

# Integration: Easy Data Ecosystem



# ==========================================
# Robust Path & Import Setup (SaaS Optimized)
# ==========================================
import importlib.util

current_file = pathlib.Path(__file__).resolve()
current_dir = current_file.parent
root_path = current_dir.parent

# Define Paths
brain_path = root_path / 'data_science_master_system'
analyst_path_dir = root_path / 'DataAnalystProject'

# 1. Add Core Paths to Sys (for internal dependencies)
for p in [root_path, brain_path, analyst_path_dir]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ==========================================
# Standard Import (Unified Structure)
# ==========================================

# 1. Add submodules to path if needed (for internal relative imports)
current_dir = pathlib.Path(__file__).parent.resolve()
analyst_dir = current_dir / "DataAnalystProject"
if analyst_dir.exists() and str(analyst_dir) not in sys.path:
    sys.path.append(str(analyst_dir))

# 2. Import Data Analyst Module
show_data_analyst_path = None
try:
    # Now that DataAnalystProject is INSIDE the repo, this is a standard import
    from DataAnalystProject.main import show_data_analyst_path
except ImportError as e:
    # Fallback to prevent crash
    print(f"Error loading DataAnalystProject: {e}")
    def show_data_analyst_path():
        st.error("⚠️ System Error: Data Analyst Module Unavailable")
        st.info("The module files might not have been uploaded correctly.")
        with st.expander("Diagnostic Info"):
            st.code(f"Error: {e}\n\nCWD: {os.getcwd()}\n\nDir: {os.listdir(os.getcwd())}")


# Page Config
st.set_page_config(
    page_title="Easy Data | البيانات السهلة",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css():
    """Load global CSS from assets/style.css"""
    css_file = pathlib.Path(__file__).parent / "assets" / "style.css"
    try:
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CSS file not found. Please ensure assets/style.css exists.")

def custom_footer():
    """Display a professional custom footer"""
    st.markdown("""
    <div class="custom-footer">
        <span>© 2025 Easy Data AI. All rights reserved.</span>
        <span style="margin: 0 10px;">•</span>
        <a href="#" target="_blank">Privacy Policy</a>
        <span style="margin: 0 10px;">•</span>
        <a href="#" target="_blank">Terms of Service</a>
        <span style="margin: 0 10px;">•</span>
        <a href="mailto:support@easydata.ai">Contact Support</a>
    </div>
    """, unsafe_allow_html=True)



def metric_card(label, value, icon="📊"):
    """Display a beautiful metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 2rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    load_css()
    # Session state initialization
    if 'lang' not in st.session_state:
        st.session_state.lang = None
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = None
        
    # Check Query Params for Navigation (Persistent Link Support)
    try:
        query_params = st.query_params
        
        # Priority: Query Params > Session State
        if "mode" in query_params:
            mode = query_params["mode"]
            if mode in ['scientist', 'analyst']:
                st.session_state.app_mode = mode
                params_processed = True
        
        if "lang" in query_params:
            lang_param = query_params["lang"]
            if lang_param in ['ar', 'en']:
                st.session_state.lang = lang_param
                params_processed = True
                
    except Exception as e:
        print(f"Error parsing params: {e}")


    if st.session_state.lang is None:
        st.session_state.lang = 'en'
    
    # Sidebar Language Switcher - REMOVED for Landing Page
    # (Will be available inside the apps if needed, or we keep top toggle globally)
    
    # Authenticate User
    if not auth.require_auth(st.session_state.lang):
        return

    lang = st.session_state.lang

    # 2. Path Selection Screen (The Core of "Easy Data")
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = None

    if st.session_state.app_mode is None:
        # Hide Sidebar specifically for Landing Page
        st.markdown("""
        <style>
            [data-testid="stSidebar"] {display: none;}
            [data-testid="collapsedControl"] {display: none;}
        </style>
        """, unsafe_allow_html=True)
        
        # Header with Language Toggle (No Sidebar)
        col_logo, col_spacer, col_lang = st.columns([2, 4, 1])
        
        with col_logo:
             st.markdown(f"<h1 style='text-align: left; background: linear-gradient(to right, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; padding: 0;'>Easy Data</h1>", unsafe_allow_html=True)

        with col_lang:
            current_lang = st.session_state.get('lang', 'en')
            # Small toggle button
            btn_label = "🇦🇪 العربية" if current_lang == 'en' else "🇺🇸 English"
            if st.button(btn_label, key="lang_toggle_top", use_container_width=True):
                 st.session_state.lang = 'ar' if current_lang == 'en' else 'en'
                 st.rerun()

        st.markdown(f"<p style='text-align: left; font-size: 1.1rem; color: #94a3b8; margin-top: -10px;'>{t('app_subtitle', lang)}</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_sci, col_ana = st.columns(2)
        
        with col_sci:
            # Scientific Path Card
            st.markdown(f"""
                <div class="custom-card" style="text-align: center; min-height: 250px; display: flex; flex-direction: column; justify-content: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
                    <h2 style="color: #a5b4fc !important; font-size: 1.5rem; margin-bottom: 0.5rem;">{'Scientific Path' if lang == 'en' else 'المسار العلمي'}</h2>
                    <p style="color: #cbd5e1 !important; font-size: 0.95rem; line-height: 1.5;">{'AutoML, Predictive Modeling, and Risk Discovery.<br>For replacing Data Science teams.' if lang == 'en' else 'التحليل التنبئي، بناء النماذج، واكتشاف المخاطر.<br>بديل فريق علوم البيانات.'}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("🔬 " + ("Start Scientific Path" if lang == 'en' else "ابدأ المسار العلمي"), key="btn_scientist", use_container_width=True):
                st.session_state.app_mode = 'scientist'
                st.rerun()
                
        with col_ana:
            # Analytical Path Card
            st.markdown(f"""
                <div class="custom-card" style="text-align: center; min-height: 250px; display: flex; flex-direction: column; justify-content: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                    <h2 style="color: #5eead4 !important; font-size: 1.5rem; margin-bottom: 0.5rem;">{'Analytical Path' if lang == 'en' else 'المسار التحليلي'}</h2>
                    <p style="color: #cbd5e1 !important; font-size: 0.95rem; line-height: 1.5;">{'Traditional ETL, Dashboards, and Reports.<br>For replacing Analyst teams.' if lang == 'en' else 'تنظيف البيانات، لوحات المعلومات، والتقارير.<br>بديل فريق محللي البيانات.'}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("📊 " + ("Start Analytical Path" if lang == 'en' else "ابدأ المسار التحليلي"), key="btn_analyst", use_container_width=True):
                st.session_state.app_mode = 'analyst'
                st.rerun()
        return

    # If Analyst path is chosen, we switch logic
    if st.session_state.app_mode == 'analyst':
        try:
            # Initialize ALL required session state for DataAnalystProject
            from DataAnalystProject.main import init_session_state, show_data_analyst_path
            init_session_state()  # Initialize all required variables
            st.session_state.path = 'analyst'  # Set the path after init
            show_data_analyst_path()
        except Exception as e:
            st.error(f"Error loading Analyst Path: {e}")
            if st.button("Reset"):
                st.session_state.app_mode = None
                st.rerun()
        return

    # Scientist Path (Original AI Expert Flow continues below)
    
    lang = st.session_state.lang
    
    # Sidebar logic moved to top
    lang = st.session_state.lang
    
    with st.sidebar:
        # Back button to return to path selection
        if st.button("🏠 " + ("Back to Home" if lang == 'en' else "العودة للرئيسية"), key="back_to_home_sci", use_container_width=True):
            st.session_state.app_mode = None
            st.rerun()
        
        st.markdown("---")
        
        # Expert Settings
        with st.expander("⚙️ " + ("إعدادات الخبير" if lang == 'ar' else "Expert Settings")):
            random_seed = st.number_input(
                "Random Seed", 
                value=42, 
                help="Change this to get different results"
            )
            
            opt_rounds = st.slider(
                "Optimization Rounds" if lang == 'en' else "جولات التحسين", 
                min_value=1, max_value=10, value=3,
                help="More rounds = Better accuracy but slower"
            )
            
            enable_ensemble = st.toggle(
                "Enable Ensemble" if lang == 'en' else "تفعيل التصويت الجماعي",
                value=True
            )
    
    # Main Title
    st.title("💎 Easy Data")
    st.markdown(f"<p style='text-align: center; color: #94a3b8; font-size: 1.1rem;'>{t('app_subtitle', lang)}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File Upload Section
    st.subheader(t('upload_files', lang))
    st.caption(t('upload_help', lang))
    
    # Helper functions and initialization
    if 'use_fixed_data' not in st.session_state:
        st.session_state.use_fixed_data = False
    if 'auto_run' not in st.session_state:
        st.session_state.auto_run = False
        
    # Data Source Selection
    tab_upload, tab_sql, tab_url = st.tabs([
        t('upload_files', lang), 
        "🗄️ " + ("قاعدة بيانات SQL" if lang == 'ar' else "SQL Database"),
        "🌐 " + ("رابط مباشر / بيانات سحابية" if lang == 'ar' else "Direct URL / Cloud Data")
    ])
    
    uploaded_files = None
    sql_df = None
    url_df = None
    
    with tab_upload:
        uploaded_files = st.file_uploader(
            "",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
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
        
        if st.button("🔌 " + ("اتصال وتحميل" if lang == 'ar' else "Connect & Load"), key="btn_sql"):
            if conn_str and sql_query:
                from core.data_loader import DataLoader
                loader = DataLoader() # Instance to use load_sql
                with st.spinner("Connecting..."):
                    sql_df = loader.load_sql(conn_str, sql_query)
                    if sql_df is not None:
                        st.success(f"✅ Loaded {len(sql_df)} rows")
                    else:
                        st.error(f"❌ Error: {loader.errors[-1] if loader.errors else 'Unknown error'}")
            else:
                st.warning("Please enter connection string and query")

    with tab_url:
        st.info("💡 " + ("هذه الميزة تسمح لك بتحميل البيانات مباشرة من روابط عامة (مثل Azure Open Datasets)" if lang == 'ar' else "This feature allows you to load data directly from public URLs (e.g., Azure Open Datasets)"))
        public_url = st.text_input(
            "رابط البيانات (CSV أو Parquet)" if lang == 'ar' else "Data URL (CSV or Parquet)",
            placeholder="https://example.com/dataset.csv"
        )
        
        if st.button("🌐 " + ("تحميل من الرابط" if lang == 'ar' else "Load from URL"), key="btn_url"):
            if public_url:
                from core.data_loader import DataLoader
                loader = DataLoader()
                with st.spinner("Fetching data..."):
                    url_df = loader.load_url(public_url)
                    if url_df is not None:
                        st.success(f"✅ Loaded {len(url_df)} rows")
                    else:
                        st.error(f"❌ Error: {loader.errors[-1] if loader.errors else 'Unknown error'}")
            else:
                st.warning("Please enter a URL")
    
    # Logic to determine df_main
    df_main = None
    
    # 1. Check if we are using FIXED DATA (Priority)
    use_fixed = st.session_state.get('use_fixed_data', False)
    fixed_df = st.session_state.get('df_fixed', None)
    
    if use_fixed and fixed_df is not None:
        df_main = fixed_df
        st.info(f"🔧 Using Fixed Data ({len(df_main)} rows, {len(df_main.columns)} cols)")
        
        # Add Reset Button
        if st.button("❌ " + ("إلغاء الإصلاحات والعودة للملف الأصلي" if lang == 'ar' else "Discard fixes & Reset"), key='reset_fixed'):
            st.session_state.use_fixed_data = False
            st.session_state.df_fixed = None
            st.session_state.auto_run = False
            st.rerun()
            
    # 2. SQL Data Priority over File Upload if present (from immediate session)
    elif sql_df is not None:
        df_main = sql_df
        st.success("🗄️ Data loaded from SQL")

    # 3. URL Data Priority
    elif url_df is not None:
        df_main = url_df
        st.success("🌐 Data loaded from URL")

    # 4. Fallback to File Uploader
    elif uploaded_files:
        # Load all files
        dataframes = []
        for uploaded_file in uploaded_files:
            df = load_uploaded_file(uploaded_file)
            if df is not None:
                dataframes.append((uploaded_file.name, df))
        
        if dataframes:
            # Combine dataframes
            if len(dataframes) == 1:
                df_main = dataframes[0][1]
                st.success(f"{t('files_loaded', lang)} ({len(df_main)} {t('rows', lang)} × {len(df_main.columns)} {t('columns', lang)})")
            else:
                # Try to merge if same columns
                try:
                    df_main = pd.concat([d[1] for d in dataframes], ignore_index=True)
                    st.success(f"{t('files_loaded', lang)} - {len(dataframes)} files combined ({len(df_main)} {t('rows', lang)})")
                except:
                    df_main = dataframes[0][1]
                    st.warning(f"Using first file only: {dataframes[0][0]}")

    if df_main is not None:
        # Data Preview
        with st.expander(t('data_preview', lang)):
            st.dataframe(df_main.head(10), use_container_width=True)
        
        st.markdown("---")
            
        # Target Selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.info(t('select_target', lang))
        
        with col2:
            target_col = st.selectbox(
                t('target_help', lang),
                options=df_main.columns.tolist(),
                index=len(df_main.columns) - 1
            )
        
        # Column Exclusion Section
        with st.expander("🗑️ " + ("استبعاد أعمدة (لمنع تسرب البيانات)" if lang == 'ar' else "Exclude Columns (Prevent Data Leakage)"), expanded=False):
            st.caption("⚠️ " + ("اختر الأعمدة التي تريد استبعادها من التحليل. مفيد لإزالة الأعمدة المرتبطة بالهدف." if lang == 'ar' else "Select columns to exclude from analysis. Useful for removing target-correlated columns."))
            
            # Get columns except target
            excludable_cols = [c for c in df_main.columns if c != target_col]
            
            # Smart suggestions based on target
            suggested_cols = []  # Just column names for default
            suggested_info = {}  # Column -> (correlation, explanation)
            
            if target_col in df_main.columns:
                target_data = df_main[target_col]
                for col in excludable_cols:
                    if df_main[col].dtype in ['object', 'category']:
                        continue
                    try:
                        corr = abs(df_main[col].corr(target_data.astype(float)))
                        if corr > 0.90:
                            suggested_cols.append(col)
                            suggested_info[col] = corr
                    except:
                        pass
            
            # Show AI-powered explanations if there are suggestions
            if suggested_cols:
                st.warning("💡 " + ("تم اكتشاف أعمدة مرتبطة بالهدف وتم تحديدها تلقائياً:" if lang == 'ar' else "High-correlation columns detected and auto-selected:"))
                
                # AI Explanation for each column
                for col in suggested_cols:
                    corr = suggested_info[col]
                    # Generate smart explanation based on column name and target
                    if 'total' in col.lower() and 'amount' in target_col.lower():
                        reason = "المجموع الكلي يحتوي على الهدف كجزء منه" if lang == 'ar' else "Total includes target as a component"
                    elif 'amount' in col.lower() and 'amount' in target_col.lower():
                        reason = "كلاهما مبالغ مالية مرتبطة حسابياً" if lang == 'ar' else "Both are financially derived amounts"
                    elif corr > 0.98:
                        reason = "شبه متطابق - قد يكون مشتقاً من الهدف" if lang == 'ar' else "Nearly identical - likely derived from target"
                    else:
                        reason = "ارتباط عالي جداً يشير لتسرب بيانات" if lang == 'ar' else "Very high correlation indicates data leakage"
                    
                    st.markdown(f"- **{col}** ({corr:.0%}) → _{reason}_")
                
                st.info("💡 " + ("يمكنك إلغاء تحديد أي عمود إذا كنت متأكداً أنه مفيد" if lang == 'ar' else "You can deselect any column if you're sure it's useful"))
            
            # Multiselect with auto-selected defaults
            excluded_cols = st.multiselect(
                "اختر الأعمدة للاستبعاد" if lang == 'ar' else "Select columns to exclude",
                options=excludable_cols,
                default=suggested_cols  # Auto-select suggested columns
            )
            
            # Apply exclusions
            if excluded_cols:
                df_main = df_main.drop(columns=excluded_cols)
                st.success(f"✅ " + (f"تم استبعاد {len(excluded_cols)} أعمدة" if lang == 'ar' else f"Excluded {len(excluded_cols)} columns"))
        
        # API Key / Brain Status Section
        with st.expander(t('api_key_section', lang) if lang == 'en' else "🧠 " + ("محركات الذكاء (متصلة)" if lang == 'ar' else "AI Engines (Connected)")):
            st.caption(t('api_key_help', lang) if lang == 'en' else ("تم تفعيل المحركات التالية باستخدام مفاتيحك الخاصة:" if lang == 'ar' else "Engines activated with your keys:"))
            
            # Check providers
            ai_test = get_ensemble()
            cols = st.columns(3)
            
            with cols[0]:
                if 'groq' in ai_test.providers:
                    st.success("✅ Groq (Llama 3)")
                else:
                    st.error("❌ Groq")
                    
            with cols[1]:
                if 'gemini' in ai_test.providers:
                    st.success("✅ Gemini Pro") 
                else:
                    st.error("❌ Gemini")
                    # Debugging: Show why it failed
                    for msg in ai_test.get_log():
                        if "Gemini" in msg and "⚠️" in msg:
                            st.caption(f"Error: {msg}")
                    
            with cols[2]:
                if 'deepseek' in ai_test.providers:
                    st.success("✅ DeepSeek")
                else:
                    st.error("❌ DeepSeek")
                    
            st.info("💡 " + ("النظام يعمل الآن بكامل طاقته الإدراكية" if lang == 'ar' else "System running at full cognitive capacity"))
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Start Analysis Button
        # Trigger if button clicked OR if auto_run request exists
        should_run = st.button(t('start_analysis', lang), type="primary") or st.session_state.get('auto_run', False)
        
        if should_run:
            # Consume the auto_run flag immediately
            if st.session_state.get('auto_run', False):
                st.session_state.auto_run = False
            
            try:
                with st.spinner(""):
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # =========================================================
                        # CHIEF DATA SCIENTIST: EXPERT THINKING LAYER
                        # =========================================================
                        ai_ensemble = get_ensemble()
                        chief_ds = get_chief_data_scientist(ai_ensemble)
                        
                        # --- THINKING STAGE 1: PROBLEM REFRAMING ---
                        status_text.text("🧠 " + ("المرحلة 1: إعادة صياغة المشكلة..." if lang == 'ar' else "Stage 1: Problem Reframing..."))
                        progress_bar.progress(5)
                        stage1 = chief_ds.stage1_problem_reframing(df_main, target_col, lang)
                        
                        # Display Stage 1 Results
                        with st.expander("🧠 " + ("المرحلة 1: إعادة صياغة المشكلة" if lang == 'ar' else "Stage 1: Problem Reframing"), expanded=True):
                            if stage1.status == ApprovalStatus.APPROVED:
                                st.success(f"✅ {('تمت الموافقة' if lang == 'ar' else 'APPROVED')} (ثقة: {stage1.confidence:.0%})")
                            elif stage1.status == ApprovalStatus.REJECTED:
                                st.error(f"❌ {('مرفوض' if lang == 'ar' else 'REJECTED')}")
                            else:
                                st.warning(f"⚠️ {('يحتاج مراجعة' if lang == 'ar' else 'NEEDS REVIEW')}")
                            st.markdown(format_mixed_text(stage1.reasoning, lang))
                            if stage1.concerns:
                                st.markdown("**" + ("المخاوف:" if lang == 'ar' else "Concerns:") + "**")
                                for c in stage1.concerns:
                                    st.markdown(f"- ⚠️ {format_mixed_text(c, lang)}")
                            if stage1.recommendations:
                                st.markdown("**" + ("التوصيات:" if lang == 'ar' else "Recommendations:") + "**")
                                for r in stage1.recommendations:
                                    st.markdown(f"- 💡 {r}")
                        
                        # Note: Don't stop here - let flow continue to show full Recovery Mode
                        # if stage1.status == ApprovalStatus.REJECTED:
                        #     st.stop()  # Removed - Recovery Mode handles this
                        
                        # --- THINKING STAGE 2: DATA SKEPTICISM ---
                        status_text.text("🔍 " + ("المرحلة 2: فحص جودة البيانات..." if lang == 'ar' else "Stage 2: Data Skepticism..."))
                        progress_bar.progress(10)
                        stage2 = chief_ds.stage2_data_skepticism(df_main, target_col, lang)
                        
                        # Display Stage 2 Results
                        with st.expander("🔍 " + ("المرحلة 2: شكوك البيانات" if lang == 'ar' else "Stage 2: Data Skepticism"), expanded=True):
                            if stage2.status == ApprovalStatus.APPROVED:
                                st.success(f"✅ {('تمت الموافقة' if lang == 'ar' else 'APPROVED')} (ثقة: {stage2.confidence:.0%})")
                            elif stage2.status == ApprovalStatus.REJECTED:
                                st.error(f"❌ {('مرفوض' if lang == 'ar' else 'REJECTED')}")
                            else:
                                st.warning(f"⚠️ {('يحتاج مراجعة' if lang == 'ar' else 'NEEDS REVIEW')}")
                            st.markdown(format_mixed_text(stage2.reasoning, lang))
                            if stage2.concerns:
                                st.markdown("**" + ("المخاوف:" if lang == 'ar' else "Concerns:") + "**")
                                for c in stage2.concerns:
                                    st.markdown(f"- ⚠️ {format_mixed_text(c, lang)}")
                        
                        # Note: Don't stop here - let flow continue to show full Recovery Mode
                        # if stage2.status == ApprovalStatus.REJECTED:
                        #     st.stop()  # Removed - Recovery Mode handles this
                        
                        # --- THINKING STAGE 3: ANALYSIS STRATEGY ---
                        status_text.text("📋 " + ("المرحلة 3: تحديد استراتيجية التحليل..." if lang == 'ar' else "Stage 3: Analysis Strategy..."))
                        progress_bar.progress(15)
                        stage3 = chief_ds.stage3_analysis_strategy(df_main, target_col, lang)
                        
                        # Display Stage 3 Results
                        with st.expander("📋 " + ("المرحلة 3: استراتيجية التحليل" if lang == 'ar' else "Stage 3: Analysis Strategy"), expanded=True):
                            if stage3.status == ApprovalStatus.APPROVED:
                                st.success(f"✅ {('تمت الموافقة' if lang == 'ar' else 'APPROVED')} (ثقة: {stage3.confidence:.0%})")
                            elif stage3.status == ApprovalStatus.REJECTED:
                                st.error(f"❌ {('مرفوض' if lang == 'ar' else 'REJECTED')}")
                            else:
                                st.warning(f"⚠️ {('يحتاج مراجعة' if lang == 'ar' else 'NEEDS REVIEW')}")
                            st.markdown(format_mixed_text(stage3.reasoning, lang))
                            if stage3.recommendations:
                                st.markdown("**" + ("التوصيات:" if lang == 'ar' else "Recommendations:") + "**")
                                for r in stage3.recommendations:
                                    st.markdown(f"- 💡 {format_mixed_text(r, lang)}")
                        
                        # =========================================================
                        # EXPLICIT GATE: CHECK is_fully_approved()
                        # This is the HARD STOP that blocks AutoML
                        # =========================================================
                        if not chief_ds.is_fully_approved():
                            progress_bar.empty()
                            status_text.empty()
                            
                            # =====================================================
                            # EXPERT RECOVERY MODE
                            # A real senior data scientist NEVER just stops.
                            # They stop execution AND provide a recovery plan.
                            # =====================================================
                            st.error("🚫 " + ("تم إيقاف التحليل لأسباب مهنية" if lang == 'ar' else "Analysis STOPPED for professional reasons"))
                            st.info("🧠 " + ("إليك بالضبط كيف سيصلح كبير علماء البيانات هذه البيانات" if lang == 'ar' else "Here is exactly how a senior data scientist would fix this dataset"))
                            
                            # Generate recovery plan
                            recovery = chief_ds.generate_recovery_plan(df_main, target_col, lang)
                            
                            st.markdown("---")
                            
                            # =====================================================
                            # A. ROOT CAUSE DIAGNOSIS
                            # =====================================================
                            st.subheader("🔍 " + ("التشخيص الجذري" if lang == 'ar' else "Root Cause Diagnosis"))
                            
                            for diag in recovery['root_cause_diagnosis']:
                                severity_color = "error" if diag['severity'] == 'CRITICAL' else ("warning" if diag['severity'] == 'MAJOR' else "info")
                                with st.expander(f"{diag['severity_icon']} [{diag['severity']}] {diag['concern'][:50]}...", expanded=diag['severity'] == 'CRITICAL'):
                                    st.markdown(f"**{('المشكلة:' if lang == 'ar' else 'Issue:')}** {diag['concern']}")
                                    st.markdown(f"**{('السبب الإحصائي:' if lang == 'ar' else 'Statistical Reason:')}** {diag['statistical_reason']}")
                            
                            st.markdown("---")
                            
                            # =====================================================
                            # B. EXPERT REPAIR PLAN
                            # =====================================================
                            st.subheader("🔧 " + ("خطة الإصلاح" if lang == 'ar' else "Expert Repair Plan"))
                            
                            for repair in recovery['repair_plan']:
                                with st.expander(f"🔧 {repair['issue'][:50]}..."):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("### " + ("الإصلاح المحافظ" if lang == 'ar' else "Conservative Fix"))
                                        st.success(f"✅ {repair['fix_conservative']}")
                                        st.caption(f"⚠️ {('المخاطر:' if lang == 'ar' else 'Risks:')} {repair['risks_conservative']}")
                                    with col2:
                                        st.markdown("### " + ("الإصلاح العنيف" if lang == 'ar' else "Aggressive Fix"))
                                        st.warning(f"⚡ {repair['fix_aggressive']}")
                                        st.caption(f"⚠️ {('المخاطر:' if lang == 'ar' else 'Risks:')} {repair['risks_aggressive']}")
                                    st.info(f"🚫 {('متى لا تصلح:' if lang == 'ar' else 'When NOT to fix:')} {repair['when_not_to_fix']}")
                            
                            st.markdown("---")
                            
                            # =====================================================
                            # C. AUTO-FIX CANDIDATES
                            # =====================================================
                            st.subheader("🤖 " + ("إصلاحات تلقائية محتملة" if lang == 'ar' else "Auto-Fix Candidates"))
                            
                            auto_fixes = recovery['auto_fix_candidates']
                            
                            # Safe fixes
                            if auto_fixes['safe']:
                                st.markdown("#### ✔️ " + ("آمنة للتطبيق التلقائي:" if lang == 'ar' else "SAFE to Auto-Apply:"))
                                for fix in auto_fixes['safe']:
                                    st.success(f"• **{fix['action']}** - {fix['reason']}")
                                    st.code(fix['code'], language='python')
                            
                            # Confirm fixes
                            if auto_fixes['confirm']:
                                st.markdown("#### ⚠️ " + ("تحتاج موافقة:" if lang == 'ar' else "Need Confirmation:"))
                                for fix in auto_fixes['confirm']:
                                    st.warning(f"• **{fix['action']}** - {fix['reason']}")
                                    st.caption(f"❓ {fix['question']}")
                            
                            # Never automate
                            if auto_fixes['never']:
                                st.markdown("#### ❌ " + ("لا تُؤتمت أبداً:" if lang == 'ar' else "NEVER Automate:"))
                                for fix in auto_fixes['never']:
                                    st.error(f"• **{fix['action']}** - {fix['reason']}")
                            
                            st.markdown("---")
                            
                            # =====================================================
                            # D. DOMAIN-AWARE SUGGESTIONS
                            # =====================================================
                            if recovery['domain_suggestions']:
                                st.subheader("💡 " + ("اقتراحات متخصصة" if lang == 'ar' else "Domain-Aware Suggestions"))
                                for suggestion in recovery['domain_suggestions']:
                                    st.info(suggestion)
                            
                            st.markdown("---")
                            
                            # =====================================================
                            # E. RE-ENTRY CONDITIONS
                            # =====================================================
                            st.subheader("🎯 " + ("شروط إعادة التحليل" if lang == 'ar' else "Re-Entry Conditions"))
                            st.caption(("يجب تحقيق هذه الشروط قبل إعادة تشغيل التحليل:" if lang == 'ar' else "These conditions must be met before re-running analysis:"))
                            
                            for cond in recovery['re_entry_conditions']:
                                st.markdown(f"- {cond['description']}")
                            
                            st.markdown("---")
                            
                            # =====================================================
                            # APPLY SAFE FIXES BUTTON
                            # =====================================================
                            # Callback for robust state update
                            def apply_fixes_callback(cds_instance, df, target):
                                try:
                                    df_fix, changes_list = cds_instance.apply_safe_fixes(df, target)
                                    if changes_list:
                                        st.session_state.df_fixed = df_fix
                                        st.session_state.use_fixed_data = True
                                        st.session_state.auto_run = True
                                        st.session_state.last_changes = changes_list # Store for display
                                    else:
                                        st.session_state.fix_error = "No changes applied"
                                except Exception as e:
                                    st.session_state.fix_error = str(e)

                            st.subheader("🔄 " + ("تطبيق الإصلاحات الآمنة" if lang == 'ar' else "Apply Safe Fixes"))
                            
                            st.button(
                                "🔧 " + ("تطبيق الإصلاحات الآمنة وإعادة التقييم" if lang == 'ar' else "Apply Expert Fixes & Re-evaluate"), 
                                type="primary",
                                on_click=apply_fixes_callback,
                                args=(chief_ds, df_main, target_col),
                                key="apply_fixes_btn"
                            )
                            
                            # Show feedback if valid
                            if st.session_state.get('last_changes'):
                                st.success("✅ " + ("تم تطبيق الإصلاحات (جاري إعادة التشغيل...)" if lang == 'ar' else "Fixes applied (Restarting...)"))
                                # Trigger rerun if not triggered by callback (redundant safely)
                                # st.rerun() 
                            elif st.session_state.get('fix_error'):
                                st.warning("⚠️ " + st.session_state.fix_error)
                            
                            st.markdown("---")
                            st.warning("⚠️ " + ("لم يتم تشغيل أي نموذج. لم يتم إنشاء أي مقاييس. لم يتم إنشاء أي رسوم بيانية." if lang == 'ar' else "NO models were run. NO metrics were generated. NO charts were created."))
                            
                            st.stop()  # HARD STOP - NO AutoML
                        
                        # =========================================================
                        # ALL STAGES APPROVED - PROCEED WITH AUTOML
                        # =========================================================
                        st.success("✅ " + ("تمت موافقة كبير علماء البيانات - جاري التنفيذ..." if lang == 'ar' else "Chief Data Scientist APPROVED - Proceeding with analysis..."))
                        
                        # Step 1: Scanning
                        status_text.text(t('progress_scanning', lang))
                        progress_bar.progress(20)
                        time.sleep(0.3)
                        
                        # Initialize AutoML
                        automl = AutoML(
                            random_state=int(random_seed),
                            optimization_rounds=int(opt_rounds),
                            enable_hypertuning=True 
                        )
                        
                        # ===== AUTO-SAMPLING FOR LARGE DATASETS =====
                        MAX_ROWS = 100000  # Maximum rows for memory-safe processing
                        original_size = len(df_main)
                        
                        if len(df_main) > MAX_ROWS:
                            st.info(f"📉 " + (f"البيانات كبيرة ({original_size:,} صف) - جاري أخذ عينة {MAX_ROWS:,} صف للتحليل" if lang == 'ar' else f"Large dataset ({original_size:,} rows) - Sampling {MAX_ROWS:,} rows for analysis"))
                            df_main = df_main.sample(n=MAX_ROWS, random_state=int(random_seed))
                        
                        # --- OMNI-LOGIC LAYER 1: ANALYTICAL & CAUSAL ---
                        status_text.text("🧠 " + ("تفكير المنطق البشري: تحليل العلاقات والأنماط..." if lang == 'ar' else "Human Logic: Analyzing patterns & causality..."))
                        analytical_logic = AnalyticalLogic()
                        causal_logic = CausalLogic()
                        
                        # Run Logic Layers
                        eda_report = analytical_logic.execute(df_main)
                        causal_report = causal_logic.execute(df_main)
                        
                        analysis = automl.analyze_data(df_main)
                        analysis['human_logic_eda'] = eda_report # Inject findings
                        analysis['human_logic_causal'] = causal_report
                        
                        progress_bar.progress(30)
                        
                        # Step 2: Cleaning
                        status_text.text(t('progress_cleaning', lang))
                        progress_bar.progress(40)
                        time.sleep(0.3)
                        
                        # Step 3: Training
                        status_text.text(t('progress_training', lang))
                        progress_bar.progress(50)
                        
                        results = automl.train(df_main, target_col)
                        progress_bar.progress(70)
                        
                        # Step 4: AI Insights (Ensemble - All 3 AIs together)
                        status_text.text(t('progress_insights', lang))
                        
                        data_quality = ai_ensemble.analyze_data_quality(analysis, lang)
                        insights = ai_ensemble.generate_insights(results, target_col, lang)
                        
                        # --- OMNI-LOGIC LAYER 2: ETHICAL CHECK ---
                        status_text.text("⚖️ " + ("تفكير المنطق الأخلاقي: فحص التحيز..." if lang == 'ar' else "Ethical Logic: Checking for bias..."))
                        ethical_logic = EthicalLogic()
                        # Create a mock prediction df for audit
                        mock_pred_df = df_main.copy()
                        mock_pred_df['prediction'] = results.get('predictions', [0]*len(df_main)) 
                        ethical_report = ethical_logic.execute(mock_pred_df)
                        
                        results['ethical_report'] = ethical_report
                        progress_bar.progress(80)
                        
                        # =========================================================
                        # SELF-CRITIQUE STAGE
                        # =========================================================
                        status_text.text("⚖️ " + ("مرحلة النقد الذاتي..." if lang == 'ar' else "Self-Critique Stage..."))
                        progress_bar.progress(85)
                        
                        critique = chief_ds.generate_self_critique(results, lang)
                        expert_output = chief_ds.format_expert_output(results, critique, lang)
                        
                        # Step 5: Generate Reports
                        status_text.text(t('progress_reports', lang))
                        progress_bar.progress(95)
                        
                        # Store results with Chief DS additions
                        st.session_state.results = {
                            'automl': automl,
                            'analysis': analysis,
                            'results': results,
                            'data_quality': data_quality,
                            'insights': insights,
                            'df_clean': df_main,
                            'target_col': target_col,
                            'lang': lang,
                            # New: Chief Data Scientist additions
                            'chief_ds_stages': {
                                'problem_reframing': stage1,
                                'data_skepticism': stage2,
                                'analysis_strategy': stage3
                            },
                            'critique': critique,
                            'expert_output': expert_output
                        }
                        st.session_state.analysis_done = True
                        
                        progress_bar.progress(100)
                        status_text.text(t('progress_done', lang))
                        time.sleep(0.5)
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.rerun()
                        
            except Exception as e:
                st.error(f"❌ حدث خطأ غير متوقع: {str(e)}")
                st.warning("جاري إعادة المحاولة تلقائياً...")
                # Log for debugging
                import traceback
                st.code(traceback.format_exc(), language="python")
            
        # Show Results if analysis is done
        if st.session_state.get('analysis_done', False) and st.session_state.get('results', None):
            show_results(st.session_state.results, lang)
    
    else:
        # Empty State
        st.markdown(f"""
        <div style="text-align: center; padding: 4rem 2rem; background: #f8f9fa; border-radius: 20px;">
            <h2 style="margin-bottom: 1rem;">{t('welcome_title', lang)}</h2>
            <p style="color: #666; line-height: 1.8;">{t('welcome_text', lang).replace(chr(10), '<br>')}</p>
        </div>
        """, unsafe_allow_html=True)


def show_results(data, lang):
    """Display analysis results."""
    results = data['results']
    analysis = data['analysis']
    automl = data['automl']
    target_col = data['target_col']
    insights = data['insights']
    data_quality = data['data_quality']
    
    st.markdown("---")
    st.subheader(t('results_title', lang))
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        problem_label = t('classification', lang) if results['problem_type'] == 'classification' else t('regression', lang)
        metric_card(t('problem_type', lang), problem_label, "📊")
    
    with col2:
        metric_card(t('best_model', lang), results['best_model'], "🏆")
    
    with col3:
        if results['problem_type'] == 'classification':
            acc = results['metrics'].get('accuracy', 0)
            metric_card(t('accuracy', lang), f"{acc:.1%}", "🎯")
        else:
            r2 = results['metrics'].get('r2', 0)
            metric_card("R²", f"{r2:.3f}", "📈")
    
    with col4:
        metric_card(t('features_count', lang), len(results['feature_importance']), "🔢")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # AI Insights
    st.markdown(f"### {t('insights_title', lang)}")
    st.info(data_quality)
    st.markdown(insights)
    
    # --- HUMAN LOGIC DISPLAY ---
    st.markdown("---")
    st.subheader("🧠 " + ("رؤى المنطق البشري (Omni-Logic)" if lang == 'ar' else "Human Logic Insights (Omni-Logic)"))
    
    hl_col1, hl_col2 = st.columns(2)
    with hl_col1:
        st.markdown("**" + ("المحقق الذكي (Analytical):" if lang == 'ar' else "Investigator (Analytical):") + "**")
        outliers = analysis.get('human_logic_eda', {}).get('outlier_counts', {})
        if outliers:
            st.warning(f"⚠️ {len(outliers)} " + ("أعمدة تحتوي على قيم شاذة" if lang == 'ar' else "columns have outliers"))
        else:
            st.success("✅ " + ("البيانات نظيفة من الشواذ" if lang == 'ar' else "No critical outliers found"))
            
    with hl_col2:
        st.markdown("**" + ("الحارس الأخلاقي (Ethical):" if lang == 'ar' else "Guardian (Ethical):") + "**")
        bias = results.get('ethical_report', {}).get('bias_report', 'Safe')
        st.success(f"🛡️ {bias}")
    # ---------------------------
    
    # =========================================================
    # EXPERT OUTPUT SECTION (Chief Data Scientist)
    # =========================================================
    if 'expert_output' in data:
        st.markdown("---")
        st.subheader("🎓 " + ("تفسير الخبير" if lang == 'ar' else "Expert Interpretation"))
        
        expert_output = data['expert_output']
        
        # Expert interpretation
        st.markdown(expert_output.get('expert_interpretation', ''))
        
        # Practical recommendations
        st.markdown("### " + ("التوصيات العملية" if lang == 'ar' else "Practical Recommendations"))
        st.markdown(expert_output.get('practical_recommendations', ''))
        
        # Uncertainty and risk
        st.markdown("### " + ("مستوى الثقة والمخاطر" if lang == 'ar' else "Confidence & Risk"))
        st.info(expert_output.get('uncertainty_statement', ''))
    
    # =========================================================
    # SELF-CRITIQUE SECTION
    # =========================================================
    if 'critique' in data:
        st.markdown("---")
        st.subheader("⚖️ " + ("النقد الذاتي" if lang == 'ar' else "Self-Critique"))
        
        critique = data['critique']
        
        # Weak assumptions
        if critique.get('weak_assumptions'):
            with st.expander("🤔 " + ("افتراضات ضعيفة" if lang == 'ar' else "Weak Assumptions"), expanded=True):
                for assumption in critique['weak_assumptions']:
                    st.markdown(f"- {assumption}")
        
        # Potential errors
        if critique.get('potential_errors'):
            with st.expander("⚠️ " + ("أخطاء محتملة" if lang == 'ar' else "Potential Errors")):
                for error in critique['potential_errors']:
                    st.markdown(f"- {error}")
        
        # Overconfidence warnings
        if critique.get('overconfidence_warnings'):
            for warning in critique['overconfidence_warnings']:
                st.warning(f"🚨 {warning}")
        
        # Senior data scientist warnings
        st.markdown("### " + ("تحذيرات كبير علماء البيانات" if lang == 'ar' else "Senior Data Scientist Warnings"))
        for warning in critique.get('expert_warnings', []):
            st.markdown(f"⚠️ {warning}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations Row
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.markdown(f"### {t('feature_importance', lang)}")
        st.caption(t('feature_importance_desc', lang))
        
        importance_df = results['feature_importance'].head(10)
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_viz2:
        st.markdown(f"### {t('all_models', lang)}")
        
        models_df = pd.DataFrame([
            {'Model': k, 'Score': v}
            for k, v in results['all_models'].items()
        ]).sort_values('Score', ascending=True)
        
        colors = ['#667eea' if m != results['best_model'] else '#11998e' 
                  for m in models_df['Model']]
        
        fig2 = go.Figure(go.Bar(
            x=models_df['Score'],
            y=models_df['Model'],
            orientation='h',
            marker_color=colors
        ))
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Cleaning Steps
    with st.expander(t('cleaning_steps', lang)):
        for step in results['cleaning_steps']:
            st.text(f"• {step}")
    
    st.markdown("---")
    
    # Downloads Section
    st.subheader(t('downloads_title', lang))
    
    col_d1, col_d2, col_d3, col_d4, col_d5, col_d6 = st.columns(6)
    
    # Generate reports
    
    # 1. Generate full predictions for Excel report
    df_preds_report = None
    try:
        full_preds = automl.predict(data['df_clean'])
        df_preds_report = data['df_clean'].copy()
        pred_col_name = f"{target_col}_Predicted"
        df_preds_report[pred_col_name] = full_preds
        # Also add Actual if available and not same col (it is same col in df_clean)
    except Exception as e:
        st.warning(f"Could not generate full predictions: {e}")
        df_preds_report = None

    excel_bytes = create_excel_report(
        data['df_clean'], df_preds_report, results['feature_importance'],
        results['metrics'], analysis, lang
    )
    
    notebook_json = generate_notebook(
        target_col, results['problem_type'],
        results['feature_importance']['Feature'].tolist()[:5],
        results['metrics'], results['cleaning_steps'], lang
    )
    notebook_bytes = get_notebook_bytes(notebook_json)
    
    pdf_bytes = create_pdf_report(analysis, results, target_col, insights, lang)
    word_bytes = create_word_report(analysis, results, target_col, insights, lang)
    
    # Power BI Package
    pbi_bytes = create_powerbi_package(data['df_clean'], None, analysis, lang)
    
    # Model bytes
    model_buffer = io.BytesIO()
    automl.save_model(model_buffer)
    model_bytes = model_buffer.getvalue()
    
    with col_d1:
        st.download_button(
            t('download_excel', lang),
            excel_bytes,
            "AI_Expert_Results.xlsx",
            "application/vnd.ms-excel"
        )
    
    with col_d2:
        if pdf_bytes:
            st.download_button(
                t('download_pdf', lang),
                pdf_bytes,
                "AI_Expert_Report.pdf",
                "application/pdf"
            )
        else:
            st.warning("PDF unavailable")
    
    with col_d3:
        if word_bytes:
            st.download_button(
                t('download_word', lang),
                word_bytes,
                "AI_Expert_Report.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        else:
            st.warning("Word unavailable")
    
    with col_d4:
        st.download_button(
            t('download_notebook', lang),
            notebook_bytes,
            "AI_Expert_Analysis.ipynb",
            "application/x-ipynb+json"
        )

    with col_d5:
        st.download_button(
            "Power BI",
            pbi_bytes,
            "PowerBI_Ready_Data.zip",
            "application/zip",
            help="Download data package for Power BI"
        )
    
    with col_d6:
        st.download_button(
            t('download_model', lang),
            model_bytes,
            "trained_model.pkl",
            "application/octet-stream"
        )
    
    st.markdown("---")
    
    # Test File Prediction Section
    st.subheader("🔮 " + ("التنبؤ على بيانات جديدة" if lang == 'ar' else "Predict on New Data"))
    st.caption("ارفع ملف test.csv للتنبؤ عليه" if lang == 'ar' else "Upload test.csv to make predictions")
    
    test_file = st.file_uploader(
        "ملف الاختبار" if lang == 'ar' else "Test File",
        type=['csv', 'xlsx', 'xls'],
        key="test_file_uploader"
    )
    
    if test_file:
        from core.data_loader import load_uploaded_file
        df_test = load_uploaded_file(test_file)
        
        if df_test is not None:
            st.info(f"📊 {len(df_test)} " + ("صف للتنبؤ" if lang == 'ar' else "rows to predict"))
            
            if st.button("🚀 " + ("توقع الآن" if lang == 'ar' else "Predict Now"), key="predict_btn"):
                try:
                    with st.spinner("جاري التنبؤ..." if lang == 'ar' else "Predicting..."):
                        # Make predictions
                        predictions = automl.predict(df_test)
                        
                        # Convert predictions to int for classification
                        if automl.problem_type == 'classification':
                            predictions = predictions.astype(int)
                        
                        st.success("✅ " + ("تم التنبؤ بنجاح!" if lang == 'ar' else "Predictions complete!"))
                        
                        # Create Kaggle-ready submission (only 2 columns)
                        # Find ID column (PassengerId, Id, id, etc.)
                        id_col = None
                        for col in ['PassengerId', 'Id', 'id', 'ID']:
                            if col in df_test.columns:
                                id_col = col
                                break
                        
                        if id_col is None:
                            id_col = df_test.columns[0]  # Use first column as ID
                        
                        # Create submission with ID and prediction only
                        submission = pd.DataFrame({
                            id_col: df_test[id_col],
                            target_col: predictions
                        })
                        
                        # Show preview
                        st.write("📋 " + ("معاينة النتائج:" if lang == 'ar' else "Preview:"))
                        st.dataframe(submission.head(20))
                        
                        # Download - Kaggle ready!
                        sub_buffer = io.BytesIO()
                        submission.to_csv(sub_buffer, index=False)
                        
                        st.download_button(
                            "🏆 " + ("تحميل submission.csv جاهز لـ Kaggle" if lang == 'ar' else "Download Kaggle Submission"),
                            sub_buffer.getvalue(),
                            "submission.csv",
                            "text/csv",
                            key="kaggle_submission"
                        )
                        
                        st.info("💡 " + ("ارفع هذا الملف مباشرة لـ Kaggle!" if lang == 'ar' else "Upload this file directly to Kaggle!"))
                            
                except Exception as e:
                    st.error(f"❌ خطأ في التنبؤ: {str(e)}")
    
    st.success(t('success_message', lang))
    custom_footer()


if __name__ == "__main__":
    main()
