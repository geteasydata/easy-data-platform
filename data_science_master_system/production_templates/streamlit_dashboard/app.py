import streamlit as st
import pandas as pd
import time
import plotly.express as px
from automl import AutoMLExpert
from notebook_gen import generate_notebook
import io

# Page Config (Cleaner, Simple)
st.set_page_config(
    page_title="المحلل الآلي | AI Expert",
    page_icon="🤖",
    layout="wide"
)

# Initialize Expert
if 'expert' not in st.session_state:
    st.session_state.expert = AutoMLExpert()

# Custom CSS for "Clean" look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        height: 3em;
        font-size: 1.2em;
        border-radius: 10px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Helper for Metric Cards
def metric_card(label, value, delta=None):
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin:0; font-size:1em; color:#7f8c8d;">{label}</h3>
        <h2 style="margin:0; font-size:2em; color:#2c3e50;">{value}</h2>
        {f'<div style="color:green">▲ {delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

# --- APP LAYOUT ---

st.title("🤖 محلل البيانات الخبير (AutoML Expert)")
st.markdown("##### أعطني بياناتك، وسأقوم بتحليلها وتنظيفها وبناء نموذج ذكي لها.. بضغطة زر وحدة.")

# 1. FILE UPLOAD SECTION
col_file1, col_file2 = st.columns(2)
with col_file1:
    train_file = st.file_uploader("📂 1. ملف التدريب (Train Data)", type=['csv', 'xlsx'])
with col_file2:
    test_file = st.file_uploader("📂 2. ملف الاختبار (Test Data - اختياري)", type=['csv', 'xlsx'])

if train_file:
    # Load Train Data
    try:
        if train_file.name.endswith('.csv'):
            df = pd.read_csv(train_file)
        else:
            df = pd.read_excel(train_file)
            
        st.success(f"✅ تم تحميل ملف التدريب بنجاح! ({len(df)} صف)")
        
        # Load Test Data (if exists)
        df_test = None
        if test_file:
            if test_file.name.endswith('.csv'):
                df_test = pd.read_csv(test_file)
            else:
                df_test = pd.read_excel(test_file)
            st.success(f"✅ تم تحميل ملف الاختبار بنجاح! ({len(df_test)} صف)")

        
        # Show small preview
        with st.expander("👀 نظرة سريعة على البيانات"):
            st.dataframe(df.head())
            
        st.markdown("---")
        
        # 2. TARGET SELECTION (What do you want to predict?)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("💡 ماذا تريد أن يتوقع الذكاء الاصطناعي؟ (الهدف)")
        with col2:
            target_col = st.selectbox("", df.columns, index=len(df.columns)-1)
            
        start_btn = st.button("🚀 ابدأ التحليل الكامل (Run Analysis)")
        
        if start_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Scanning
            status_text.text("🧐 جاري فحص البيانات...")
            time.sleep(1)
            progress_bar.progress(25)
            
            # Step 2: Cleaning
            status_text.text("🧹 جاري تنظيف الأخطاء والبيانات المفقودة...")
            time.sleep(1)
            progress_bar.progress(50)
            
            # Step 3: Training
            status_text.text("🧠 جاري تدريب الذكاء الاصطناعي واختبار أفضل النماذج...")
            
            # RUN EXPERT LOGIC
            report, df_result = st.session_state.expert.generate_report(df, target_col, test_df=df_test)
            
            progress_bar.progress(100)
            status_text.text("✅ انتهى التحليل!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            # --- RESULTS SECTION ---
            st.markdown("## 📊 تقرير الخبير (Expert Report)")
            
            # Key Metrics Row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("نوع المشكلة", "توقع رقم" if report['model_type'] == "Regression" else "تصنيف")
            with c2:
                # Show accuracy or R2 if available
                if report.get('metrics'):
                    key_metric = list(report['metrics'].keys())[0]
                    val = report['metrics'][key_metric]
                    metric_card(f"الدقة ({key_metric})", val)
                else:
                     metric_card("الحالة", "تم التوقع بنجاح ✅")
            with c3:
                metric_card("الصفوف المعالجة", len(df_result))
            with c4:
                metric_card("عدد المؤثرات", len(report['importance']))

            # --- AI BUSINESS INSIGHTS ---
            with st.container():
                st.info("🧠 **تحليل الذكاء الاستراتيجي (AI Strategic Insights):**")
                insights = st.session_state.expert.generate_business_insights(report, target_col)
                st.markdown(insights)
            
            # Clean/Insights Log
            with st.expander("🛠️ تفاصيل التنظيف (Technical Logs)"):
                st.markdown("### 🛠️ ماذا فعلنا في بياناتك؟")
                for step in report['cleaning_steps']:
                    st.text(f"• {step}")
            
            st.markdown("---")

            # Visuals Row
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.markdown("### 🔥 أهم العوامل المؤثرة")
                st.markdown("هذه الأعمدة هي التي تؤثر أكثر شيء على النتيجة المتوقعة")
                fig = px.bar(report['importance'].head(10), x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig, use_container_width=True)
                
            with col_viz2:
                st.markdown(f"### 📈 توزيع الهدف ({target_col})")
                # Only show target distribution if we have it (train set) or predicted
                plot_df = df_result if (target_col + '_PREDICTED') in df_result.columns else df
                plot_col = (target_col + '_PREDICTED') if (target_col + '_PREDICTED') in df_result.columns else target_col
                
                fig2 = px.histogram(plot_df, x=plot_col, color_discrete_sequence=['#2ecc71'])
                st.plotly_chart(fig2, use_container_width=True)
                
            # Download Section
            st.markdown("---")
            st.markdown("### 📥 استلام العمل")
            
            st.markdown("---")
            st.markdown("### 📥 استلام العمل")
            
            c_down1, c_down2, c_down3 = st.columns(3)
            
            # 1. Excel Download
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_result.to_excel(writer, index=False, sheet_name='Predictions')
                report['importance'].to_excel(writer, index=False, sheet_name='Feature Importance')
            
            with c_down1:
                st.download_button(
                    "📄 تحميل النتائج (Excel)",
                    buffer.getvalue(),
                    "results.xlsx",
                    "application/vnd.ms-excel"
                )
            
            # 2. Notebook Download
            notebook_content = generate_notebook(report, target_col)
            with c_down2:
                st.download_button(
                    "📓 تحميل الكود (Jupyter)",
                    notebook_content,
                    "analysis_notebook.ipynb",
                    "application/x-ipynb+json"
                )
                
            # 3. CSV (Backup)
            csv = df_result.to_csv(index=False).encode('utf-8')
            with c_down3:
                st.download_button(
                    "📊 تحميل النتائج (CSV)",
                    csv,
                    "results.csv",
                    "text/csv"
                )
            
            st.success("🎉 تم إنجاز المهمة! هل لديك ملف آخر؟")

    except Exception as e:
        st.error(f"حدث خطأ أثناء قراءة الملف: {e}")
        st.warning("تأكد أن الملف CSV أو Excel سليم.")

else:
    # Empty State with Animation or Image
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h2>👋 مرحباً بك</h2>
        <p>أنا مساعدك الذكي لعلوم البيانات.</p>
        <p>لا تحتاج لخبرة برمجية.. فقط ارفع ملفك وسأقوم بالباقي.</p>
    </div>
    """, unsafe_allow_html=True)
