# Data Science Hub - Instructions

## 🚀 Quick Start

### Option 1: Windows Launcher
Double-click `Start_App.bat` to automatically install dependencies and launch.

### Option 2: Manual Launch
```bash
cd DataAnalystProject
pip install -r requirements.txt
streamlit run main.py
```

---

## 📊 Two Main Paths

### 1. Data Analyst Path
For actionable insights and practical analysis.

**Features:**
- Data quality analysis (missing values, outliers, duplicates)
- Correlation detection and statistical summaries
- Multi-tool cleaning (Python, Excel Power Query, Power BI)
- Domain-specific insights (HR, Finance, Healthcare, etc.)
- Dashboard generation (Jupyter, Excel, Power BI)
- Report export (Word, PDF)

**Workflow:**
1. Upload or select sample data
2. Choose domain from sidebar
3. Run analysis in the "Analysis" tab
4. Clean data with your preferred tool
5. Generate insights and dashboards
6. Export reports

### 2. Data Scientist Path
For advanced ML and predictive models.

**Features:**
- Automatic model selection and training
- Feature engineering (interactions, binning, polynomial)
- Cross-validation and evaluation
- Model comparison and ranking
- Feature importance analysis
- SHAP explanations

**Workflow:**
1. Upload data
2. Select target variable
3. Engineer features (optional)
4. Train all models
5. Review results and feature importance

---

## 📁 Project Structure

```
DataAnalystProject/
├── main.py                 # Streamlit app entry point
├── config.py               # Domain and tool configurations
├── requirements.txt        # Python dependencies
├── Start_App.bat           # Windows launcher
│
├── paths/
│   ├── data_analyst/       # Analyst path modules
│   │   ├── analyzer.py     # Data quality analysis
│   │   ├── cleaner.py      # Multi-tool cleaning
│   │   ├── insights.py     # Domain insights
│   │   └── dashboard_gen.py # Dashboard generation
│   │
│   └── data_scientist/     # Scientist path modules
│       ├── ml_pipeline.py  # ML training pipeline
│       ├── feature_engineer.py # Feature engineering
│       ├── model_evaluator.py # Model evaluation
│       └── model_explainer.py # SHAP/interpretability
│
├── scripts/                # CLI scripts
│   ├── load_data.py
│   ├── clean_data.py
│   ├── analyze_data.py
│   ├── generate_dashboard.py
│   └── export_reports.py
│
├── sample_data/            # Example datasets
├── outputs/                # Generated outputs
├── templates/              # Dashboard templates
└── docs/                   # Documentation
```

---

## 🏢 Supported Domains

| Domain | Icon | Key Metrics |
|--------|------|-------------|
| HR | 👥 | Turnover, Satisfaction, Tenure |
| Finance | 💰 | Revenue, Profit, Cash Flow |
| Healthcare | 🏥 | Readmission, Length of Stay |
| Retail | 🛒 | Sales, Conversion, AOV |
| Marketing | 📢 | ROI, CTR, Conversions |
| Education | 🎓 | Graduation, GPA, Retention |
| Logistics | 🚚 | Delivery Time, On-Time Rate |
| Manufacturing | 🏭 | Production Rate, Defects |
| Energy | ⚡ | Consumption, Efficiency |
| Tourism | ✈️ | Occupancy, RevPAR |
| Technology | 💻 | Active Users, Churn, MRR |
| Sports | 🏆 | Attendance, Performance |
| Custom | 🔧 | Auto-detected |

---

## 🛠️ Tool Options

### Processing Tools
- **Python (Pandas)**: Direct data manipulation
- **Excel (Power Query)**: Generates M code
- **Power BI**: Generates DAX/Power Query

### Output Formats
- Jupyter Notebook (.ipynb)
- Excel Dashboard (.xlsx)
- Power BI Report (.pbix)
- Word Report (.docx)
- PDF Report (.pdf)

---

## 💡 Tips

1. **Large Datasets**: The app handles datasets of any size efficiently
2. **Domain Selection**: Choose the closest domain for better insights
3. **Feature Engineering**: Start with 'basic_stats' and 'interactions'
4. **Model Selection**: The pipeline tests 9+ models automatically
5. **Custom Domains**: Add new domains via config.py

---

## 🔧 Command-Line Usage

```bash
# Analyze data
python scripts/analyze_data.py --file data.csv --domain hr

# Clean data
python scripts/clean_data.py --file data.csv --tool python

# Generate dashboards
python scripts/generate_dashboard.py --file data.csv --format all

# Export reports
python scripts/export_reports.py --file data.csv --format all
```

---

## 📧 Support

For issues or feature requests, please check the documentation or open an issue.
