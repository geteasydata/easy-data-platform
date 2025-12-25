# Streamlit ML Dashboard Template

Interactive dashboard for ML model exploration, predictions, and analytics.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Access at http://localhost:8501
```

## 📁 Project Structure

```
streamlit_dashboard/
├── app.py               # Main application
├── pages/               # Multi-page components
│   ├── 01_🏠_Home.py
│   ├── 02_📊_EDA.py
│   ├── 03_🤖_Model.py
│   └── 04_🚀_Deploy.py
├── components/          # Reusable components
│   ├── charts.py
│   ├── forms.py
│   └── metrics.py
├── models/              # Model files
├── assets/              # Images, CSS
├── .streamlit/
│   └── config.toml     # Streamlit config
├── requirements.txt
└── README.md
```

## ✨ Features

- 📊 **Data Explorer**: Upload, preview, and visualize data
- 🎯 **Predictions**: Single and batch predictions
- 📈 **Model Analysis**: Performance metrics, confusion matrix, feature importance
- ⚙️ **Settings**: Configurable model paths and display options
- 📤 **Export**: Download results as CSV

## 🌐 Deployment

### Streamlit Cloud
```bash
# Push to GitHub, then connect at streamlit.io/cloud
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

### Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT" > Procfile
heroku create
git push heroku main
```

## 🎨 Customization

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```
