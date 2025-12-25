import json

def generate_notebook(report, target_col):
    """
    Generates a Jupyter Notebook (.ipynb) content as a JSON string.
    Replicates the analysis done by the AutoML tool.
    """
    
    model_type = report.get('model_type', 'Classification')
    is_classification = model_type == 'Classification'
    
    # Notebook Structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    def add_markdown(text):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })
        
    def add_code(code):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.split("\n")]
        })

    # Header
    add_markdown(f"# 🤖 Auto-Generated AI Analysis Notebook\nTarget Variable: **{target_col}**\nModel Type: **{model_type}**")
    
    # Imports
    add_markdown("## 1. Imports")
    imports_code = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error

%matplotlib inline"""
    add_code(imports_code)
    
    # Load Data
    add_markdown("## 2. Load Data\nReplace 'data.csv' with your actual file path.")
    add_code("""# Load your dataset
# df = pd.read_csv('data.csv') 
# For demonstration, creating a dummy dataframe if file not found:
try:
    df = pd.read_csv('data.csv')
    print("Dataset loaded.")
except:
    print("Please provide the path to your dataset in the line above.")
""")

    # Cleaning
    add_markdown("## 3. Data Cleaning & Preprocessing\nBased on the automated analysis, here are the recommended steps.")
    
    cleaning_code = f"""# Target Variable
target = '{target_col}'

# 1. Drop Duplicates
df = df.drop_duplicates()

# 2. Handle Missing Values
# Numeric columns: fill with Mean
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Categorical columns: fill with Mode
cat_cols = df.select_dtypes(exclude=np.number).columns
for col in cat_cols:
    if not df[col].empty:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Data cleaning completed.")
"""
    add_code(cleaning_code)
    
    # Encoding
    add_markdown("## 4. Feature Encoding")
    encoding_code = """# Label Encoding for Categorical Variables
le_dict = {}
for col in df.select_dtypes(exclude=np.number).columns:
    if col != target:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

# Encode Target if needed
if df[target].dtype == 'object':
    target_le = LabelEncoder()
    df[target] = target_le.fit_transform(df[target])
    print(f"Target '{target}' encoded. Classes: {target_le.classes_}")
"""
    add_code(encoding_code)
    
    # Modeling
    add_markdown("## 5. Model Training (Random Forest)")
    
    model_class = "RandomForestClassifier" if is_classification else "RandomForestRegressor"
    metric_calc = "acc = accuracy_score(y_test, y_pred)\\nprint(f'Accuracy: {acc:.2%}')\\nprint(classification_report(y_test, y_pred))" if is_classification else "r2 = r2_score(y_test, y_pred)\\nprint(f'R2 Score: {r2:.4f}')"
    
    modeling_code = f"""X = df.drop(columns=[target])
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Model
model = {model_class}(n_estimators=100, random_state=42)

# Train
model.fit(X_train, y_train)
print("Model trained successfully.")

# Predict
y_pred = model.predict(X_test)

# Evaluate
{metric_calc}
"""
    add_code(modeling_code)
    
    # Feature Importance
    add_markdown("## 6. Feature Importance")
    viz_code = """import pandas as pd
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance.head(10), x='Importance', y='Feature')
plt.title('Top 10 Feature Importance')
plt.show()
"""
    add_code(viz_code)
    
    return json.dumps(notebook, indent=2)
