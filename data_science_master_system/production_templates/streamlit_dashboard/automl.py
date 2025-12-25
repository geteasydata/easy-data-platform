"""
AutoML Engine - The "Virtual Expert" Logic
Currently supports: Tabular data (CSV/Excel)
Capabilities:
- Auto-cleaning
- Type detection
- Auto-modeling (Classification/Regression)
- Insights generation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report, mean_absolute_error
import joblib

class AutoMLExpert:
    def __init__(self):
        self.model = None
        self.params = {}
        self.history = []
        self.target_col = None
        self.problem_type = None
        self.le_dict = {} # Label encoders
        self.feature_names = []
        
    def analyze_dataset(self, df):
        """Initial check of the data like an expert would do."""
        analysis = {
            "rows": len(df),
            "cols": len(df.columns),
            "numeric_cols": list(df.select_dtypes(include=np.number).columns),
            "cat_cols": list(df.select_dtypes(exclude=np.number).columns),
            "missing": df.isnull().sum().sum(),
            "duplicates": df.duplicated().sum()
        }
        return analysis

    def clean_data(self, df):
        """Auto-clean the data."""
        df = df.copy()
        steps = []
        
        # 1. Drop duplicates
        dupes = df.duplicated().sum()
        if dupes > 0:
            df = df.drop_duplicates()
            steps.append(f"🗑️ Removed {dupes} duplicate rows")
            
        # 2. Handle missing values
        # Numeric -> Mean
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            
        # Categorical -> Mode (Most frequent) or "Unknown"
        cat_cols = df.select_dtypes(exclude=np.number).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        if df.isnull().sum().sum() > 0:
            steps.append("🧹 Fill remaining missing values with defaults")
        else:
            steps.append("✨ Data cleaned (missing values handled)")
            
        return df, steps

    def auto_train(self, df, target_col, test_df=None):
        """Train model. If test_df is provided, predict on it."""
        self.target_col = target_col
        self.feature_names = [c for c in df.columns if c != target_col]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Detect problem type
        if y.dtype == 'object' or len(y.unique()) < 20:
            self.problem_type = "Classification"
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.problem_type = "Regression"
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
        # Preprocessing (Label Encoding for categories)
        X_processed = X.copy()
        
        # Encode Categorical Features
        for col in X_processed.select_dtypes(exclude=np.number).columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            self.le_dict[col] = le
            
        # Handle Target if categorical
        if self.problem_type == "Classification" and y.dtype == 'object':
            target_le = LabelEncoder()
            y = target_le.fit_transform(y)
            self.le_dict['target'] = target_le

        # Train Split
        if test_df is None:
            X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        else:
            X_train, y_train = X_processed, y
            # Process Test Data
            X_test_raw = test_df.copy()
            # If target exists in test, drop it
            if target_col in X_test_raw.columns:
                 X_test_raw = X_test_raw.drop(columns=[target_col])
            
            # Apply same encodings
            for col, le in self.le_dict.items():
                if col in X_test_raw.columns:
                    # Handle unknown categories safely
                    X_test_raw[col] = X_test_raw[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            
            X_test = X_test_raw[self.feature_names] # Ensure same order
            y_test = None # We might not have y_test

        # Train
        model.fit(X_train, y_train)
        self.model = model
        
        # Evaluate/Predict
        metrics = {}
        predictions = model.predict(X_test)
        
        # If we have y_test (internal split), calculate metrics
        if y_test is not None:
            if self.problem_type == "Classification":
                acc = accuracy_score(y_test, y_pred=predictions)
                metrics['Accuracy'] = f"{acc:.1%}"
            else:
                r2 = r2_score(y_test, predictions)
                metrics['R² Score'] = f"{r2:.2f}"
            
        # Feature Importance
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return metrics, importance, predictions

    def generate_report(self, df, target, test_df=None):
        """Run workflow with optional test set."""
        report = {}
        report['initial'] = self.analyze_dataset(df)
        
        # Clean Train
        df_clean, clean_steps = self.clean_data(df)
        report['cleaning_steps'] = clean_steps
        
        # Clean Test if exists
        df_test_clean = None
        if test_df is not None:
            df_test_clean, _ = self.clean_data(test_df)
        
        # Train
        metrics, importance, preds = self.auto_train(df_clean, target, df_test_clean)
        report['metrics'] = metrics
        report['importance'] = importance
        report['model_type'] = self.problem_type
        
        # If external test set, attach predictions to it
        output_df = df_clean
        if df_test_clean is not None:
            output_df = df_test_clean.copy()
            # Inverse transform predictions if classification
            if self.problem_type == "Classification" and 'target' in self.le_dict:
                preds = self.le_dict['target'].inverse_transform(preds)
                
            output_df[target + '_PREDICTED'] = preds
        
    def generate_business_insights(self, report, target_col):
        """
        Generate strict, expert-level business recommendations based on findings.
        This simulates a Senior Data Scientist's thought process.
        """
        insights = []
        metrics = report['metrics']
        top_features = report['importance'].head(3)['Feature'].tolist()
        model_type = report['model_type']
        
        # 1. Performance Interpretation
        main_score = metrics.get('Main Metric', 0)
        
        if model_type == "Classification":
            score_text = f"Acc: {main_score:.1%}"
            if main_score > 0.90:
                insights.append(f"✅ **أداء ممتاز ({score_text}):** النموذج جاهز للإنتاج ويمكن الاعتماد عليه في اتخاذ القرارات الآلية.")
            elif main_score > 0.75:
                insights.append(f"⚠️ **أداء جيد ({score_text}):** النموذج مفيد ولكن يحتاج مراقبة بشرية لبعض الحالات الصعبة.")
            else:
                insights.append(f"❌ **أداء ضعيف ({score_text}):** البيانات الحالية غير كافية للتنبؤ بدقة. نحتاج بيانات إضافية أو ميزات جديدة.")
        else:
            score_text = f"R²: {main_score:.2f}"
            if main_score > 0.80:
                insights.append(f"✅ **تفسير قوي ({score_text}):** النموذج يفسر معظم العوامل المؤثرة في {target_col}.")
            else:
                insights.append(f"⚠️ **تفسير متوسط ({score_text}):** هناك عوامل خارجية تؤثر في {target_col} لم يتم رصدها في البيانات.")

        # 2. Strategic Recommendations based on Features
        insights.append("\n**💡 التوصيات الاستراتيجية (Strategic Actions):**")
        
        f1 = top_features[0] if len(top_features) > 0 else "N/A"
        f2 = top_features[1] if len(top_features) > 1 else "N/A"
        
        insights.append(f"1. **شريان العمل هو '{f1}':** أظهر التحليل أن هذا هو المحرك الرئيسي للنتائج. أي تحسين أو تغيير في {f1} سيؤثر بشكل مباشر وقوي على {target_col}. ركز مواردك هنا.")
        
        if len(top_features) > 1:
            insights.append(f"2. **راقب '{f2}':** هذا العامل هو المؤثر الثاني. قد يكون هناك فرصة غير مستغلة لتحسين النتائج من خلال ضبط {f2}.")
            
        # 3. Data Quality Audit
        cleaning = report['cleaning_steps']
        if len(cleaning) > 2:
            insights.append("\n**🛡️ ملاحظات جودة البيانات (Data Audit):**")
            insights.append("• جودة البيانات الأصلية كانت منخفضة واحتوت على شوائب. تحسين جمع البيانات سيزيد الدقة مستقبلاً بشكل كبير.")

        return "\n".join(insights)
