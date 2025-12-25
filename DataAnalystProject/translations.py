"""
Translations Module
All UI text in English and Arabic
"""

TRANSLATIONS = {
    'en': {
        # Main titles
        'app_title': '🔬 Data Science Hub',
        'app_subtitle': 'Choose your analytical path',
        
        # Path selection
        'data_analyst': '📊 Data Analyst',
        'data_analyst_desc': 'Actionable insights & practical analysis',
        'data_scientist': '🤖 Data Scientist',
        'data_scientist_desc': 'Advanced ML & predictive models',
        'start_analyst': 'Start Data Analyst Path',
        'start_scientist': 'Start Data Scientist Path',
        
        # Sample data
        'sample_data_title': '📁 Try with Sample Data',
        'sample_hr': '👥 HR',
        'sample_finance': '💰 Finance',
        'sample_healthcare': '🏥 Healthcare',
        'sample_retail': '🛒 Retail',
        'sample_marketing': '📢 Marketing',
        'sample_education': '🎓 Education',
        'loaded_sample': 'Loaded {} sample data!',
        
        # Sidebar
        'configuration': '⚙️ Configuration',
        'current_path': 'Current Path',
        'switch_path': '↩️ Switch Path',
        'domain': '🏢 Domain',
        'select_domain': 'Select Domain',
        'processing_tool': '🛠️ Processing Tool',
        'select_tool': 'Select Tool',
        'output_format': '📤 Output Format',
        'select_format': 'Select Format',
        'data_info': '📊 Data Info',
        'rows': 'Rows',
        'columns': 'Columns',
        'memory': 'Memory',
        
        # Data upload
        'or_use_sample': 'Or start with sample data:',
        'upload_files': '📂 Upload Files',
        'upload_title': '📂 Upload Your Data',
        'upload_hint': 'Choose a file (CSV, Excel, JSON, Parquet)',
        'loaded_success': '✅ Loaded {:,} rows and {} columns',
        'data_preview': '📋 Data Preview',
        'error_loading': 'Error loading file',
        
        # Tabs
        'tab_analysis': '📈 Analysis',
        'tab_cleaning': '🧹 Cleaning',
        'tab_insights': '💡 Insights',
        'tab_dashboard': '📊 Dashboard',
        'tab_export': '📄 Export',
        'tab_target': '🎯 Target Setup',
        'tab_features': '🔧 Feature Engineering',
        'tab_training': '🚀 Model Training',
        'tab_results': '📊 Results',
        
        # Analysis
        'data_quality': '📈 Data Quality Analysis',
        'run_analysis': '🔍 Run Analysis',
        'analyzing': 'Analyzing data...',
        'column_stats': 'Column Statistics',
        'correlation_matrix': 'Correlation Matrix',
        'data_issues': '⚠️ Data Issues',
        'missing': 'Missing',
        'duplicates': 'Duplicates',
        'unique': 'Unique',
        'mean': 'Mean',
        'type': 'Type',
        
        # Cleaning
        'data_cleaning': '🧹 Data Cleaning',
        'using_tool': 'Using tool',
        'select_operations': 'Select cleaning operations',
        'clean_data': '🧹 Clean Data',
        'cleaning_complete': 'Cleaning complete!',
        'generated_code': 'Generated Code',
        'download_cleaned': '📥 Download Cleaned Data',
        'op_missing': 'missing_values',
        'op_duplicates': 'duplicates',
        'op_outliers': 'outliers',
        'op_normalize': 'normalize',
        'op_standardize': 'standardize',
        'op_encode': 'encode',
        
        # Insights
        'domain_insights': '💡 Domain Insights',
        'generate_insights': '🔮 Generate Insights',
        'generating_insights': 'Generating insights...',
        
        # Dashboard
        'dashboard_gen': '📊 Dashboard Generation',
        'select_format_dash': 'Select format',
        'generate_dashboard': '📊 Generate Dashboard',
        'generating_dashboard': 'Generating dashboard...',
        'dashboard_generated': '✅ Dashboard generated',
        'download_notebook': '📥 Download Notebook',
        
        # Export
        'export_reports': '📄 Export Reports',
        'export_word': '📝 Export to Word',
        'export_pdf': '📄 Export to PDF',
        'quick_export': 'Quick Export',
        'download_csv': '📥 Download CSV',
        'word_requires': 'Word export requires python-docx package',
        'pdf_requires': 'PDF export requires reportlab package',
        
        # ML
        'select_target': '🎯 Select Target Variable',
        'target_column': 'Target Column',
        'selected': 'Selected',
        'classification_task': 'Classification Task Detected',
        'regression_task': 'Regression Task Detected',
        'feature_engineering': '🔧 Feature Engineering',
        'select_operations_fe': 'Select operations',
        'engineer_features': '🔧 Engineer Features',
        'engineering_features': 'Engineering features...',
        'created_features': '✅ Created {} new features',
        'train_models': '🚀 Train Models',
        'select_features': 'Select Features',
        'train_all': '🚀 Train All Models',
        'training_models': 'Training models... This may take a while.',
        'training_complete': '✅ Training complete! Best model',
        'select_target_first': 'Please select a target variable in the Target Setup tab first.',
        'model_results': '📊 Model Results',
        'best_model': '🏆 Best Model',
        'cv_score': 'CV Score',
        'test_score': 'Test Score',
        'train_score': 'Train Score',
        'model_comparison': 'Model Comparison',
        'feature_importance': 'Feature Importance',
        'train_first': 'Train models first to see results.',
    },
    
    'ar': {
        # Main titles
        'app_title': '🔬 مركز علوم البيانات',
        'app_subtitle': 'اختر مسار التحليل',
        
        # Path selection
        'data_analyst': '📊 محلل البيانات',
        'data_analyst_desc': 'رؤى قابلة للتنفيذ وتحليل عملي',
        'data_scientist': '🤖 عالم البيانات',
        'data_scientist_desc': 'تعلم آلي متقدم ونماذج تنبؤية',
        'start_analyst': 'ابدأ مسار محلل البيانات',
        'start_scientist': 'ابدأ مسار عالم البيانات',
        
        # Sample data
        'sample_data_title': '📁 جرّب مع بيانات تجريبية',
        'sample_hr': '👥 موارد بشرية',
        'sample_finance': '💰 مالية',
        'sample_healthcare': '🏥 صحة',
        'sample_retail': '🛒 تجزئة',
        'sample_marketing': '📢 تسويق',
        'sample_education': '🎓 تعليم',
        'loaded_sample': 'تم تحميل بيانات {} التجريبية!',
        
        # Sidebar
        'configuration': '⚙️ الإعدادات',
        'current_path': 'المسار الحالي',
        'switch_path': '↩️ تغيير المسار',
        'domain': '🏢 المجال',
        'select_domain': 'اختر المجال',
        'processing_tool': '🛠️ أداة المعالجة',
        'select_tool': 'اختر الأداة',
        'output_format': '📤 صيغة الإخراج',
        'select_format': 'اختر الصيغة',
        'data_info': '📊 معلومات البيانات',
        'rows': 'الصفوف',
        'columns': 'الأعمدة',
        'memory': 'الذاكرة',
        
        # Data upload
        'or_use_sample': 'أو ابدأ ببيانات تجريبية:',
        'upload_files': '📂 رفع الملفات',
        'upload_title': '📂 ارفع بياناتك',
        'upload_hint': 'اختر ملف (CSV, Excel, JSON, Parquet)',
        'loaded_success': '✅ تم تحميل {:,} صف و {} عمود',
        'data_preview': '📋 معاينة البيانات',
        'error_loading': 'خطأ في تحميل الملف',
        
        # Tabs
        'tab_analysis': '📈 التحليل',
        'tab_cleaning': '🧹 التنظيف',
        'tab_insights': '💡 الرؤى',
        'tab_dashboard': '📊 لوحة المعلومات',
        'tab_export': '📄 التصدير',
        'tab_target': '🎯 اختيار الهدف',
        'tab_features': '🔧 هندسة الميزات',
        'tab_training': '🚀 تدريب النماذج',
        'tab_results': '📊 النتائج',
        
        # Analysis
        'data_quality': '📈 تحليل جودة البيانات',
        'run_analysis': '🔍 تشغيل التحليل',
        'analyzing': 'جاري التحليل...',
        'column_stats': 'إحصائيات الأعمدة',
        'correlation_matrix': 'مصفوفة الارتباط',
        'data_issues': '⚠️ مشاكل البيانات',
        'missing': 'مفقود',
        'duplicates': 'مكررات',
        'unique': 'فريد',
        'mean': 'المتوسط',
        'type': 'النوع',
        
        # Cleaning
        'data_cleaning': '🧹 تنظيف البيانات',
        'using_tool': 'الأداة المستخدمة',
        'select_operations': 'اختر عمليات التنظيف',
        'clean_data': '🧹 تنظيف البيانات',
        'cleaning_complete': 'اكتمل التنظيف!',
        'generated_code': 'الكود المُولَّد',
        'download_cleaned': '📥 تحميل البيانات المنظفة',
        'op_missing': 'القيم المفقودة',
        'op_duplicates': 'المكررات',
        'op_outliers': 'القيم الشاذة',
        'op_normalize': 'التطبيع',
        'op_standardize': 'التوحيد',
        'op_encode': 'الترميز',
        
        # Insights
        'domain_insights': '💡 رؤى المجال',
        'generate_insights': '🔮 توليد الرؤى',
        'generating_insights': 'جاري توليد الرؤى...',
        
        # Dashboard
        'dashboard_gen': '📊 إنشاء لوحة المعلومات',
        'select_format_dash': 'اختر الصيغة',
        'generate_dashboard': '📊 إنشاء لوحة المعلومات',
        'generating_dashboard': 'جاري الإنشاء...',
        'dashboard_generated': '✅ تم إنشاء لوحة المعلومات',
        'download_notebook': '📥 تحميل الدفتر',
        
        # Export
        'export_reports': '📄 تصدير التقارير',
        'export_word': '📝 تصدير إلى Word',
        'export_pdf': '📄 تصدير إلى PDF',
        'quick_export': 'تصدير سريع',
        'download_csv': '📥 تحميل CSV',
        'word_requires': 'تصدير Word يتطلب حزمة python-docx',
        'pdf_requires': 'تصدير PDF يتطلب حزمة reportlab',
        
        # ML
        'select_target': '🎯 اختر المتغير الهدف',
        'target_column': 'عمود الهدف',
        'selected': 'تم الاختيار',
        'classification_task': 'تم اكتشاف مهمة تصنيف',
        'regression_task': 'تم اكتشاف مهمة انحدار',
        'feature_engineering': '🔧 هندسة الميزات',
        'select_operations_fe': 'اختر العمليات',
        'engineer_features': '🔧 هندسة الميزات',
        'engineering_features': 'جاري هندسة الميزات...',
        'created_features': '✅ تم إنشاء {} ميزة جديدة',
        'train_models': '🚀 تدريب النماذج',
        'select_features': 'اختر الميزات',
        'train_all': '🚀 تدريب جميع النماذج',
        'training_models': 'جاري التدريب... قد يستغرق بعض الوقت.',
        'training_complete': '✅ اكتمل التدريب! أفضل نموذج',
        'select_target_first': 'الرجاء اختيار المتغير الهدف في تبويب اختيار الهدف أولاً.',
        'model_results': '📊 نتائج النماذج',
        'best_model': '🏆 أفضل نموذج',
        'cv_score': 'نتيجة CV',
        'test_score': 'نتيجة الاختبار',
        'train_score': 'نتيجة التدريب',
        'model_comparison': 'مقارنة النماذج',
        'feature_importance': 'أهمية الميزات',
        'train_first': 'قم بتدريب النماذج أولاً لرؤية النتائج.',
    }
}


def t(key: str, lang: str = 'en') -> str:
    """Get translation for key"""
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)
