"""
Configuration Module for Data Science Application
Central configuration for domains, thresholds, and settings
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any
import json
import logging

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "sample_data"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMPLATES_DIR = BASE_DIR / "templates"
DOMAINS_DIR = BASE_DIR / "domains"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, TEMPLATES_DIR, DOMAINS_DIR, 
                 OUTPUT_DIR / "reports", OUTPUT_DIR / "dashboards", OUTPUT_DIR / "notebooks"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisThresholds:
    """Thresholds for data analysis"""
    missing_value_warning: float = 0.05  # 5% missing triggers warning
    missing_value_critical: float = 0.30  # 30% missing triggers critical
    outlier_iqr_multiplier: float = 1.5
    outlier_zscore_threshold: float = 3.0
    correlation_strong: float = 0.7
    correlation_moderate: float = 0.4
    duplicate_sample_size: int = 10000


@dataclass
class ChartConfig:
    """Chart configuration settings"""
    default_palette: List[str] = field(default_factory=lambda: [
        "#667eea", "#764ba2", "#f093fb", "#f5576c", 
        "#4facfe", "#00f2fe", "#43e97b", "#38f9d7"
    ])
    figure_width: int = 10
    figure_height: int = 6
    font_family: str = "Arial"
    title_font_size: int = 14
    label_font_size: int = 12


@dataclass
class ExportSettings:
    """Export configuration"""
    pdf_page_size: str = "A4"
    word_template: str = "default"
    excel_max_rows: int = 1048576
    include_code_in_reports: bool = True
    image_dpi: int = 150


# Domain definitions
DOMAINS = {
    "hr": {
        "name": "Human Resources",
        "name_ar": "الموارد البشرية",
        "icon": "👥",
        "key_metrics": ["employee_count", "turnover_rate", "satisfaction_score", "avg_tenure"],
        "common_columns": ["employee_id", "department", "salary", "hire_date", "performance_score"],
        "insights_focus": ["retention", "performance", "compensation", "diversity"]
    },
    "healthcare": {
        "name": "Healthcare",
        "name_ar": "الرعاية الصحية",
        "icon": "🏥",
        "key_metrics": ["patient_count", "readmission_rate", "avg_stay", "mortality_rate"],
        "common_columns": ["patient_id", "diagnosis", "treatment", "admission_date", "outcome"],
        "insights_focus": ["patient_outcomes", "resource_utilization", "quality_metrics"]
    },
    "finance": {
        "name": "Finance",
        "name_ar": "المالية",
        "icon": "💰",
        "key_metrics": ["revenue", "profit_margin", "cash_flow", "roi"],
        "common_columns": ["transaction_id", "amount", "date", "category", "account"],
        "insights_focus": ["profitability", "risk", "cash_management", "forecasting"]
    },
    "retail": {
        "name": "Retail & Commerce",
        "name_ar": "التجارة والمبيعات",
        "icon": "🛒",
        "key_metrics": ["sales_revenue", "conversion_rate", "avg_order_value", "customer_lifetime"],
        "common_columns": ["order_id", "product", "quantity", "price", "customer_id"],
        "insights_focus": ["sales_trends", "customer_behavior", "inventory", "pricing"]
    },
    "marketing": {
        "name": "Marketing & Advertising",
        "name_ar": "التسويق والإعلان",
        "icon": "📢",
        "key_metrics": ["campaign_roi", "conversion_rate", "cac", "engagement_rate"],
        "common_columns": ["campaign_id", "channel", "spend", "impressions", "conversions"],
        "insights_focus": ["campaign_performance", "channel_attribution", "audience_insights"]
    },
    "education": {
        "name": "Education",
        "name_ar": "التعليم",
        "icon": "🎓",
        "key_metrics": ["graduation_rate", "avg_gpa", "enrollment", "retention_rate"],
        "common_columns": ["student_id", "course", "grade", "enrollment_date", "status"],
        "insights_focus": ["academic_performance", "student_success", "resource_allocation"]
    },
    "logistics": {
        "name": "Logistics & Transportation",
        "name_ar": "النقل والخدمات اللوجستية",
        "icon": "🚚",
        "key_metrics": ["delivery_time", "on_time_rate", "cost_per_delivery", "fleet_utilization"],
        "common_columns": ["shipment_id", "origin", "destination", "weight", "delivery_date"],
        "insights_focus": ["route_optimization", "cost_efficiency", "capacity_planning"]
    },
    "manufacturing": {
        "name": "Manufacturing & Industry",
        "name_ar": "التصنيع والصناعة",
        "icon": "🏭",
        "key_metrics": ["production_rate", "defect_rate", "oee", "inventory_turnover"],
        "common_columns": ["product_id", "batch", "quantity", "quality_score", "production_date"],
        "insights_focus": ["quality_control", "production_efficiency", "maintenance"]
    },
    "energy": {
        "name": "Energy & Environment",
        "name_ar": "الطاقة والبيئة",
        "icon": "⚡",
        "key_metrics": ["energy_consumption", "carbon_footprint", "efficiency_ratio", "renewable_share"],
        "common_columns": ["meter_id", "consumption", "date", "source", "cost"],
        "insights_focus": ["consumption_patterns", "sustainability", "cost_optimization"]
    },
    "tourism": {
        "name": "Tourism & Hospitality",
        "name_ar": "السياحة والضيافة",
        "icon": "✈️",
        "key_metrics": ["occupancy_rate", "adr", "revpar", "guest_satisfaction"],
        "common_columns": ["booking_id", "guest_id", "check_in", "room_type", "revenue"],
        "insights_focus": ["booking_patterns", "revenue_management", "guest_experience"]
    },
    "technology": {
        "name": "Technology & Software",
        "name_ar": "التكنولوجيا والبرمجيات",
        "icon": "💻",
        "key_metrics": ["active_users", "churn_rate", "mrr", "nps"],
        "common_columns": ["user_id", "feature", "event", "timestamp", "session_id"],
        "insights_focus": ["user_engagement", "product_analytics", "growth_metrics"]
    },
    "sports": {
        "name": "Sports & Entertainment",
        "name_ar": "الرياضة والترفيه",
        "icon": "🏆",
        "key_metrics": ["attendance", "revenue", "performance_score", "fan_engagement"],
        "common_columns": ["event_id", "team", "score", "attendance", "revenue"],
        "insights_focus": ["performance_analysis", "fan_analytics", "revenue_optimization"]
    },
    "custom": {
        "name": "Custom Domain",
        "name_ar": "مجال مخصص",
        "icon": "🔧",
        "key_metrics": [],
        "common_columns": [],
        "insights_focus": ["general_analysis", "pattern_detection", "anomaly_detection"]
    }
}

# Tool options
TOOLS = {
    "python": {
        "name": "Python (Pandas)",
        "name_ar": "بايثون",
        "icon": "🐍",
        "description": "Full Python-based processing with pandas"
    },
    "excel": {
        "name": "Excel (Power Query)",
        "name_ar": "إكسل",
        "icon": "📊",
        "description": "Generate Power Query M code for Excel"
    },
    "powerbi": {
        "name": "Power BI",
        "name_ar": "باور بي آي",
        "icon": "📈",
        "description": "Generate DAX and Power Query for Power BI"
    }
}

# Output formats
OUTPUT_FORMATS = {
    "jupyter": {
        "name": "Jupyter Notebook",
        "name_ar": "دفتر جوبيتر",
        "icon": "📓",
        "extension": ".ipynb"
    },
    "excel": {
        "name": "Excel Dashboard",
        "name_ar": "لوحة إكسل",
        "icon": "📊",
        "extension": ".xlsx"
    },
    "powerbi": {
        "name": "Power BI Report",
        "name_ar": "تقرير باور بي آي",
        "icon": "📈",
        "extension": ".pbix"
    },
    "word": {
        "name": "Word Report",
        "name_ar": "تقرير وورد",
        "icon": "📝",
        "extension": ".docx"
    },
    "pdf": {
        "name": "PDF Report",
        "name_ar": "تقرير PDF",
        "icon": "📄",
        "extension": ".pdf"
    }
}


def get_domain_config(domain: str) -> Dict[str, Any]:
    """Get configuration for a specific domain"""
    return DOMAINS.get(domain, DOMAINS["custom"])


def add_custom_domain(domain_id: str, config: Dict[str, Any]) -> bool:
    """Add a new custom domain"""
    try:
        DOMAINS[domain_id] = config
        # Save to custom domains file
        custom_file = DOMAINS_DIR / "custom_domains" / f"{domain_id}.json"
        custom_file.parent.mkdir(parents=True, exist_ok=True)
        with open(custom_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Added custom domain: {domain_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to add custom domain: {e}")
        return False


def load_custom_domains():
    """Load custom domains from files"""
    custom_dir = DOMAINS_DIR / "custom_domains"
    if custom_dir.exists():
        for file in custom_dir.glob("*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    DOMAINS[file.stem] = config
            except Exception as e:
                logger.error(f"Failed to load custom domain {file}: {e}")


# Load custom domains on import
load_custom_domains()


# Global instances
thresholds = AnalysisThresholds()
chart_config = ChartConfig()
export_settings = ExportSettings()
