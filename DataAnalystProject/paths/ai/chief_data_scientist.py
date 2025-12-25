"""
Chief Data Scientist Agent - The Expert Thinking Layer
========================================================
This agent does NOT clean data, write code, or build models.
It ONLY thinks, questions, rejects, and decides.

Rules:
- Prefer rejecting actions over doing them
- Simpler logic is better than complex models
- Reasoning quality is the main goal, not accuracy
- If data quality is weak, say so clearly
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


class ApprovalStatus(Enum):
    """Status of each thinking stage."""
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


@dataclass
class ThinkingStageResult:
    """Result of a thinking stage."""
    status: ApprovalStatus
    reasoning: str
    concerns: List[str]
    recommendations: List[str]
    confidence: float  # 0.0 to 1.0


class ChiefDataScientist:
    """
    The Senior Expert Mind.
    
    Does NOT clean data.
    Does NOT write code.
    Does NOT build models.
    ONLY thinks, questions, rejects, and decides.
    
    CRITICAL: This agent has AUTHORITY to STOP execution.
    AutoML MUST NOT run unless this agent gives FULL APPROVAL.
    """
    
    # =========================================================================
    # HARD STOP CONDITIONS - NON-NEGOTIABLE
    # These are NOT warnings. These are STOP CONDITIONS.
    # =========================================================================
    HARD_STOP_CONDITIONS = {
        'min_rows': 100,                    # Dataset size < 100 rows
        'min_samples_per_feature': 5,       # Rows < 5 × number of columns
        'max_missing_pct_per_col': 0.30,    # Any column has > 30% missing
        'max_leakage_correlation': 0.95,    # Target leakage risk
        'max_id_like_columns': 2,           # ID-like columns detected
    }
    
    def __init__(self, ai_ensemble=None):
        """Initialize with optional AI ensemble for LLM-powered thinking."""
        self.ai_ensemble = ai_ensemble
        self.thinking_log = []
        self.stage_results = {}
        self._fully_approved = False
        self._rejection_reasons = []
        
    def log(self, message: str):
        """Add to thinking log."""
        self.thinking_log.append(message)
        
    def get_log(self) -> List[str]:
        """Get all thinking log messages."""
        return self.thinking_log
    
    def is_fully_approved(self) -> bool:
        """
        EXPLICIT GATE: Check if ALL stages are APPROVED.
        
        AutoML MUST call this before running.
        If False, AutoML MUST NOT execute.
        """
        if not self.stage_results:
            return False
        
        for stage_name, result in self.stage_results.items():
            if result.status != ApprovalStatus.APPROVED:
                return False
        
        return self._fully_approved
    
    def get_rejection_summary(self, lang: str = 'ar') -> Dict[str, Any]:
        """
        Get summary of why analysis was rejected.
        This is shown INSTEAD of AutoML results.
        """
        summary = {
            'rejected': not self.is_fully_approved(),
            'reasons': self._rejection_reasons,
            'stage_details': {},
            'what_not_to_conclude': [],
            'what_senior_would_do': []
        }
        
        for stage_name, result in self.stage_results.items():
            summary['stage_details'][stage_name] = {
                'status': result.status.value,
                'reasoning': result.reasoning,
                'concerns': result.concerns
            }
            if result.status == ApprovalStatus.REJECTED:
                summary['reasons'].extend(result.concerns)
        
        # What NOT to conclude
        if lang == 'ar':
            summary['what_not_to_conclude'] = [
                "❌ لا تستنتج أن البيانات صالحة للتحليل",
                "❌ لا تستنتج أن أي نموذج سيعمل على هذه البيانات",
                "❌ لا تستنتج نتائج إحصائية من بيانات غير موثوقة"
            ]
            summary['what_senior_would_do'] = [
                "1️⃣ مراجعة مصدر البيانات وعملية جمعها",
                "2️⃣ التحقق من جودة البيانات يدوياً",
                "3️⃣ جمع بيانات إضافية إذا لزم الأمر",
                "4️⃣ استشارة خبير المجال قبل المتابعة"
            ]
        else:
            summary['what_not_to_conclude'] = [
                "❌ Do NOT conclude that this data is suitable for analysis",
                "❌ Do NOT conclude that any model will work on this data",
                "❌ Do NOT draw statistical conclusions from unreliable data"
            ]
            summary['what_senior_would_do'] = [
                "1️⃣ Review data source and collection process",
                "2️⃣ Manually verify data quality",
                "3️⃣ Collect additional data if necessary",
                "4️⃣ Consult domain expert before proceeding"
            ]
        
        return summary
    
    # =========================================================================
    # EXPERT RECOVERY MODE
    # =========================================================================
    
    def generate_recovery_plan(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        lang: str = 'ar'
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive recovery plan when analysis is rejected.
        
        A real senior data scientist NEVER just stops.
        They stop execution AND provide a recovery plan.
        """
        recovery = {
            'root_cause_diagnosis': [],
            'repair_plan': [],
            'auto_fix_candidates': {
                'safe': [],      # ✔ Can be automated
                'confirm': [],   # ⚠ Need human approval
                'never': []      # ❌ Must NEVER automate
            },
            're_entry_conditions': [],
            'domain_suggestions': []
        }
        
        n_samples = len(df)
        n_features = len(df.columns) - 1
        
        # =====================================================================
        # A. ROOT CAUSE DIAGNOSIS
        # =====================================================================
        for stage_name, result in self.stage_results.items():
            if result.status != ApprovalStatus.APPROVED:
                for concern in result.concerns:
                    diagnosis = self._diagnose_issue(concern, df, target_col, lang)
                    recovery['root_cause_diagnosis'].append(diagnosis)
        
        # =====================================================================
        # B. EXPERT REPAIR PLAN
        # =====================================================================
        for diagnosis in recovery['root_cause_diagnosis']:
            repair = self._generate_repair_options(diagnosis, df, target_col, lang)
            recovery['repair_plan'].append(repair)
        
        # =====================================================================
        # C. AUTO-FIX CANDIDATES CLASSIFICATION
        # =====================================================================
        recovery['auto_fix_candidates'] = self._classify_auto_fixes(df, target_col, lang)
        
        # =====================================================================
        # D. RE-ENTRY CONDITIONS
        # =====================================================================
        recovery['re_entry_conditions'] = self._define_re_entry_conditions(lang)
        
        # =====================================================================
        # E. DOMAIN-AWARE SUGGESTIONS
        # =====================================================================
        recovery['domain_suggestions'] = self._get_domain_suggestions(df, target_col, lang)
        
        return recovery
    
    def _diagnose_issue(
        self, 
        concern: str, 
        df: pd.DataFrame, 
        target_col: str,
        lang: str
    ) -> Dict[str, Any]:
        """Diagnose a single issue with severity and statistical explanation."""
        
        # Determine severity
        if "HARD STOP" in concern or "LEAKAGE" in concern.upper():
            severity = "CRITICAL"
        elif "missing" in concern.lower() or "ID" in concern:
            severity = "MAJOR"
        else:
            severity = "MINOR"
        
        # Statistical explanation
        explanations = {
            'rows': "Small samples lead to high variance in model estimates, unreliable validation",
            'samples per feature': "Violates statistical rule of thumb, causes overfitting",
            'missing': "Missing data can bias model, imputation may introduce artifacts",
            'leakage': "Model learns target information, performance won't generalize",
            'ID': "ID columns have no predictive meaning, create spurious correlations",
            'imbalance': "Class imbalance biases accuracy metrics, may need resampling"
        }
        
        stat_reason = "General data quality concern"
        for key, explanation in explanations.items():
            if key.lower() in concern.lower():
                stat_reason = explanation
                break
        
        return {
            'concern': concern,
            'severity': severity,
            'statistical_reason': stat_reason,
            'severity_icon': '🔴' if severity == 'CRITICAL' else ('🟠' if severity == 'MAJOR' else '🟡')
        }
    
    def _generate_repair_options(
        self, 
        diagnosis: Dict, 
        df: pd.DataFrame, 
        target_col: str,
        lang: str
    ) -> Dict[str, Any]:
        """Generate repair options for a diagnosed issue."""
        
        concern = diagnosis['concern'].lower()
        repair = {
            'issue': diagnosis['concern'],
            'severity': diagnosis['severity'],
            'fix_conservative': '',
            'fix_aggressive': '',
            'risks_conservative': '',
            'risks_aggressive': '',
            'when_not_to_fix': ''
        }
        
        # Generate appropriate fixes based on issue type
        if 'rows' in concern or 'samples' in concern:
            repair['fix_conservative'] = "Collect more data (recommended: 10× current size)"
            repair['fix_aggressive'] = "Use data augmentation or synthetic data generation"
            repair['risks_conservative'] = "Time and cost to collect data"
            repair['risks_aggressive'] = "Synthetic data may not represent real distribution"
            repair['when_not_to_fix'] = "When data collection is impossible or too expensive"
            
        elif 'missing' in concern:
            repair['fix_conservative'] = "Remove rows/columns with >30% missing"
            repair['fix_aggressive'] = "Impute using KNN or iterative imputation"
            repair['risks_conservative'] = "Loss of potentially useful data"
            repair['risks_aggressive'] = "Imputed values may introduce bias"
            repair['when_not_to_fix'] = "When missingness is informative (MNAR)"
            
        elif 'leakage' in concern:
            repair['fix_conservative'] = "Remove the leaking column entirely"
            repair['fix_aggressive'] = "Investigate if column is available at prediction time"
            repair['risks_conservative'] = "May lose genuinely useful feature"
            repair['risks_aggressive'] = "May still have subtle leakage"
            repair['when_not_to_fix'] = "When column IS legitimately available in production"
            
        elif 'id' in concern:
            repair['fix_conservative'] = "Remove all ID-like columns before modeling"
            repair['fix_aggressive'] = "Keep only if ID encodes meaningful information"
            repair['risks_conservative'] = "None - IDs should always be removed"
            repair['risks_aggressive'] = "High risk of spurious correlations"
            repair['when_not_to_fix'] = "Never - ID columns must always be removed"
            
        elif 'imbalance' in concern:
            repair['fix_conservative'] = "Use class weights in training"
            repair['fix_aggressive'] = "Apply SMOTE or undersampling"
            repair['risks_conservative'] = "May not fully address imbalance"
            repair['risks_aggressive'] = "SMOTE can create unrealistic samples"
            repair['when_not_to_fix'] = "When imbalance reflects real-world distribution"
            
        else:
            repair['fix_conservative'] = "Review data collection process"
            repair['fix_aggressive'] = "Consult domain expert for data cleaning"
            repair['risks_conservative'] = "May miss fixable issues"
            repair['risks_aggressive'] = "May over-engineer the data"
            repair['when_not_to_fix'] = "When issue is inherent to the problem"
        
        return repair
    
    def _classify_auto_fixes(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        lang: str
    ) -> Dict[str, List[Dict]]:
        """
        Classify fixes into safe/confirm/never categories.
        MUST match apply_safe_fixes logic!
        """
        
        classification = {
            'safe': [],      # ✔ Can be automated
            'confirm': [],   # ⚠ Need human approval
            'never': []      # ❌ Must NEVER automate
        }
        
        # =====================================================================
        # SAFE: ID columns (by name)
        # =====================================================================
        id_name_patterns = ['id', 'Id', 'ID', 'index', 'Index', 'INDEX', 'Unnamed: 0']
        for col in df.columns:
            if col in id_name_patterns and col != target_col:
                classification['safe'].append({
                    'action': f"Remove ID column '{col}'",
                    'reason': "ID provides no predictive value",
                    'code': f"df = df.drop(columns=['{col}'])"
                })
        
        # =====================================================================
        # SAFE: Columns with >30% missing (will be auto-removed - Hard Stop)
        # =====================================================================
        missing_pct = df.isnull().mean()
        for col in missing_pct[missing_pct > 0.30].index:
            if col != target_col:
                classification['safe'].append({
                    'action': f"Remove column '{col}' ({missing_pct[col]:.0%} missing)",
                    'reason': "Fails Hard Stop (>30%) - Must be removed",
                    'code': f"df = df.drop(columns=['{col}'])"
                })
        
        # =====================================================================
        # SAFE: Constant columns
        # =====================================================================
        for col in df.columns:
            if df[col].nunique() <= 1 and col != target_col:
                classification['safe'].append({
                    'action': f"Remove constant column '{col}'",
                    'reason': "Zero variance provides no information",
                    'code': f"df = df.drop(columns=['{col}'])"
                })
        
        # =====================================================================
        # SAFE: Remaining missing (<30%) will be filled
        # =====================================================================
        remaining_missing = missing_pct[(missing_pct > 0) & (missing_pct <= 0.30)]
        if len(remaining_missing) > 0:
            classification['safe'].append({
                'action': f"Fill missing in {len(remaining_missing)} columns",
                'reason': "Numeric → median, Categorical → mode",
                'code': "# Automatic imputation"
            })
        
        # =====================================================================
        # CONFIRM: Borderline cases (None in strict mode, but keeping structure)
        # =====================================================================
        # Currently empty as we handle >30% strictly
        
        # =====================================================================
        # NEVER: Target modifications
        # =====================================================================
        classification['never'].append({
            'action': "Modify target variable",
            'reason': "Target definition must be deliberate human decision"
        })
        classification['never'].append({
            'action': "Remove rows based on target value",
            'reason': "Would change problem definition and introduce bias"
        })
        classification['never'].append({
            'action': "Impute target variable",
            'reason': "Target must reflect ground truth, not estimates"
        })
        
        return classification
    
    def _define_re_entry_conditions(self, lang: str) -> List[Dict]:
        """Define conditions for re-running expert approval."""
        
        conditions = [
            {
                'condition': 'min_rows_met',
                'description': 'Dataset has ≥ 100 rows' if lang != 'ar' else 'البيانات تحتوي على 100 صف على الأقل',
                'check': f"len(df) >= {self.HARD_STOP_CONDITIONS['min_rows']}"
            },
            {
                'condition': 'samples_per_feature_met',
                'description': 'At least 5 samples per feature' if lang != 'ar' else '5 عينات على الأقل لكل ميزة',
                'check': f"len(df) / (len(df.columns)-1) >= {self.HARD_STOP_CONDITIONS['min_samples_per_feature']}"
            },
            {
                'condition': 'missing_rate_acceptable',
                'description': 'No column has > 30% missing' if lang != 'ar' else 'لا يوجد عمود به أكثر من 30% قيم مفقودة',
                'check': "df.isnull().mean().max() <= 0.30"
            },
            {
                'condition': 'no_leakage',
                'description': 'No feature has > 0.95 correlation with target' if lang != 'ar' else 'لا توجد ميزة بارتباط > 0.95 مع الهدف',
                'check': "max(correlations) < 0.95"
            },
            {
                'condition': 'id_columns_removed',
                'description': 'All ID-like columns removed' if lang != 'ar' else 'تمت إزالة جميع أعمدة المعرف',
                'check': "no monotonic columns with unique values"
            }
        ]
        
        return conditions
    
    def _get_domain_suggestions(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        lang: str
    ) -> List[str]:
        """Get domain-aware suggestions based on problem type."""
        
        suggestions = []
        target = df[target_col] if target_col in df.columns else None
        n_samples = len(df)
        
        if target is None:
            return suggestions
        
        # Determine problem type
        if target.dtype in ['object', 'category'] or target.nunique() <= 20:
            problem_type = 'classification'
        else:
            problem_type = 'regression'
        
        # Classification-specific
        if problem_type == 'classification':
            class_counts = target.value_counts(normalize=True)
            if class_counts.iloc[0] > 0.8:
                suggestions.append("⚖️ Class imbalance detected → Consider SMOTE, class weights, or threshold tuning")
            if target.nunique() > 10:
                suggestions.append("📊 Many classes → Consider grouping rare classes or hierarchical classification")
        
        # Regression-specific
        if problem_type == 'regression':
            suggestions.append("📈 Regression task → Consider target transformations (log, Box-Cox) if skewed")
            suggestions.append("🎯 Use RMSE/MAE for evaluation, not just R²")
        
        # Small data suggestions
        if n_samples < 500:
            suggestions.extend([
                "📉 Small dataset → Use cross-validation instead of holdout",
                "🌳 Prefer simple models: Logistic/Linear Regression, Decision Trees",
                "❌ Avoid: Deep Learning, Large Ensembles, Neural Networks",
                "📊 Consider: Regularization (L1/L2) to prevent overfitting"
            ])
        
        # High dimensional
        n_features = len(df.columns) - 1
        if n_features > n_samples / 5:
            suggestions.extend([
                "📐 High-dimensional → Apply PCA or feature selection first",
                "🎯 Use Lasso (L1) regularization for automatic feature selection"
            ])
        
        return suggestions
    
    def apply_safe_fixes(
        self, 
        df: pd.DataFrame, 
        target_col: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply safe fixes that a senior data scientist would approve automatically.
        
        SAFE to auto-fix:
        - ID columns (always remove)
        - Constant columns (zero information)
        - Columns with >80% missing (objectively useless)
        - Columns with >50% missing (likely not worth imputing)
        - Duplicate rows
        """
        changes = []
        df_fixed = df.copy()
        
        # =====================================================================
        # 1. REMOVE ID COLUMNS - Be conservative, only clear ID patterns
        # =====================================================================
        # Check for columns named like IDs
        id_name_patterns = ['id', 'Id', 'ID', 'index', 'Index', 'INDEX', 'Unnamed: 0']
        for col in list(df_fixed.columns):
            if col in id_name_patterns and col != target_col and col in df_fixed.columns:
                df_fixed = df_fixed.drop(columns=[col])
                changes.append(f"✔️ Removed ID column: '{col}'")
        
        # Check for integer sequence columns (0,1,2,3... or 1,2,3,4...)
        for col in df_fixed.select_dtypes(include=[np.number]).columns[:10]:
            if col == target_col or col not in df_fixed.columns:
                continue
            col_data = df_fixed[col]
            # Must be all unique, monotonic, and be an exact sequence
            if col_data.nunique() == len(df_fixed):
                sorted_vals = sorted(col_data.dropna().values)
                is_sequence = all(sorted_vals[i] == sorted_vals[0] + i for i in range(len(sorted_vals)))
                if is_sequence and (sorted_vals[0] == 0 or sorted_vals[0] == 1):
                    df_fixed = df_fixed.drop(columns=[col])
                    changes.append(f"✔️ Removed sequential ID column: '{col}'")
        
        # =====================================================================
        # 2. REMOVE COLUMNS WITH >30% MISSING (Strict Quality Control)
        # Why 30%? Because Chief Data Scientist sets a HARD STOP at 30%.
        # If we keep 30-50% missing, the analysis will just be rejected again.
        # =====================================================================
        missing_pct = df_fixed.isnull().mean()
        high_missing = missing_pct[missing_pct > 0.30].index.tolist()
        for col in high_missing:
            if col != target_col and col in df_fixed.columns:
                df_fixed = df_fixed.drop(columns=[col])
                changes.append(f"✔️ Removed column '{col}' ({missing_pct[col]:.0%} missing - fails quality gate)")
        
        # =====================================================================
        # 3. FILL REMAINING MISSING VALUES (<30%)
        # =====================================================================
        # Recalculate only if needed, but we know all >30% are gone.
        remaining_missing_cols = df_fixed.columns[df_fixed.isnull().any()].tolist()
        
        # =====================================================================
        # 4. REMOVE CONSTANT COLUMNS
        # =====================================================================
        for col in list(df_fixed.columns):  # Use list() to avoid modification during iteration
            if col in df_fixed.columns and df_fixed[col].nunique() <= 1 and col != target_col:
                df_fixed = df_fixed.drop(columns=[col])
                changes.append(f"✔️ Removed constant column: '{col}'")
        
        # =====================================================================
        # 5. REMOVE DUPLICATE ROWS
        # =====================================================================
        n_before = len(df_fixed)
        df_fixed = df_fixed.drop_duplicates()
        n_removed = n_before - len(df_fixed)
        if n_removed > 0:
            changes.append(f"✔️ Removed {n_removed} duplicate rows")
        
        # =====================================================================
        # 6. FILL REMAINING MISSING VALUES (simple strategy)
        # =====================================================================
        for col in df_fixed.columns:
            if col == target_col:
                continue
            missing_count = df_fixed[col].isnull().sum()
            if missing_count > 0:
                if df_fixed[col].dtype in ['object', 'category']:
                    # Fill categorical with mode
                    mode_val = df_fixed[col].mode()
                    if len(mode_val) > 0:
                        df_fixed[col] = df_fixed[col].fillna(mode_val[0])
                        changes.append(f"✔️ Filled {missing_count} missing in '{col}' with mode")
                else:
                    # Fill numeric with median
                    median_val = df_fixed[col].median()
                    df_fixed[col] = df_fixed[col].fillna(median_val)
                    changes.append(f"✔️ Filled {missing_count} missing in '{col}' with median")
        
        return df_fixed, changes
    
    # =========================================================================
    # STAGE 1: PROBLEM REFRAMING
    # =========================================================================
    
    def stage1_problem_reframing(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        lang: str = 'ar'
    ) -> ThinkingStageResult:
        """
        Stage 1: Reframe the problem before any analysis.
        
        HARD STOP CONDITIONS CHECKED HERE:
        - Dataset size < 100 rows → REJECT
        - Rows < 5 × columns → REJECT  
        - Target is ID-like → REJECT
        """
        self.log("🧠 Stage 1: Problem Reframing")
        
        concerns = []
        recommendations = []
        hard_stop = False
        
        n_samples = len(df)
        n_features = len(df.columns) - 1
        samples_per_feature = n_samples / max(n_features, 1)
        
        # Analyze target column
        target = df[target_col] if target_col in df.columns else None
        
        if target is None:
            self._rejection_reasons.append(f"HARD STOP: Target column '{target_col}' does not exist")
            return ThinkingStageResult(
                status=ApprovalStatus.REJECTED,
                reasoning=f"❌ I REFUSE to analyze: Target column '{target_col}' not found.",
                concerns=["Target column does not exist"],
                recommendations=["Select a valid target column"],
                confidence=0.0
            )
        
        # =====================================================================
        # HARD STOP CONDITION 1: Dataset too small
        # =====================================================================
        if n_samples < self.HARD_STOP_CONDITIONS['min_rows']:
            hard_stop = True
            reason = f"HARD STOP: Dataset has only {n_samples} rows. Minimum required: {self.HARD_STOP_CONDITIONS['min_rows']}"
            concerns.append(reason)
            self._rejection_reasons.append(reason)
            recommendations.append("❌ Using ML here would be STATISTICALLY IRRESPONSIBLE")
        
        # =====================================================================
        # HARD STOP CONDITION 2: Rows < 5 × columns (severe underfitting risk)
        # =====================================================================
        if samples_per_feature < self.HARD_STOP_CONDITIONS['min_samples_per_feature']:
            hard_stop = True
            reason = f"HARD STOP: Only {samples_per_feature:.1f} samples per feature. Minimum required: {self.HARD_STOP_CONDITIONS['min_samples_per_feature']}"
            concerns.append(reason)
            self._rejection_reasons.append(reason)
            recommendations.append("❌ Results would be MISLEADING due to overfitting")
        
        # =====================================================================
        # HARD STOP CONDITION 3: Target is ID-like (trivially predictable)
        # =====================================================================
        target_unique_ratio = target.nunique() / len(target)
        if target_unique_ratio > 0.95:
            hard_stop = True
            reason = "HARD STOP: Target appears to be an ID column (nearly unique values)"
            concerns.append(reason)
            self._rejection_reasons.append(reason)
            recommendations.append("❌ This is NOT a valid prediction problem")
        
        # =====================================================================
        # WARNINGS (not hard stops, but concerning)
        # =====================================================================
        if target.dtype in ['object', 'category'] or target.nunique() <= 10:
            value_counts = target.value_counts(normalize=True)
            if value_counts.iloc[0] > 0.9:
                concerns.append(f"⚠️ Target is heavily imbalanced: {value_counts.iloc[0]:.1%} in majority class")
        
        if n_features <= 3 and target.nunique() <= 5:
            recommendations.append("💡 Simple rules might work better than ML")
        
        # Generate reasoning with AI if available
        if hard_stop:
            if lang == 'ar':
                reasoning = f"❌ **أرفض تحليل هذه البيانات.**\n\nالأسباب:\n" + "\n".join([f"• {c}" for c in concerns])
            else:
                reasoning = f"❌ **I REFUSE to analyze this data.**\n\nReasons:\n" + "\n".join([f"• {c}" for c in concerns])
        else:
            reasoning = self._generate_problem_reframing_reasoning(df, target_col, concerns, lang)
        
        # =====================================================================
        # DETERMINE STATUS
        # =====================================================================
        if hard_stop:
            status = ApprovalStatus.REJECTED
            confidence = 0.0
        elif len(concerns) >= 2:
            status = ApprovalStatus.REJECTED  # Changed: 2+ concerns = REJECT, not NEEDS_REVIEW
            confidence = 0.3
        else:
            status = ApprovalStatus.APPROVED
            confidence = 0.8 - (len(concerns) * 0.15)
        
        result = ThinkingStageResult(
            status=status,
            reasoning=reasoning,
            concerns=concerns,
            recommendations=recommendations,
            confidence=max(0.0, confidence)
        )
        
        self.stage_results['problem_reframing'] = result
        return result
    
    # =========================================================================
    # STAGE 2: DATA SKEPTICISM
    # =========================================================================
    
    def stage2_data_skepticism(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        lang: str = 'ar'
    ) -> ThinkingStageResult:
        """
        Stage 2: Question the dataset validity.
        
        HARD STOP CONDITIONS CHECKED HERE:
        - Any column has > 30% missing → REJECT
        - Data leakage detected (correlation > 0.95) → REJECT
        - ID-like columns dominate variance → REJECT
        """
        self.log("🔍 Stage 2: Data Skepticism & Quality Check")
        
        concerns = []
        recommendations = []
        hard_stop = False
        id_like_count = 0
        
        # =====================================================================
        # HARD STOP CONDITION 1: Any column has > 30% missing
        # =====================================================================
        missing_pct = df.isnull().mean()
        high_missing_cols = missing_pct[missing_pct > self.HARD_STOP_CONDITIONS['max_missing_pct_per_col']].index.tolist()
        if high_missing_cols:
            hard_stop = True
            reason = f"HARD STOP: {len(high_missing_cols)} columns have >30% missing data: {high_missing_cols[:3]}"
            concerns.append(reason)
            self._rejection_reasons.append(reason)
            recommendations.append("❌ Data quality is TOO POOR for reliable analysis")
        
        # =====================================================================
        # HARD STOP CONDITION 2: Data leakage detected
        # =====================================================================
        if target_col in df.columns:
            target = df[target_col]
            for col in df.columns:
                if col == target_col:
                    continue
                if df[col].dtype in ['object', 'category']:
                    continue
                try:
                    corr = df[col].corr(target.astype(float))
                    if abs(corr) > self.HARD_STOP_CONDITIONS['max_leakage_correlation']:
                        hard_stop = True
                        reason = f"HARD STOP: DATA LEAKAGE - '{col}' has {corr:.2f} correlation with target"
                        concerns.append(reason)
                        self._rejection_reasons.append(reason)
                        recommendations.append(f"❌ '{col}' is likely DERIVED from the target - analysis would be FRAUDULENT")
                except:
                    pass
        
        # =====================================================================
        # HARD STOP CONDITION 3: ID-like columns dominate
        # =====================================================================
        for col in df.select_dtypes(include=[np.number]).columns[:15]:
            if df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing:
                if df[col].nunique() == len(df):
                    id_like_count += 1
                    concerns.append(f"⚠️ '{col}' is a sequential ID - MUST be excluded")
        
        if id_like_count > self.HARD_STOP_CONDITIONS['max_id_like_columns']:
            hard_stop = True
            reason = f"HARD STOP: {id_like_count} ID-like columns detected dominating the data"
            concerns.append(reason)
            self._rejection_reasons.append(reason)
            recommendations.append("❌ Remove all ID columns before analysis")
        
        # =====================================================================
        # WARNINGS (not hard stops)
        # =====================================================================
        dup_pct = df.duplicated().mean()
        if dup_pct > 0.05:
            concerns.append(f"⚠️ {dup_pct:.1%} duplicate rows detected")
        
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            concerns.append(f"⚠️ {len(constant_cols)} columns have no variance (constant)")
        
        # Generate reasoning
        if hard_stop:
            if lang == 'ar':
                reasoning = f"❌ **أرفض المتابعة - البيانات غير موثوقة.**\n\n" + "\n".join([f"• {c}" for c in concerns])
            else:
                reasoning = f"❌ **I REFUSE to proceed - data is UNRELIABLE.**\n\n" + "\n".join([f"• {c}" for c in concerns])
        else:
            reasoning = self._generate_data_skepticism_reasoning(df, target_col, concerns, lang)
        
        # =====================================================================
        # DETERMINE STATUS - STRICT
        # =====================================================================
        if hard_stop:
            status = ApprovalStatus.REJECTED
            confidence = 0.0
        elif len(concerns) >= 3:
            status = ApprovalStatus.REJECTED  # 3+ concerns = REJECT
            confidence = 0.2
        elif len(concerns) >= 1:
            status = ApprovalStatus.APPROVED  # 1-2 concerns = APPROVED with lower confidence
            confidence = 0.6 - (len(concerns) * 0.15)
        else:
            status = ApprovalStatus.APPROVED
            confidence = 0.85
        
        result = ThinkingStageResult(
            status=status,
            reasoning=reasoning,
            concerns=concerns,
            recommendations=recommendations,
            confidence=max(0.0, confidence)
        )
        
        self.stage_results['data_skepticism'] = result
        return result
    
    # =========================================================================
    # STAGE 3: ANALYSIS STRATEGY
    # =========================================================================
    
    def stage3_analysis_strategy(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        lang: str = 'ar'
    ) -> ThinkingStageResult:
        """
        Stage 3: Decide the analysis approach BEFORE doing it.
        
        Questions to answer:
        - What modeling approach is appropriate?
        - What should we NOT do?
        - What validation strategy is suitable?
        - What are the success criteria?
        """
        self.log("📋 Stage 3: Analysis Strategy Decision")
        
        concerns = []
        recommendations = []
        
        n_samples = len(df)
        n_features = len(df.columns) - 1
        target = df[target_col] if target_col in df.columns else None
        
        # Determine problem type
        if target is not None:
            if target.dtype in ['object', 'category'] or target.nunique() <= 20:
                problem_type = 'classification'
                n_classes = target.nunique()
            else:
                problem_type = 'regression'
                n_classes = None
        else:
            problem_type = 'unknown'
            n_classes = None
        
        # Strategy decisions
        strategy = {
            'problem_type': problem_type,
            'recommended_models': [],
            'avoid_models': [],
            'validation_strategy': '',
            'success_criteria': []
        }
        
        # Model recommendations based on data characteristics
        if n_samples < 1000:
            recommendations.append("Small dataset: prefer simpler models (Logistic Regression, Decision Trees)")
            strategy['recommended_models'] = ['Logistic Regression', 'Decision Tree', 'Random Forest']
            strategy['avoid_models'] = ['Deep Learning', 'Large Ensembles']
            concerns.append("Dataset may be too small for complex models to generalize")
        elif n_samples < 10000:
            recommendations.append("Medium dataset: tree-based models should work well")
            strategy['recommended_models'] = ['Random Forest', 'Gradient Boosting', 'XGBoost']
        else:
            recommendations.append("Large dataset: can consider more complex approaches")
            strategy['recommended_models'] = ['XGBoost', 'LightGBM', 'Neural Networks']
        
        # Validation strategy
        if n_samples < 500:
            strategy['validation_strategy'] = 'Leave-One-Out or 10-Fold CV (small sample)'
            concerns.append("Small sample size limits validation reliability")
        elif n_samples < 5000:
            strategy['validation_strategy'] = '5-Fold Cross-Validation'
        else:
            strategy['validation_strategy'] = 'Train/Validation/Test Split (70/15/15)'
        
        # Success criteria
        if problem_type == 'classification':
            if n_classes and n_classes == 2:
                strategy['success_criteria'] = [
                    'AUC-ROC as primary metric (handles imbalance better)',
                    'Precision/Recall based on business cost of errors',
                    'Confusion matrix analysis for error patterns'
                ]
            else:
                strategy['success_criteria'] = [
                    'Macro F1-Score for balanced class importance',
                    'Per-class metrics to identify weak spots'
                ]
        else:
            strategy['success_criteria'] = [
                'RMSE for absolute error magnitude',
                'R² for explained variance',
                'Residual analysis for model assumptions'
            ]
        
        # What NOT to do
        if n_features > n_samples / 10:
            concerns.append("⚠️ High-dimensional data: feature selection is CRITICAL")
            recommendations.append("Apply aggressive feature selection before modeling")
        
        # Generate reasoning with AI if available
        reasoning = self._generate_strategy_reasoning(df, target_col, strategy, concerns, lang)
        
        # Store strategy for later use
        self.analysis_strategy = strategy
        
        # =====================================================================
        # DETERMINE STATUS - STRICT (no NEEDS_REVIEW - either APPROVED or REJECTED)
        # =====================================================================
        if problem_type == 'unknown':
            status = ApprovalStatus.REJECTED
            confidence = 0.0
            self._rejection_reasons.append("REJECTED: Cannot determine problem type")
        elif len(concerns) >= 3:
            status = ApprovalStatus.REJECTED  # Changed: 3+ concerns = REJECT
            confidence = 0.3
            self._rejection_reasons.append(f"REJECTED: Too many concerns ({len(concerns)}) for reliable analysis")
        else:
            status = ApprovalStatus.APPROVED
            confidence = 0.75 - (len(concerns) * 0.1)
        
        result = ThinkingStageResult(
            status=status,
            reasoning=reasoning,
            concerns=concerns,
            recommendations=recommendations,
            confidence=max(0.0, confidence)
        )
        
        self.stage_results['analysis_strategy'] = result
        
        # =====================================================================
        # SET FULLY APPROVED FLAG
        # Only True if ALL THREE stages are APPROVED
        # =====================================================================
        all_approved = all(
            r.status == ApprovalStatus.APPROVED 
            for r in self.stage_results.values()
        )
        self._fully_approved = all_approved
        
        if not all_approved:
            self.log("❌ EXECUTION BLOCKED: Not all stages approved")
        else:
            self.log("✅ ALL STAGES APPROVED: AutoML may proceed")
        
        return result
    
    # =========================================================================
    # SELF-CRITIQUE STAGE (POST-ANALYSIS)
    # =========================================================================
    
    def generate_self_critique(
        self,
        results: Dict[str, Any],
        lang: str = 'ar'
    ) -> Dict[str, Any]:
        """
        Generate self-critique of the analysis results.
        
        Challenges:
        - Weak assumptions made
        - How the analysis could be wrong
        - Overconfidence warnings
        """
        self.log("⚖️ Self-Critique Stage")
        
        critique = {
            'weak_assumptions': [],
            'potential_errors': [],
            'overconfidence_warnings': [],
            'expert_warnings': [],
            'confidence_level': 'medium'
        }
        
        metrics = results.get('metrics', {})
        problem_type = results.get('problem_type', '')
        best_model = results.get('best_model', '')
        
        # Check 1: Accuracy illusion
        if problem_type == 'classification':
            acc = metrics.get('accuracy', 0)
            if acc > 0.95:
                critique['overconfidence_warnings'].append(
                    f"Accuracy of {acc:.1%} is suspiciously high - possible data leakage or overfitting"
                )
                critique['confidence_level'] = 'low'
            elif acc > 0.85:
                critique['weak_assumptions'].append(
                    "High accuracy may not transfer to production data"
                )
        
        # Check 2: Model complexity concerns
        if 'Ensemble' in best_model or 'XGBoost' in best_model or 'LightGBM' in best_model:
            critique['potential_errors'].append(
                "Complex model selected - may overfit to training patterns"
            )
            critique['expert_warnings'].append(
                "A senior data scientist would recommend testing with simpler baselines first"
            )
        
        # Check 3: Validation concerns
        critique['weak_assumptions'].append(
            "We assume the test/train split is representative of future data"
        )
        critique['weak_assumptions'].append(
            "We assume no temporal drift or distribution shift in production"
        )
        
        # Check 4: Feature importance concerns
        feature_importance = results.get('feature_importance')
        if feature_importance is not None and len(feature_importance) > 0:
            top_feature = feature_importance.iloc[0]['Feature'] if isinstance(feature_importance, pd.DataFrame) else 'Unknown'
            top_importance = feature_importance.iloc[0]['Importance'] if isinstance(feature_importance, pd.DataFrame) else 0
            if top_importance > 0.5:
                critique['potential_errors'].append(
                    f"Model heavily relies on '{top_feature}' ({top_importance:.1%}) - is this feature always available?"
                )
        
        # Generate expert warnings with AI if available
        if self.ai_ensemble:
            try:
                ai_warnings = self._generate_ai_critique(results, lang)
                if ai_warnings:
                    critique['expert_warnings'].extend(ai_warnings)
            except:
                pass
        
        # Add standard expert warnings
        critique['expert_warnings'].extend([
            "Always validate on truly held-out data before deployment",
            "Monitor model performance continuously in production",
            "This analysis is a starting point, not a final answer"
        ])
        
        return critique
    
    # =========================================================================
    # EXPERT OUTPUT FORMATTING
    # =========================================================================
    
    def format_expert_output(
        self,
        results: Dict[str, Any],
        critique: Dict[str, Any],
        lang: str = 'ar'
    ) -> Dict[str, str]:
        """
        Format the output like a senior data scientist would present it.
        """
        output = {}
        
        # Expert interpretation
        interpretation = self._generate_expert_interpretation(results, lang)
        output['expert_interpretation'] = interpretation
        
        # Practical recommendations
        recommendations = self._generate_practical_recommendations(results, lang)
        output['practical_recommendations'] = recommendations
        
        # Uncertainty and risk
        output['uncertainty_statement'] = self._generate_uncertainty_statement(results, critique, lang)
        
        # Senior warnings
        output['senior_warnings'] = "\n".join([f"⚠️ {w}" for w in critique['expert_warnings']])
        
        return output
    
    # =========================================================================
    # HELPER METHODS FOR AI-POWERED REASONING
    # =========================================================================
    
    def _generate_problem_reframing_reasoning(
        self, df: pd.DataFrame, target_col: str, concerns: List[str], lang: str
    ) -> str:
        """Generate reasoning for problem reframing stage."""
        if self.ai_ensemble and hasattr(self.ai_ensemble, '_call_groq'):
            try:
                prompt = f"""أنت كبير علماء البيانات. أعد صياغة هذه المشكلة بإيجاز:
                
الهدف: {target_col}
الأعمدة: {len(df.columns)}
الصفوف: {len(df)}
المخاوف: {concerns}

أجب في 2-3 جمل فقط. ركز على: هل هذه المشكلة الصحيحة للحل؟
اللغة: {'العربية' if lang == 'ar' else 'English'}"""
                return self.ai_ensemble._call_groq(prompt)
            except:
                pass
        
        # Fallback reasoning
        if lang == 'ar':
            return f"تحليل الهدف '{target_col}' على {len(df)} صف و {len(df.columns)} عمود. " + \
                   ("توجد مخاوف يجب معالجتها." if concerns else "المشكلة واضحة ومناسبة للتحليل.")
        else:
            return f"Analyzing target '{target_col}' on {len(df)} rows and {len(df.columns)} columns. " + \
                   ("There are concerns that need attention." if concerns else "Problem is clear and suitable for analysis.")
    
    def _generate_data_skepticism_reasoning(
        self, df: pd.DataFrame, target_col: str, concerns: List[str], lang: str
    ) -> str:
        """Generate reasoning for data skepticism stage."""
        if self.ai_ensemble and hasattr(self.ai_ensemble, '_call_groq'):
            try:
                prompt = f"""أنت كبير علماء البيانات. قيّم جودة هذه البيانات بإيجاز:

القيم المفقودة: {df.isnull().sum().sum()}
المكررات: {df.duplicated().sum()}
المخاوف: {concerns}

أجب في 2-3 جمل فقط. ركز على: هل البيانات موثوقة؟
اللغة: {'العربية' if lang == 'ar' else 'English'}"""
                return self.ai_ensemble._call_groq(prompt)
            except:
                pass
        
        # Fallback reasoning
        missing_pct = df.isnull().mean().mean() * 100
        if lang == 'ar':
            quality = "جيدة" if missing_pct < 5 else ("متوسطة" if missing_pct < 20 else "ضعيفة")
            return f"جودة البيانات {quality}. نسبة القيم المفقودة: {missing_pct:.1f}%. " + \
                   (f"تم رصد {len(concerns)} مخاوف تحتاج مراجعة." if concerns else "البيانات تبدو سليمة.")
        else:
            quality = "good" if missing_pct < 5 else ("moderate" if missing_pct < 20 else "poor")
            return f"Data quality is {quality}. Missing values: {missing_pct:.1f}%. " + \
                   (f"Found {len(concerns)} concerns requiring review." if concerns else "Data appears clean.")
    
    def _generate_strategy_reasoning(
        self, df: pd.DataFrame, target_col: str, strategy: Dict, concerns: List[str], lang: str
    ) -> str:
        """Generate reasoning for analysis strategy stage."""
        if self.ai_ensemble and hasattr(self.ai_ensemble, '_call_groq'):
            try:
                prompt = f"""أنت كبير علماء البيانات. اقترح استراتيجية التحليل بإيجاز:

نوع المشكلة: {strategy['problem_type']}
النماذج المقترحة: {strategy['recommended_models']}
النماذج المتجنبة: {strategy['avoid_models']}

أجب في 2-3 جمل فقط. ركز على: لماذا هذه الاستراتيجية؟
اللغة: {'العربية' if lang == 'ar' else 'English'}"""
                return self.ai_ensemble._call_groq(prompt)
            except:
                pass
        
        # Fallback reasoning
        if lang == 'ar':
            return f"نوع المشكلة: {strategy['problem_type']}. " + \
                   f"النماذج المقترحة: {', '.join(strategy['recommended_models'][:3])}. " + \
                   f"استراتيجية التحقق: {strategy['validation_strategy']}."
        else:
            return f"Problem type: {strategy['problem_type']}. " + \
                   f"Recommended models: {', '.join(strategy['recommended_models'][:3])}. " + \
                   f"Validation: {strategy['validation_strategy']}."
    
    def _generate_ai_critique(self, results: Dict, lang: str) -> List[str]:
        """Generate AI-powered critique."""
        if not self.ai_ensemble:
            return []
        
        try:
            prompt = f"""أنت كبير علماء البيانات. انتقد هذه النتائج:

أفضل نموذج: {results.get('best_model', 'غير محدد')}
الدقة: {results.get('metrics', {}).get('accuracy', results.get('metrics', {}).get('r2', 'N/A'))}

أعطني 2 تحذيرات مهمة فقط. كل تحذير في سطر واحد.
اللغة: {'العربية' if lang == 'ar' else 'English'}"""
            
            response = self.ai_ensemble._call_groq(prompt)
            return [line.strip() for line in response.split('\n') if line.strip()][:2]
        except:
            return []
    
    def _generate_expert_interpretation(self, results: Dict, lang: str) -> str:
        """Generate expert interpretation of results."""
        metrics = results.get('metrics', {})
        problem_type = results.get('problem_type', '')
        best_model = results.get('best_model', '')
        
        if problem_type == 'classification':
            acc = metrics.get('accuracy', 0)
            if lang == 'ar':
                if acc > 0.85:
                    return f"النموذج يحقق دقة {acc:.1%} وهي نتيجة جيدة، لكن يجب التحقق من عدم وجود تسرب بيانات."
                elif acc > 0.7:
                    return f"النموذج يحقق دقة {acc:.1%} وهي نتيجة مقبولة للنماذج الأولية."
                else:
                    return f"النموذج يحقق دقة {acc:.1%} فقط - يحتاج تحسين كبير أو مراجعة جودة البيانات."
            else:
                if acc > 0.85:
                    return f"Model achieves {acc:.1%} accuracy - good result, but verify no data leakage."
                elif acc > 0.7:
                    return f"Model achieves {acc:.1%} accuracy - acceptable for initial model."
                else:
                    return f"Model achieves only {acc:.1%} accuracy - needs significant improvement or data quality review."
        else:
            r2 = metrics.get('r2', 0)
            if lang == 'ar':
                return f"النموذج يفسر {r2:.1%} من التباين في البيانات."
            else:
                return f"Model explains {r2:.1%} of variance in the data."
    
    def _generate_practical_recommendations(self, results: Dict, lang: str) -> str:
        """Generate practical recommendations."""
        if lang == 'ar':
            return """
1. 📊 **اختبر على بيانات جديدة** قبل الاعتماد على النتائج
2. 🔄 **راقب الأداء** بشكل دوري في الإنتاج
3. 📉 **ابدأ بنموذج بسيط** كخط أساس للمقارنة
4. 🎯 **ركز على مقاييس العمل** وليس فقط الدقة التقنية
"""
        else:
            return """
1. 📊 **Test on new data** before relying on results
2. 🔄 **Monitor performance** regularly in production
3. 📉 **Start with simple model** as baseline for comparison
4. 🎯 **Focus on business metrics**, not just technical accuracy
"""
    
    def _generate_uncertainty_statement(
        self, results: Dict, critique: Dict, lang: str
    ) -> str:
        """Generate statement about uncertainty and risk."""
        confidence = critique.get('confidence_level', 'medium')
        n_warnings = len(critique.get('overconfidence_warnings', []))
        
        if lang == 'ar':
            if confidence == 'low' or n_warnings > 0:
                return "⚠️ **مستوى الثقة: منخفض** - توجد مؤشرات تستدعي الحذر قبل الاعتماد على هذه النتائج."
            elif confidence == 'medium':
                return "📊 **مستوى الثقة: متوسط** - النتائج معقولة لكن تحتاج تحققاً إضافياً."
            else:
                return "✅ **مستوى الثقة: مقبول** - النتائج تبدو سليمة مع المحاذير المذكورة."
        else:
            if confidence == 'low' or n_warnings > 0:
                return "⚠️ **Confidence: LOW** - There are indicators that warrant caution before relying on these results."
            elif confidence == 'medium':
                return "📊 **Confidence: MEDIUM** - Results are reasonable but need additional validation."
            else:
                return "✅ **Confidence: ACCEPTABLE** - Results appear sound with noted caveats."


# =========================================================================
# FACTORY FUNCTION
# =========================================================================

def get_chief_data_scientist(ai_ensemble=None) -> ChiefDataScientist:
    """Get a ChiefDataScientist instance."""
    return ChiefDataScientist(ai_ensemble=ai_ensemble)
