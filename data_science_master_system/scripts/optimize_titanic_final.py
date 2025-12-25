import sys
import os
import pandas as pd
import logging

# Ensure project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_science_master_system.core.logger import get_logger
from data_science_master_system.logic import (
    CoreLogic, AnalyticalLogic, StatisticalLogic, PredictiveLogic,
    CausalLogic, EngineeringLogic, CommercialLogic, EthicalLogic
)
from data_science_master_system.features.engineering.titanic_features import TitanicFeatureGenerator
# We will use the same optimization function as Ultra, but called via Logic
from scripts.optimize_titanic_ultra import main as run_ultra_optimization

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("==================================================")
    logger.info("   INITIATING OMNI-LOGIC ENGINE (8 LAYERS)   ")
    logger.info("==================================================")
    
    # 1. Initialize The Brain
    core = CoreLogic()
    analytical = AnalyticalLogic()
    statistical = StatisticalLogic()
    predictive = PredictiveLogic()
    causal = CausalLogic()
    engineering = EngineeringLogic()
    commercial = CommercialLogic()
    ethical = EthicalLogic()
    
    # 2. Load Data
    train_path = os.path.join(project_root, 'data', 'train.csv')
    df = pd.read_csv(train_path)
    logger.info(f"Loaded Raw Data: {df.shape}")
    
    # 3. Layer 2: Analytical (Investigator)
    eda_report = analytical.execute(df)
    logger.info(f"Analytical Report: Outliers detected in {len(eda_report.get('outlier_counts', {}))} cols")
    
    # 4. Layer 5: Causal (Scientist)
    causal_report = causal.execute(df)
    
    # 5. Layer 6: Engineering (Architect)
    # This prepares the blueprint. The actual transformation is handled by our Ultra script for now.
    eng_report = engineering.execute(df)
    
    # 6. Layer 4: Predictive (Forecaster) - Execution Phase
    logger.info(">>> HANDING OVER TO PREDICTIVE ENGINE (Ultra Strategy) <<<")
    # We invoke the 'Ultra' script we perfected in Phase 3 as the "Execution Arm" of the Predictive Layer
    # Ideally, we refactor run_ultra_optimization to accept arguments, but it runs main() independently.
    # We will trigger it.
    
    try:
        run_ultra_optimization() # This runs the actual optimization and generates submission_ultra.csv
        logger.info(">>> PREDICTIVE ENGINE COMPLETE <<<")
        
        # 7. Layer 8: Ethical (Guardian)
        # Verify the output
        sub_path = 'submission_ultra.csv'
        if os.path.exists(sub_path):
            sub_df = pd.read_csv(sub_path)
            ethical_report = ethical.execute(sub_df)
            logger.info(f"Ethical Audit: {ethical_report['bias_report']}")
            
            logger.info("==================================================")
            logger.info("   MISSION ACCOMPLISHED: SUBMISSION GENERATED   ")
            logger.info("==================================================")
        else:
            logger.error("Predictive Engine failed to generate submission.")
            
    except Exception as e:
        logger.error(f"Critical Failure in Predictive Layer: {e}")

if __name__ == "__main__":
    main()
