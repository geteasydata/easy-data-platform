import pandas as pd
import numpy as np
import re
from typing import Optional, List, Dict, Any
from data_science_master_system.core.base_classes import BaseProcessor
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)

class TitanicFeatureGenerator(BaseProcessor):
    """
    Specialized feature generator for Titanic dataset.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.medians = {}
    
    def fit(self, data: pd.DataFrame, **kwargs) -> "TitanicFeatureGenerator":
        """Learn stats for imputation."""
        # Median Age by Title
        df = data.copy()
        df['Title'] = df['Name'].apply(self._get_title)
        self.medians['Age'] = df.groupby('Title')['Age'].median().to_dict()
        self.medians['Fare'] = df['Fare'].median()
        self.medians['Embarked'] = df['Embarked'].mode()[0]
        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate Titanic features."""
        df = data.copy()
        
        # 1. Title Extraction
        df['Title'] = df['Name'].apply(self._get_title)
        
        # 2. Imputation
        # Age based on Title
        df['Age'] = df.apply(
            lambda x: self.medians['Age'].get(x['Title'], df['Age'].median()) if pd.isnull(x['Age']) else x['Age'], 
            axis=1
        )
        # Fare
        df['Fare'] = df['Fare'].fillna(self.medians.get('Fare', df['Fare'].median()))
        # Embarked
        df['Embarked'] = df['Embarked'].fillna(self.medians.get('Embarked', 'S'))
        
        # 3. Family Size
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # 4. Fare Binning
        df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)
        
        # 5. Age Binning
        df['AgeBin'] = pd.cut(df['Age'].astype(int), 5, labels=False)
        
        # 6. Name Length
        df['NameLen'] = df['Name'].apply(len)
        
        # 7. Cabin Processing
        df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
        df['Deck'] = df['Cabin'].apply(lambda x: x[0] if type(x) == str else 'M')
        
        # Drop unused
        drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        # Encoding
        # Sex
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        # Embarked
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        # Title
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        df['Title'] = df['Title'].map(title_mapping).fillna(0)
        # Deck
        deck_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8, "M": 0}
        df['Deck'] = df['Deck'].map(deck_mapping).fillna(0)
        
        return df

    def _get_title(self, name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            title = title_search.group(1)
            if title in ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
                return 'Rare'
            elif title in ['Mlle', 'Ms']:
                return 'Miss'
            elif title == 'Mme':
                return 'Mrs'
            return title
        return ""
