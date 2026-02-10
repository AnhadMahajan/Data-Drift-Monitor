import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class DataLoader:
    def __init__(self, categorical_threshold: int = 10):
        self.categorical_threshold = categorical_threshold
        self.reference_schema = None
        self.numerical_features = []
        self.categorical_features = []
        
    def load_csv(self, file_path_or_buffer) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path_or_buffer)
            return df
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")
    
    def infer_schema(self, df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> Dict:
        if exclude_columns is None:
            exclude_columns = []
        
        numerical = []
        categorical = []
        
        for col in df.columns:
            if col in exclude_columns:
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                n_unique = df[col].nunique()
                if n_unique <= self.categorical_threshold and n_unique > 1:
                    categorical.append(col)
                else:
                    numerical.append(col)
            else:
                categorical.append(col)
        
        schema = {
            'numerical': numerical,
            'categorical': categorical,
            'all_columns': list(df.columns)
        }
        
        return schema
    
    def set_reference_schema(self, df: pd.DataFrame, exclude_columns: Optional[List[str]] = None):
        self.reference_schema = self.infer_schema(df, exclude_columns)
        self.numerical_features = self.reference_schema['numerical']
        self.categorical_features = self.reference_schema['categorical']
        
        self.reference_categories = {}
        for col in self.categorical_features:
            if col in df.columns:
                self.reference_categories[col] = set(df[col].dropna().unique())
    
    def align_schema(self, df: pd.DataFrame, handle_unseen: str = 'mark_as_other') -> pd.DataFrame:
        if self.reference_schema is None:
            raise ValueError("Reference schema not set. Call set_reference_schema first.")
        
        aligned_df = df.copy()
        
        missing_cols = set(self.reference_schema['all_columns']) - set(df.columns)
        if missing_cols:
            print(f"Warning: Missing columns in incoming data: {missing_cols}")
            for col in missing_cols:
                aligned_df[col] = np.nan
        
        extra_cols = set(df.columns) - set(self.reference_schema['all_columns'])
        if extra_cols:
            print(f"Warning: Extra columns in incoming data (will be dropped): {extra_cols}")
            aligned_df = aligned_df.drop(columns=list(extra_cols))
        
        for col in self.categorical_features:
            if col not in aligned_df.columns:
                continue
                
            if col in self.reference_categories:
                ref_cats = self.reference_categories[col]
                current_cats = set(aligned_df[col].dropna().unique())
                unseen_cats = current_cats - ref_cats
                
                if unseen_cats:
                    if handle_unseen == 'mark_as_other':
                        aligned_df.loc[aligned_df[col].isin(unseen_cats), col] = 'UNSEEN_CATEGORY'
                    elif handle_unseen == 'drop':
                        aligned_df = aligned_df[~aligned_df[col].isin(unseen_cats)]
        
        aligned_df = aligned_df[self.reference_schema['all_columns']]
        
        return aligned_df
    
    def get_feature_lists(self) -> Tuple[List[str], List[str]]:
        return self.numerical_features, self.categorical_features
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        stats = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'missing_values': {},
            'data_types': {},
            'unique_counts': {}
        }
        
        for col in df.columns:
            stats['missing_values'][col] = df[col].isna().sum()
            stats['data_types'][col] = str(df[col].dtype)
            stats['unique_counts'][col] = df[col].nunique()
        
        return stats
    
    def create_batches(self, df: pd.DataFrame, batch_size: int = 100) -> List[pd.DataFrame]:
        n_batches = int(np.ceil(len(df) / batch_size))
        batches = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batch = df.iloc[start_idx:end_idx].copy()
            batches.append(batch)
        
        return batches
    
    def compute_reference_stats(self, df: pd.DataFrame) -> Dict:
        stats = {
            'numerical': {},
            'categorical': {}
        }
        
        for col in self.numerical_features:
            if col in df.columns:
                data = df[col].dropna()
                stats['numerical'][col] = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'median': data.median(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75),
                    'data': data.values  # Store for KS test
                }
        
        for col in self.categorical_features:
            if col in df.columns:
                value_counts = df[col].value_counts(normalize=True)
                stats['categorical'][col] = {
                    'categories': list(value_counts.index),
                    'proportions': value_counts.to_dict(),
                    'n_categories': len(value_counts),
                    'data': df[col].values  # Store for PSI
                }
        
        return stats
