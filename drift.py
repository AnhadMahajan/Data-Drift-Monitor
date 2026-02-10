import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List
from enum import Enum


class DriftSeverity(Enum):
    NO_DRIFT = "NO_DRIFT"
    MODERATE_DRIFT = "MODERATE_DRIFT"
    SEVERE_DRIFT = "SEVERE_DRIFT"


class DriftDetector:
    def __init__(self, ks_threshold_moderate=0.05, ks_threshold_severe=0.1,
                 psi_threshold_moderate=0.1, psi_threshold_severe=0.25):
        self.ks_threshold_moderate = ks_threshold_moderate
        self.ks_threshold_severe = ks_threshold_severe
        self.psi_threshold_moderate = psi_threshold_moderate
        self.psi_threshold_severe = psi_threshold_severe
        
    def ks_test(self, reference: np.ndarray, current: np.ndarray) -> Dict:
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        
        if len(reference) == 0 or len(current) == 0:
            return {
                'ks_statistic': np.nan,
                'p_value': np.nan,
                'severity': DriftSeverity.NO_DRIFT,
                'explanation': 'Insufficient data for KS test'
            }
        
        ks_statistic, p_value = stats.ks_2samp(reference, current)
        
        if ks_statistic >= self.ks_threshold_severe:
            severity = DriftSeverity.SEVERE_DRIFT
            explanation = f'KS statistic {ks_statistic:.4f} >= {self.ks_threshold_severe} (severe threshold)'
        elif ks_statistic >= self.ks_threshold_moderate:
            severity = DriftSeverity.MODERATE_DRIFT
            explanation = f'KS statistic {ks_statistic:.4f} >= {self.ks_threshold_moderate} (moderate threshold)'
        else:
            severity = DriftSeverity.NO_DRIFT
            explanation = f'KS statistic {ks_statistic:.4f} < {self.ks_threshold_moderate} (no drift detected)'
        
        return {
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'severity': severity,
            'explanation': explanation
        }
    
    def psi(self, reference: np.ndarray, current: np.ndarray, 
            buckets: int = 10, is_categorical: bool = False) -> Dict:
        if is_categorical:
            return self._psi_categorical(reference, current)
        else:
            return self._psi_numerical(reference, current, buckets)
    
    def _psi_numerical(self, reference: np.ndarray, current: np.ndarray, 
                       buckets: int = 10) -> Dict:

        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        
        if len(reference) == 0 or len(current) == 0:
            return {
                'psi_score': np.nan,
                'severity': DriftSeverity.NO_DRIFT,
                'explanation': 'Insufficient data for PSI calculation'
            }
        
        breakpoints = np.linspace(0, 100, buckets + 1)
        breakpoints = np.percentile(reference, breakpoints)
        breakpoints = np.unique(breakpoints)  
        
        if len(breakpoints) < 2:
            return {
                'psi_score': 0.0,
                'severity': DriftSeverity.NO_DRIFT,
                'explanation': 'Constant reference distribution'
            }
        
        ref_counts, _ = np.histogram(reference, bins=breakpoints)
        cur_counts, _ = np.histogram(current, bins=breakpoints)
        
        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)
        
        epsilon = 1e-10
        ref_props = np.where(ref_props == 0, epsilon, ref_props)
        cur_props = np.where(cur_props == 0, epsilon, cur_props)
        
        psi_score = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        if psi_score >= self.psi_threshold_severe:
            severity = DriftSeverity.SEVERE_DRIFT
            explanation = f'PSI score {psi_score:.4f} >= {self.psi_threshold_severe} (severe threshold)'
        elif psi_score >= self.psi_threshold_moderate:
            severity = DriftSeverity.MODERATE_DRIFT
            explanation = f'PSI score {psi_score:.4f} >= {self.psi_threshold_moderate} (moderate threshold)'
        else:
            severity = DriftSeverity.NO_DRIFT
            explanation = f'PSI score {psi_score:.4f} < {self.psi_threshold_moderate} (no drift detected)'
        
        return {
            'psi_score': psi_score,
            'severity': severity,
            'explanation': explanation
        }
    
    def _psi_categorical(self, reference: np.ndarray, current: np.ndarray) -> Dict:
        ref_categories = pd.Series(reference).value_counts(normalize=True)
        cur_categories = pd.Series(current).value_counts(normalize=True)
        
        all_categories = set(ref_categories.index) | set(cur_categories.index)
        
        epsilon = 1e-10
        ref_props = []
        cur_props = []
        
        for cat in all_categories:
            ref_prop = ref_categories.get(cat, 0)
            cur_prop = cur_categories.get(cat, 0)
            
            ref_prop = max(ref_prop, epsilon)
            cur_prop = max(cur_prop, epsilon)
            
            ref_props.append(ref_prop)
            cur_props.append(cur_prop)
        
        ref_props = np.array(ref_props)
        cur_props = np.array(cur_props)
        
        psi_score = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        if psi_score >= self.psi_threshold_severe:
            severity = DriftSeverity.SEVERE_DRIFT
            explanation = f'PSI score {psi_score:.4f} >= {self.psi_threshold_severe} (severe threshold)'
        elif psi_score >= self.psi_threshold_moderate:
            severity = DriftSeverity.MODERATE_DRIFT
            explanation = f'PSI score {psi_score:.4f} >= {self.psi_threshold_moderate} (moderate threshold)'
        else:
            severity = DriftSeverity.NO_DRIFT
            explanation = f'PSI score {psi_score:.4f} < {self.psi_threshold_moderate} (no drift detected)'
        
        return {
            'psi_score': psi_score,
            'severity': severity,
            'explanation': explanation
        }
    
    def detect_feature_drift(self, reference_df: pd.DataFrame, 
                           current_df: pd.DataFrame,
                           numerical_features: List[str],
                           categorical_features: List[str]) -> pd.DataFrame:
        results = []
        
        for feature in numerical_features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue
                
            ref_data = reference_df[feature].values
            cur_data = current_df[feature].values
            
            ks_result = self.ks_test(ref_data, cur_data)
            
            psi_result = self.psi(ref_data, cur_data, is_categorical=False)
            
            severities = [ks_result['severity'], psi_result['severity']]
            if DriftSeverity.SEVERE_DRIFT in severities:
                overall_severity = DriftSeverity.SEVERE_DRIFT
            elif DriftSeverity.MODERATE_DRIFT in severities:
                overall_severity = DriftSeverity.MODERATE_DRIFT
            else:
                overall_severity = DriftSeverity.NO_DRIFT
            
            results.append({
                'feature': feature,
                'type': 'numerical',
                'ks_statistic': ks_result['ks_statistic'],
                'ks_p_value': ks_result['p_value'],
                'psi_score': psi_result['psi_score'],
                'severity': overall_severity.value,
                'explanation': f"{ks_result['explanation']}; {psi_result['explanation']}"
            })
        
        for feature in categorical_features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue
                
            ref_data = reference_df[feature].values
            cur_data = current_df[feature].values
            
            # PSI only for categorical
            psi_result = self.psi(ref_data, cur_data, is_categorical=True)
            
            results.append({
                'feature': feature,
                'type': 'categorical',
                'ks_statistic': np.nan,
                'ks_p_value': np.nan,
                'psi_score': psi_result['psi_score'],
                'severity': psi_result['severity'].value,
                'explanation': psi_result['explanation']
            })
        
        return pd.DataFrame(results)
    
    def detect_concept_drift(self, reference_target: np.ndarray, 
                           current_target: np.ndarray,
                           is_classification: bool = True) -> Dict:
        if is_classification:
            result = self._psi_categorical(reference_target, current_target)
            result['drift_type'] = 'concept_drift_classification'
        else:
            ks_result = self.ks_test(reference_target, current_target)
            psi_result = self._psi_numerical(reference_target, current_target)
            
            # Combine results
            if ks_result['severity'] == DriftSeverity.SEVERE_DRIFT or \
               psi_result['severity'] == DriftSeverity.SEVERE_DRIFT:
                severity = DriftSeverity.SEVERE_DRIFT
            elif ks_result['severity'] == DriftSeverity.MODERATE_DRIFT or \
                 psi_result['severity'] == DriftSeverity.MODERATE_DRIFT:
                severity = DriftSeverity.MODERATE_DRIFT
            else:
                severity = DriftSeverity.NO_DRIFT
            
            result = {
                'ks_statistic': ks_result['ks_statistic'],
                'psi_score': psi_result['psi_score'],
                'severity': severity,
                'explanation': f"{ks_result['explanation']}; {psi_result['explanation']}",
                'drift_type': 'concept_drift_regression'
            }
        
        return result
