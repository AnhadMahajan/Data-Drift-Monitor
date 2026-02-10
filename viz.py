import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from drift import DriftSeverity


class DriftVisualizer:
    
    COLORS = {
        'NO_DRIFT': '#2ecc71',
        'MODERATE_DRIFT': '#f39c12',
        'SEVERE_DRIFT': '#e74c3c',
        'reference': '#3498db',
        'current': '#e67e22'
    }
    
    def __init__(self, figsize=(12, 6)):
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_distribution_comparison(self, reference: np.ndarray, current: np.ndarray,
                                    feature_name: str, is_categorical: bool = False,
                                    drift_result: Optional[Dict] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if is_categorical:
            self._plot_categorical_distribution(ax, reference, current, feature_name)
        else:
            self._plot_numerical_distribution(ax, reference, current, feature_name)
        
        if drift_result:
            severity = drift_result.get('severity', DriftSeverity.NO_DRIFT)
            if isinstance(severity, str):
                color = self.COLORS.get(severity, '#95a5a6')
            else:
                color = self.COLORS.get(severity.value, '#95a5a6')
            
            # Add text box with drift info
            info_text = self._format_drift_info(drift_result)
            ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                   fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def _plot_numerical_distribution(self, ax, reference, current, feature_name):
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        
        all_data = np.concatenate([reference, current])
        bins = np.histogram_bin_edges(all_data, bins='auto')
        
        ax.hist(reference, bins=bins, alpha=0.5, label='Reference', 
               color=self.COLORS['reference'], density=True, edgecolor='black', linewidth=0.5)
        ax.hist(current, bins=bins, alpha=0.5, label='Current', 
               color=self.COLORS['current'], density=True, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel(feature_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Distribution Comparison: {feature_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    def _plot_categorical_distribution(self, ax, reference, current, feature_name):
        ref_counts = pd.Series(reference).value_counts()
        cur_counts = pd.Series(current).value_counts()
        
        all_categories = sorted(set(ref_counts.index) | set(cur_counts.index))
        
        ref_props = [ref_counts.get(cat, 0) / len(reference) for cat in all_categories]
        cur_props = [cur_counts.get(cat, 0) / len(current) for cat in all_categories]
        
        x = np.arange(len(all_categories))
        width = 0.35
        
        ax.bar(x - width/2, ref_props, width, label='Reference', 
              color=self.COLORS['reference'], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, cur_props, width, label='Current', 
              color=self.COLORS['current'], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Category', fontsize=11, fontweight='bold')
        ax.set_ylabel('Proportion', fontsize=11)
        ax.set_title(f'Distribution Comparison: {feature_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_categories, rotation=45, ha='right')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _format_drift_info(self, drift_result: Dict) -> str:
        """Format drift information for annotation"""
        lines = []
        
        if 'ks_statistic' in drift_result and not np.isnan(drift_result['ks_statistic']):
            lines.append(f"KS: {drift_result['ks_statistic']:.4f}")
        
        if 'psi_score' in drift_result and not np.isnan(drift_result['psi_score']):
            lines.append(f"PSI: {drift_result['psi_score']:.4f}")
        
        severity = drift_result.get('severity', 'NO_DRIFT')
        if isinstance(severity, DriftSeverity):
            severity = severity.value
        lines.append(f"Status: {severity}")
        
        return '\n'.join(lines)
    
    def plot_drift_timeline(self, drift_history: List[Dict], feature_name: str) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        timestamps = [d['timestamp'] for d in drift_history]
        ks_scores = [d.get('ks_statistic', np.nan) for d in drift_history]
        psi_scores = [d.get('psi_score', np.nan) for d in drift_history]
        severities = [d.get('severity', 'NO_DRIFT') for d in drift_history]
        
        colors = [self.COLORS.get(s if isinstance(s, str) else s.value, '#95a5a6')
                 for s in severities]
        
        ax1.plot(timestamps, ks_scores, marker='o', linewidth=2, markersize=6, color='#3498db')
        ax1.scatter(timestamps, ks_scores, c=colors, s=100, edgecolor='black', linewidth=1, zorder=3)
        ax1.set_ylabel('KS Statistic', fontsize=11, fontweight='bold')
        ax1.set_title(f'Drift Timeline: {feature_name}', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(timestamps, psi_scores, marker='s', linewidth=2, markersize=6, color='#e67e22')
        ax2.scatter(timestamps, psi_scores, c=colors, s=100, edgecolor='black', linewidth=1, zorder=3)
        ax2.set_ylabel('PSI Score', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Batch Number', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_global_drift_overview(self, drift_summary: pd.DataFrame) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        severity_counts = drift_summary['severity'].value_counts()
        
        labels = []
        sizes = []
        colors = []
        
        for severity in ['NO_DRIFT', 'MODERATE_DRIFT', 'SEVERE_DRIFT']:
            if severity in severity_counts.index:
                labels.append(severity.replace('_', ' ').title())
                sizes.append(severity_counts[severity])
                colors.append(self.COLORS[severity])
        
        if sizes:
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
            ax1.set_title('Drift Distribution Across Features', fontsize=12, fontweight='bold')
        
        feature_colors = [self.COLORS[s] for s in drift_summary['severity']]
        ax2.barh(drift_summary['feature'], drift_summary['psi_score'], 
                color=feature_colors, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('PSI Score', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Feature', fontsize=11, fontweight='bold')
        ax2.set_title('PSI Score by Feature', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        legend_elements = [
            mpatches.Patch(facecolor=self.COLORS['NO_DRIFT'], edgecolor='black', label='No Drift'),
            mpatches.Patch(facecolor=self.COLORS['MODERATE_DRIFT'], edgecolor='black', label='Moderate Drift'),
            mpatches.Patch(facecolor=self.COLORS['SEVERE_DRIFT'], edgecolor='black', label='Severe Drift')
        ]
        ax2.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
        
        plt.tight_layout()
        return fig
    
    def plot_drift_heatmap(self, drift_history_all_features: Dict[str, List[Dict]]) -> plt.Figure:
        features = list(drift_history_all_features.keys())
        n_batches = max(len(history) for history in drift_history_all_features.values())
        
        psi_matrix = np.zeros((len(features), n_batches))
        severity_matrix = np.zeros((len(features), n_batches))
        
        for i, feature in enumerate(features):
            history = drift_history_all_features[feature]
            for j, record in enumerate(history):
                psi_matrix[i, j] = record.get('psi_score', 0)
                
                severity = record.get('severity', 'NO_DRIFT')
                if isinstance(severity, str):
                    severity_str = severity
                else:
                    severity_str = severity.value
                
                if severity_str == 'SEVERE_DRIFT':
                    severity_matrix[i, j] = 2
                elif severity_str == 'MODERATE_DRIFT':
                    severity_matrix[i, j] = 1
                else:
                    severity_matrix[i, j] = 0
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(features) * 0.4)))
        
        cmap = plt.cm.colors.ListedColormap([
            self.COLORS['NO_DRIFT'],
            self.COLORS['MODERATE_DRIFT'],
            self.COLORS['SEVERE_DRIFT']
        ])
        
        im = ax.imshow(severity_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
        
        ax.set_xticks(np.arange(n_batches))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels(np.arange(1, n_batches + 1))
        ax.set_yticklabels(features)
        
        ax.set_xlabel('Batch Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=11, fontweight='bold')
        ax.set_title('Drift Heatmap Over Time', fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(['No Drift', 'Moderate', 'Severe'])
        
        ax.set_xticks(np.arange(n_batches) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(features)) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        
        plt.tight_layout()
        return fig
    
    def plot_concept_drift(self, reference_target: np.ndarray, current_target: np.ndarray,
                          target_name: str, is_classification: bool = True,
                          drift_result: Optional[Dict] = None) -> plt.Figure:
        return self.plot_distribution_comparison(
            reference_target, current_target, target_name,
            is_categorical=is_classification, drift_result=drift_result
        )
