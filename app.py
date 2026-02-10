import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import json

from data_loader import DataLoader
from drift import DriftDetector, DriftSeverity
from viz import DriftVisualizer


st.set_page_config(
    page_title="Data Drift Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-severe {
        background-color: #ffebee;
        border-left: 5px solid #e74c3c;
    }
    .alert-moderate {
        background-color: #fff3e0;
        border-left: 5px solid #f39c12;
    }
    .alert-good {
        background-color: #e8f5e9;
        border-left: 5px solid #2ecc71;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    if 'reference_data' not in st.session_state:
        st.session_state.reference_data = None
    if 'reference_stats' not in st.session_state:
        st.session_state.reference_stats = None
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    if 'drift_detector' not in st.session_state:
        st.session_state.drift_detector = None
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'drift_history' not in st.session_state:
        st.session_state.drift_history = {}
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
    if 'batch_counter' not in st.session_state:
        st.session_state.batch_counter = 0
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []


def sidebar_controls():
    st.sidebar.markdown("##  Data Upload")
    
    st.sidebar.markdown("### Reference Dataset")
    reference_file = st.sidebar.file_uploader(
        "Upload reference/training data (CSV)",
        type=['csv'],
        key='reference_upload'
    )
    
    if reference_file is not None:
        try:
            df = st.session_state.data_loader.load_csv(reference_file)
            
            st.sidebar.markdown("### Target Configuration")
            all_columns = ['None'] + list(df.columns)
            target_col = st.sidebar.selectbox(
                "Select target column (optional)",
                all_columns,
                key='target_selector'
            )
            
            exclude_cols = [target_col] if target_col != 'None' else []
            st.session_state.target_column = target_col if target_col != 'None' else None
            
            st.session_state.data_loader.set_reference_schema(df, exclude_cols)
            st.session_state.reference_data = df
            st.session_state.reference_stats = st.session_state.data_loader.compute_reference_stats(df)
            
            st.sidebar.success(f" Reference data loaded: {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            st.sidebar.error(f"Error loading reference data: {str(e)}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("##  Drift Thresholds")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        ks_moderate = st.sidebar.slider(
            "KS Moderate",
            0.01, 0.20, 0.05, 0.01,
            help="KS statistic threshold for moderate drift"
        )
        psi_moderate = st.sidebar.slider(
            "PSI Moderate",
            0.05, 0.30, 0.10, 0.01,
            help="PSI threshold for moderate drift"
        )
    
    with col2:
        ks_severe = st.sidebar.slider(
            "KS Severe",
            0.05, 0.30, 0.10, 0.01,
            help="KS statistic threshold for severe drift"
        )
        psi_severe = st.sidebar.slider(
            "PSI Severe",
            0.10, 0.50, 0.25, 0.01,
            help="PSI threshold for severe drift"
        )
    
    st.session_state.drift_detector = DriftDetector(
        ks_threshold_moderate=ks_moderate,
        ks_threshold_severe=ks_severe,
        psi_threshold_moderate=psi_moderate,
        psi_threshold_severe=psi_severe
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Incoming Data")
    
    current_file = st.sidebar.file_uploader(
        "Upload new data for monitoring (CSV)",
        type=['csv'],
        key='current_upload'
    )
    
    if current_file is not None and st.session_state.reference_data is not None:
        try:
            current_df = st.session_state.data_loader.load_csv(current_file)
            aligned_df = st.session_state.data_loader.align_schema(current_df)
            st.session_state.current_data = aligned_df
            
            st.sidebar.success(f"Current data loaded: {len(aligned_df)} rows")
            
        except Exception as e:
            st.sidebar.error(f"Error loading current data: {str(e)}")
    
    if st.session_state.current_data is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("## Batch Processing")
        
        batch_size = st.sidebar.slider(
            "Batch size",
            10, 500, 100, 10,
            help="Number of rows per batch"
        )
        
        if st.sidebar.button("Process Next Batch", use_container_width=True):
            process_batch(batch_size)
        
        if st.sidebar.button("Reset Monitor", use_container_width=True):
            reset_monitor()


def process_batch(batch_size):
    if st.session_state.current_data is None:
        st.error("No current data loaded")
        return
    
    start_idx = st.session_state.batch_counter * batch_size
    end_idx = min((st.session_state.batch_counter + 1) * batch_size, 
                  len(st.session_state.current_data))
    
    if start_idx >= len(st.session_state.current_data):
        st.warning("All batches processed")
        return
    
    batch_df = st.session_state.current_data.iloc[start_idx:end_idx]
    
    numerical_features, categorical_features = st.session_state.data_loader.get_feature_lists()
    
    drift_results = st.session_state.drift_detector.detect_feature_drift(
        st.session_state.reference_data,
        batch_df,
        numerical_features,
        categorical_features
    )
    
    st.session_state.batch_counter += 1
    batch_record = {
        'batch_num': st.session_state.batch_counter,
        'timestamp': datetime.now(),
        'n_rows': len(batch_df),
        'drift_results': drift_results
    }
    st.session_state.batch_results.append(batch_record)
    
    for _, row in drift_results.iterrows():
        feature = row['feature']
        if feature not in st.session_state.drift_history:
            st.session_state.drift_history[feature] = []
        
        st.session_state.drift_history[feature].append({
            'timestamp': st.session_state.batch_counter,
            'ks_statistic': row.get('ks_statistic', np.nan),
            'psi_score': row['psi_score'],
            'severity': row['severity']
        })
    
    severe_drifts = drift_results[drift_results['severity'] == 'SEVERE_DRIFT']
    moderate_drifts = drift_results[drift_results['severity'] == 'MODERATE_DRIFT']
    
    if len(severe_drifts) > 0:
        for _, row in severe_drifts.iterrows():
            alert = {
                'timestamp': datetime.now(),
                'batch_num': st.session_state.batch_counter,
                'severity': 'SEVERE',
                'feature': row['feature'],
                'message': f"SEVERE DRIFT detected in {row['feature']}: {row['explanation']}"
            }
            st.session_state.alerts.append(alert)
    
    if len(moderate_drifts) > 0:
        for _, row in moderate_drifts.iterrows():
            alert = {
                'timestamp': datetime.now(),
                'batch_num': st.session_state.batch_counter,
                'severity': 'MODERATE',
                'feature': row['feature'],
                'message': f"Moderate drift detected in {row['feature']}: {row['explanation']}"
            }
            st.session_state.alerts.append(alert)

    st.success(f"Batch {st.session_state.batch_counter} processed ({len(batch_df)} rows)")


def reset_monitor():
    st.session_state.batch_counter = 0
    st.session_state.batch_results = []
    st.session_state.drift_history = {}
    st.session_state.alerts = []
    st.success("Monitor reset successfully")


def render_overview():
    if not st.session_state.batch_results:
        st.info("Upload data and process batches to start monitoring")
        return
    
    latest_batch = st.session_state.batch_results[-1]
    drift_results = latest_batch['drift_results']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Batches Processed", st.session_state.batch_counter)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        n_severe = len(drift_results[drift_results['severity'] == 'SEVERE_DRIFT'])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Severe Drift", n_severe, delta=None)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        n_moderate = len(drift_results[drift_results['severity'] == 'MODERATE_DRIFT'])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Moderate Drift", n_moderate, delta=None)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        total_features = len(drift_results)
        pct_drifting = ((n_severe + n_moderate) / total_features * 100) if total_features > 0 else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("% Features Drifting", f"{pct_drifting:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)


def render_data_drift_tab():
    st.markdown("### Feature-Level Drift Analysis")
    
    if not st.session_state.batch_results:
        st.info("Process batches to see drift analysis")
        return
    
    latest_batch = st.session_state.batch_results[-1]
    drift_results = latest_batch['drift_results']
    
    st.markdown("#### Current Drift Status")
    
    def highlight_severity(val):
        if val == 'SEVERE_DRIFT':
            return 'background-color: #c62828; color: #fff;'
        elif val == 'MODERATE_DRIFT':
            return 'background-color: #f57c00; color: #fff;'
        else:
            return 'background-color: #2e7d32; color: #fff;'
    
    styled_df = drift_results.style.applymap(
        highlight_severity, 
        subset=['severity']
    ).format({
        'ks_statistic': '{:.4f}',
        'ks_p_value': '{:.4f}',
        'psi_score': '{:.4f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("#### Global Drift Overview")
    visualizer = DriftVisualizer()
    fig = visualizer.plot_global_drift_overview(drift_results)
    st.pyplot(fig)
    
    st.markdown("#### Detailed Feature Analysis")
    
    feature_to_analyze = st.selectbox(
        "Select feature to analyze",
        drift_results['feature'].tolist()
    )
    
    if feature_to_analyze:
        feature_row = drift_results[drift_results['feature'] == feature_to_analyze].iloc[0]
        is_categorical = feature_row['type'] == 'categorical'
        
        ref_data = st.session_state.reference_data[feature_to_analyze].values
        cur_data = st.session_state.current_data.iloc[:st.session_state.batch_counter * 100][feature_to_analyze].values
        
        drift_info = {
            'ks_statistic': feature_row['ks_statistic'],
            'psi_score': feature_row['psi_score'],
            'severity': feature_row['severity']
        }
        
        fig = visualizer.plot_distribution_comparison(
            ref_data, cur_data, feature_to_analyze, 
            is_categorical=is_categorical,
            drift_result=drift_info
        )
        st.pyplot(fig)
        
        if feature_to_analyze in st.session_state.drift_history:
            st.markdown("#### Drift Timeline")
            fig = visualizer.plot_drift_timeline(
                st.session_state.drift_history[feature_to_analyze],
                feature_to_analyze
            )
            st.pyplot(fig)
    
    if len(st.session_state.drift_history) > 0:
        st.markdown("#### Drift Heatmap Over Time")
        fig = visualizer.plot_drift_heatmap(st.session_state.drift_history)
        st.pyplot(fig)


def render_concept_drift_tab():
    st.markdown("### Concept Drift Analysis")
    
    if st.session_state.target_column is None:
        st.warning("No target column selected. Please select a target column in the sidebar.")
        return
    
    if not st.session_state.batch_results:
        st.info("Process batches to see concept drift analysis")
        return
    
    target_col = st.session_state.target_column
    ref_target = st.session_state.reference_data[target_col].values
    cur_target = st.session_state.current_data.iloc[:st.session_state.batch_counter * 100][target_col].values
    
    is_classification = not pd.api.types.is_numeric_dtype(st.session_state.reference_data[target_col])
    
    st.markdown(f"**Target Variable:** {target_col}")
    st.markdown(f"**Task Type:** {'Classification' if is_classification else 'Regression'}")
    
    concept_drift_result = st.session_state.drift_detector.detect_concept_drift(
        ref_target, cur_target, is_classification=is_classification
    )
    
    st.markdown("#### Concept Drift Detection Results")
    
    severity = concept_drift_result['severity']
    if isinstance(severity, DriftSeverity):
        severity = severity.value
    
    if severity == 'SEVERE_DRIFT':
        st.markdown('<div class="alert-box alert-severe">', unsafe_allow_html=True)
        st.markdown(f"**SEVERE CONCEPT DRIFT DETECTED**")
        st.markdown(f"{concept_drift_result['explanation']}")
        st.markdown('</div>', unsafe_allow_html=True)
    elif severity == 'MODERATE_DRIFT':
        st.markdown('<div class="alert-box alert-moderate">', unsafe_allow_html=True)
        st.markdown(f"**Moderate Concept Drift Detected**")
        st.markdown(f"{concept_drift_result['explanation']}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-box alert-good">', unsafe_allow_html=True)
        st.markdown(f"**No Concept Drift Detected**")
        st.markdown(f"{concept_drift_result['explanation']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    visualizer = DriftVisualizer()
    fig = visualizer.plot_concept_drift(
        ref_target, cur_target, target_col,
        is_classification=is_classification,
        drift_result=concept_drift_result
    )
    st.pyplot(fig)
    
    st.markdown("#### Understanding Concept Drift")
    st.markdown("""
    **Concept Drift** occurs when the relationship between features and the target variable changes over time.
    This is different from **Data Drift**, which only looks at feature distributions.
    
    - **Data Drift**: Changes in feature distributions (X)
    - **Concept Drift**: Changes in target distribution or P(Y|X)
    
    If concept drift is detected, it may indicate that:
    - The underlying problem is changing
    - Your model's predictions may become less accurate
    - Model retraining may be necessary
    """)


def render_alerts_tab():
    st.markdown("### 🚨 Drift Alerts")
    
    if not st.session_state.alerts:
        st.info("No alerts generated yet. Alerts will appear when drift is detected.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        severity_filter = st.multiselect(
            "Filter by severity",
            ['SEVERE', 'MODERATE'],
            default=['SEVERE', 'MODERATE']
        )
    
    filtered_alerts = [a for a in st.session_state.alerts if a['severity'] in severity_filter]
    
    st.markdown(f"**Total Alerts:** {len(filtered_alerts)}")
    
    for alert in reversed(filtered_alerts):  # Show most recent first
        if alert['severity'] == 'SEVERE':
            st.markdown('<div class="alert-box alert-severe">', unsafe_allow_html=True)
            icon = ""
        else:
            st.markdown('<div class="alert-box alert-moderate">', unsafe_allow_html=True)
            icon = ""
        
        st.markdown(f"{icon} **Batch {alert['batch_num']}** - {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown(f"Feature: **{alert['feature']}**")
        st.markdown(f"{alert['message']}")
        st.markdown('</div>', unsafe_allow_html=True)


def render_export_section():
    st.markdown("### Export Reports")
    
    if not st.session_state.batch_results:
        st.info("No data to export yet")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(" Export Drift Summary (CSV)", use_container_width=True):
            latest_batch = st.session_state.batch_results[-1]
            drift_results = latest_batch['drift_results']
            
            csv_buffer = io.StringIO()
            drift_results.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"drift_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export Full History (JSON)", use_container_width=True):
            export_data = {
                'batch_count': st.session_state.batch_counter,
                'alerts': [
                    {
                        **alert,
                        'timestamp': alert['timestamp'].isoformat()
                    }
                    for alert in st.session_state.alerts
                ],
                'drift_history': {
                    feature: history
                    for feature, history in st.session_state.drift_history.items()
                }
            }
            
            json_str = json.dumps(export_data, indent=2, default=str)
            
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"drift_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


def main():
    initialize_session_state()
    
    st.markdown('<div class="main-header">Production Data Drift Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time statistical drift detection with KS Test & PSI</div>', unsafe_allow_html=True)
    
    sidebar_controls()
    
    if st.session_state.reference_data is None:
        st.info("👈 Please upload reference data to begin monitoring")
        return
    
    render_overview()
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Drift",
        "Concept Drift",
        "Alerts",
        "Export"
    ])
    
    with tab1:
        render_data_drift_tab()
    
    with tab2:
        render_concept_drift_tab()
    
    with tab3:
        render_alerts_tab()
    
    with tab4:
        render_export_section()


if __name__ == "__main__":
    main()
