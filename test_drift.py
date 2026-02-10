import pandas as pd
import numpy as np
from data_loader import DataLoader
from drift import DriftDetector

def test_drift_detection():
    print("=" * 60)
    print("Testing Data Drift Monitor Components")
    print("=" * 60)
    
    print("\n1. Initializing components...")
    loader = DataLoader(categorical_threshold=10)
    detector = DriftDetector(
        ks_threshold_moderate=0.05,
        ks_threshold_severe=0.10,
        psi_threshold_moderate=0.10,
        psi_threshold_severe=0.25
    )
    print("✓ Components initialized")
    
    print("\n2. Loading reference data...")
    ref_df = pd.read_csv('reference_data.csv')
    print(f"✓ Loaded {len(ref_df)} rows, {len(ref_df.columns)} columns")
    
    print("\n3. Inferring schema...")
    loader.set_reference_schema(ref_df, exclude_columns=['churn'])
    numerical, categorical = loader.get_feature_lists()
    print(f"✓ Numerical features: {numerical}")
    print(f"✓ Categorical features: {categorical}")
    
    print("\n4. Testing with NO DRIFT data...")
    no_drift_df = pd.read_csv('current_data_no_drift.csv')
    aligned_df = loader.align_schema(no_drift_df)
    
    drift_results = detector.detect_feature_drift(
        ref_df, aligned_df, numerical, categorical
    )
    
    severe_count = len(drift_results[drift_results['severity'] == 'SEVERE_DRIFT'])
    moderate_count = len(drift_results[drift_results['severity'] == 'MODERATE_DRIFT'])
    no_drift_count = len(drift_results[drift_results['severity'] == 'NO_DRIFT'])
    
    print(f"✓ Results: {no_drift_count} no drift, {moderate_count} moderate, {severe_count} severe")
    assert severe_count == 0, "Should have no severe drift in no-drift dataset"
    
    print("\n5. Testing with MODERATE DRIFT data...")
    mod_drift_df = pd.read_csv('current_data_moderate_drift.csv')
    aligned_df = loader.align_schema(mod_drift_df)
    
    drift_results = detector.detect_feature_drift(
        ref_df, aligned_df, numerical, categorical
    )
    
    severe_count = len(drift_results[drift_results['severity'] == 'SEVERE_DRIFT'])
    moderate_count = len(drift_results[drift_results['severity'] == 'MODERATE_DRIFT'])
    no_drift_count = len(drift_results[drift_results['severity'] == 'NO_DRIFT'])
    
    print(f"✓ Results: {no_drift_count} no drift, {moderate_count} moderate, {severe_count} severe")
    assert moderate_count > 0, "Should have some moderate drift"
    
    print("\n6. Testing with SEVERE DRIFT data...")
    sev_drift_df = pd.read_csv('current_data_severe_drift.csv')
    aligned_df = loader.align_schema(sev_drift_df)
    
    drift_results = detector.detect_feature_drift(
        ref_df, aligned_df, numerical, categorical
    )
    
    severe_count = len(drift_results[drift_results['severity'] == 'SEVERE_DRIFT'])
    moderate_count = len(drift_results[drift_results['severity'] == 'MODERATE_DRIFT'])
    no_drift_count = len(drift_results[drift_results['severity'] == 'NO_DRIFT'])
    
    print(f"✓ Results: {no_drift_count} no drift, {moderate_count} moderate, {severe_count} severe")
    assert severe_count > 0, "Should have some severe drift"
    
    print("\n7. Detailed Results for SEVERE DRIFT:")
    print(drift_results[['feature', 'type', 'ks_statistic', 'psi_score', 'severity']].to_string())
    
    print("\n8. Testing concept drift detection...")
    ref_target = ref_df['churn'].values
    sev_target = sev_drift_df['churn'].values
    
    concept_result = detector.detect_concept_drift(ref_target, sev_target, is_classification=True)
    print(f"✓ Concept drift severity: {concept_result['severity'].value}")
    print(f"✓ PSI Score: {concept_result['psi_score']:.4f}")
    
    print("\n9. Testing batch creation...")
    batches = loader.create_batches(sev_drift_df, batch_size=100)
    print(f"✓ Created {len(batches)} batches")
    print(f"✓ First batch: {len(batches[0])} rows")
    print(f"✓ Last batch: {len(batches[-1])} rows")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_drift_detection()
