import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_reference_data(n_samples=1000):
    data = {
        'age': np.random.normal(35, 10, n_samples).astype(int).clip(18, 80),
        'income': np.random.lognormal(10.5, 0.5, n_samples).astype(int),
        'credit_score': np.random.normal(700, 50, n_samples).astype(int).clip(300, 850),
        'account_balance': np.random.exponential(5000, n_samples),
        'transaction_amount': np.random.gamma(2, 50, n_samples),
        
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples, p=[0.3, 0.25, 0.25, 0.2]),
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_samples, p=[0.2, 0.5, 0.3]),
        'product_type': np.random.choice(['Type_A', 'Type_B', 'Type_C', 'Type_D'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples, p=[0.5, 0.35, 0.15]),
        
        'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    return df


def generate_drifted_data(n_samples=500, drift_severity='moderate'):
    if drift_severity == 'none':
        # No drift - same distribution as reference
        return generate_reference_data(n_samples)
    
    elif drift_severity == 'moderate':
        data = {
            'age': np.random.normal(38, 11, n_samples).astype(int).clip(18, 80),  # Slightly older
            'income': np.random.lognormal(10.6, 0.55, n_samples).astype(int),
            'credit_score': np.random.normal(710, 55, n_samples).astype(int).clip(300, 850),
            'account_balance': np.random.exponential(5500, n_samples),
            'transaction_amount': np.random.gamma(2.2, 52, n_samples),
            
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples, p=[0.35, 0.25, 0.22, 0.18]),
            'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_samples, p=[0.25, 0.45, 0.3]),
            'product_type': np.random.choice(['Type_A', 'Type_B', 'Type_C', 'Type_D'], n_samples, p=[0.38, 0.32, 0.2, 0.1]),
            'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples, p=[0.55, 0.3, 0.15]),
            
            'churn': np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
        }
    
    else:
        data = {
            'age': np.random.normal(45, 15, n_samples).astype(int).clip(18, 80),  # Much older
            'income': np.random.lognormal(11.0, 0.7, n_samples).astype(int),
            'credit_score': np.random.normal(650, 70, n_samples).astype(int).clip(300, 850),
            'account_balance': np.random.exponential(8000, n_samples),
            'transaction_amount': np.random.gamma(3, 70, n_samples),
            
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples, p=[0.5, 0.2, 0.2, 0.1]),
            'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_samples, p=[0.4, 0.35, 0.25]),
            'product_type': np.random.choice(['Type_A', 'Type_B', 'Type_C', 'Type_D'], n_samples, p=[0.25, 0.25, 0.25, 0.25]),
            'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples, p=[0.7, 0.2, 0.1]),
            
            'churn': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        }
    
    df = pd.DataFrame(data)
    return df


def generate_progressive_drift_data(n_batches=10, batch_size=100):
    batches = []
    
    for i in range(n_batches):
        drift_factor = i / (n_batches - 1)  # 0 to 1
        
        data = {
            'age': np.random.normal(35 + 10*drift_factor, 10 + 5*drift_factor, batch_size).astype(int).clip(18, 80),
            'income': np.random.lognormal(10.5 + 0.5*drift_factor, 0.5 + 0.2*drift_factor, batch_size).astype(int),
            'credit_score': np.random.normal(700 - 50*drift_factor, 50 + 20*drift_factor, batch_size).astype(int).clip(300, 850),
            'account_balance': np.random.exponential(5000 + 3000*drift_factor, batch_size),
            'transaction_amount': np.random.gamma(2 + drift_factor, 50 + 20*drift_factor, batch_size),
            
            'region': np.random.choice(['North', 'South', 'East', 'West'], batch_size, 
                                      p=[0.3 + 0.2*drift_factor, 0.25 - 0.05*drift_factor, 
                                         0.25 - 0.05*drift_factor, 0.2 - 0.1*drift_factor]),
            'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], batch_size, 
                                                 p=[0.2 + 0.2*drift_factor, 0.5 - 0.1*drift_factor, 0.3 - 0.1*drift_factor]),
            'product_type': np.random.choice(['Type_A', 'Type_B', 'Type_C', 'Type_D'], batch_size, 
                                            p=[0.4 - 0.15*drift_factor, 0.3, 0.2, 0.1 + 0.15*drift_factor]),
            'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], batch_size, 
                                      p=[0.5 + 0.2*drift_factor, 0.35 - 0.15*drift_factor, 0.15 - 0.05*drift_factor]),
            
            'churn': np.random.choice([0, 1], batch_size, p=[0.8 - 0.2*drift_factor, 0.2 + 0.2*drift_factor])
        }
        
        batches.append(pd.DataFrame(data))
    
    return pd.concat(batches, ignore_index=True)


if __name__ == "__main__":
    print("Generating sample datasets...")
    
    print("Creating reference dataset...")
    ref_data = generate_reference_data(1000)
    ref_data.to_csv('reference_data.csv', index=False)
    print(f"✓ Reference data saved: {len(ref_data)} rows")
    
    print("Creating no-drift dataset...")
    no_drift = generate_drifted_data(500, 'none')
    no_drift.to_csv('current_data_no_drift.csv', index=False)
    print(f"✓ No-drift data saved: {len(no_drift)} rows")
    
    print("Creating moderate drift dataset...")
    moderate_drift = generate_drifted_data(500, 'moderate')
    moderate_drift.to_csv('current_data_moderate_drift.csv', index=False)
    print(f"✓ Moderate drift data saved: {len(moderate_drift)} rows")
    
    print("Creating severe drift dataset...")
    severe_drift = generate_drifted_data(500, 'severe')
    severe_drift.to_csv('current_data_severe_drift.csv', index=False)
    print(f"✓ Severe drift data saved: {len(severe_drift)} rows")
    
    print("Creating progressive drift dataset...")
    progressive = generate_progressive_drift_data(10, 100)
    progressive.to_csv('current_data_progressive_drift.csv', index=False)
    print(f"✓ Progressive drift data saved: {len(progressive)} rows")
    
    print("\n✅ All sample datasets generated successfully!")
    print("\nDataset Summary:")
    print("- reference_data.csv: Baseline reference data (1000 rows)")
    print("- current_data_no_drift.csv: Same distribution as reference (500 rows)")
    print("- current_data_moderate_drift.csv: Moderate drift (500 rows)")
    print("- current_data_severe_drift.csv: Severe drift (500 rows)")
    print("- current_data_progressive_drift.csv: Gradual drift over time (1000 rows)")
