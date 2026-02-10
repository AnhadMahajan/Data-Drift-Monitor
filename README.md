# 📊 Production Data Drift Monitor

A robust, production-style data drift monitoring system built with Streamlit for real-time detection of **data drift** and **concept drift** in machine learning systems.

## 🎯 Overview

This system implements industry-standard statistical tests to monitor ML model inputs and detect when the data distribution shifts from the training baseline. It helps MLOps teams:

- **Detect Data Drift**: Monitor feature distributions using KS test and PSI
- **Detect Concept Drift**: Track changes in target variable distribution
- **Visualize Drift**: Interactive dashboards with distribution comparisons
- **Generate Alerts**: Automated warnings when drift exceeds thresholds
- **Simulate Production**: Batch processing to simulate streaming data

## 🏗️ Architecture

```
.
├── app.py              # Main Streamlit application
├── drift.py            # Drift detection logic (KS test, PSI)
├── data_loader.py      # Data ingestion and schema management
├── viz.py              # Visualization components
├── generate_sample_data.py  # Sample data generator
└── requirements.txt    # Python dependencies
```

### Core Components

**1. DriftDetector (`drift.py`)**
- Kolmogorov-Smirnov (KS) test for numerical features
- Population Stability Index (PSI) for numerical & categorical features
- Configurable thresholds for moderate and severe drift
- Concept drift detection for target variables

**2. DataLoader (`data_loader.py`)**
- Automatic schema inference (numerical vs categorical)
- Schema alignment between reference and incoming data
- Handling of missing values and unseen categories
- Batch creation for streaming simulation

**3. DriftVisualizer (`viz.py`)**
- Distribution comparison plots
- Drift timeline charts
- Global drift overview
- Drift heatmap across features and time

**4. Streamlit App (`app.py`)**
- State management with `st.session_state`
- Interactive controls for threshold tuning
- Tabbed interface for different views
- Export functionality (CSV, JSON)

## 📋 Features

### ✅ Implemented

- **Statistical Drift Detection**
  - ✓ KS test for numerical features
  - ✓ PSI for numerical and categorical features
  - ✓ Configurable thresholds (moderate/severe)
  - ✓ Per-feature drift analysis

- **Concept Drift**
  - ✓ Target distribution monitoring
  - ✓ Classification and regression support
  - ✓ Separate analysis from data drift

- **Real-Time Simulation**
  - ✓ Batch-based processing
  - ✓ State persistence across reruns
  - ✓ Incremental drift history

- **Visualization**
  - ✓ Reference vs current distribution plots
  - ✓ Drift score timelines
  - ✓ Global drift overview dashboard
  - ✓ Drift heatmap
  - ✓ Color-coded severity indicators

- **Alerts & Monitoring**
  - ✓ Automated drift alerts
  - ✓ Severity-based filtering
  - ✓ Detailed explanations

- **Export**
  - ✓ CSV export of drift summaries
  - ✓ JSON export of full history

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Generate Sample Data

```bash
# Create test datasets with various drift patterns
python generate_sample_data.py
```

This generates:
- `reference_data.csv` - Reference/training data (1000 rows)
- `current_data_no_drift.csv` - No drift (500 rows)
- `current_data_moderate_drift.csv` - Moderate drift (500 rows)
- `current_data_severe_drift.csv` - Severe drift (500 rows)
- `current_data_progressive_drift.csv` - Progressive drift (1000 rows)

### Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload Reference Data

1. Click **Browse files** in the sidebar under "Reference Dataset"
2. Upload `reference_data.csv`
3. Select target column (e.g., `churn`) for concept drift monitoring

### 2. Configure Thresholds

Adjust drift detection sensitivity:
- **KS Moderate**: Default 0.05
- **KS Severe**: Default 0.10
- **PSI Moderate**: Default 0.10
- **PSI Severe**: Default 0.25

**Interpretation:**
- KS statistic > threshold indicates significant distribution shift
- PSI > 0.1: Moderate drift, investigate
- PSI > 0.25: Severe drift, immediate action needed

### 3. Upload Incoming Data

1. Upload new data under "Incoming Data"
2. System automatically aligns schema with reference data
3. Handles missing columns and unseen categories

### 4. Process Batches

1. Set batch size (default 100 rows)
2. Click **Process Next Batch** to simulate streaming
3. View results in real-time across tabs

### 5. Monitor Drift

**Data Drift Tab:**
- View per-feature drift table
- Analyze distribution comparisons
- Track drift over time
- Examine drift heatmap

**Concept Drift Tab:**
- Monitor target variable shifts
- Distinguish from data drift
- Understand model risk

**Alerts Tab:**
- Review all drift warnings
- Filter by severity
- Track chronological history

**Export Tab:**
- Download CSV summaries
- Export JSON history
- Share reports with team

## 📊 Understanding the Metrics

### Kolmogorov-Smirnov (KS) Test

**Purpose:** Detect distribution shifts in numerical features

**Formula:** Maximum difference between cumulative distributions
```
KS = max|F_ref(x) - F_cur(x)|
```

**Interpretation:**
- 0.00 - 0.05: No significant drift
- 0.05 - 0.10: Moderate drift (monitor)
- > 0.10: Severe drift (take action)

### Population Stability Index (PSI)

**Purpose:** Measure distribution stability for both numerical and categorical features

**Formula:**
```
PSI = Σ (P_cur - P_ref) × ln(P_cur / P_ref)
```

**Interpretation:**
- < 0.10: No significant drift
- 0.10 - 0.25: Moderate drift (investigate)
- > 0.25: Severe drift (retrain model)

### Data Drift vs Concept Drift

**Data Drift (Feature Drift)**
- Changes in input feature distributions
- Example: Customer age distribution shifts older
- Impact: Model receives different inputs than training

**Concept Drift**
- Changes in target variable distribution or relationship
- Example: Churn rate increases from 20% to 40%
- Impact: Model's learned patterns no longer valid

## 🛠️ Advanced Usage

### Custom Thresholds

Tune thresholds based on your use case:

```python
# In sidebar
ks_moderate = 0.03  # More sensitive
psi_severe = 0.30   # Less sensitive
```

### Handling Unseen Categories

The system offers three strategies:
- `mark_as_other`: Replace with 'UNSEEN_CATEGORY' (default)
- `drop`: Remove rows with unseen values
- `keep`: Retain original values

### Batch Processing Strategy

**Small batches (10-50):** High granularity, more frequent checks
**Medium batches (100-200):** Balanced approach
**Large batches (500+):** Less noise, weekly/monthly checks

## 🎨 Visualization Guide

### Distribution Comparison
- **Blue histogram**: Reference distribution
- **Orange histogram**: Current distribution
- **Info box**: Drift metrics and severity

### Drift Timeline
- **Top plot**: KS statistic over time
- **Bottom plot**: PSI score over time
- **Color dots**: Severity (green/orange/red)

### Drift Heatmap
- **Rows**: Features
- **Columns**: Batch numbers
- **Colors**: Green (no drift) → Orange (moderate) → Red (severe)

## 🔧 Troubleshooting

### "Schema mismatch" errors
- Ensure incoming data has same columns as reference
- System auto-adds missing columns with NaN

### "Insufficient data" warnings
- Increase batch size
- Ensure features have sufficient non-null values

### High false positive rate
- Increase thresholds (make detection less sensitive)
- Consider feature selection (remove noisy features)

### Missing visualizations
- Check that batch has been processed
- Verify matplotlib is installed correctly

## 📚 Technical Details

### State Management

The app uses `st.session_state` to persist:
- Reference data and statistics (cached)
- Drift detection history
- Alert log
- Batch counter

**Never recomputed:**
- Reference statistics
- Historical drift scores

**Computed on demand:**
- Current batch drift
- New visualizations

### Performance Optimization

- Reference statistics cached after first computation
- Batch processing prevents memory overflow
- Incremental history tracking
- Efficient numpy/pandas operations

## 🧪 Testing with Sample Data

### Test Scenario 1: No Drift
```
Reference: reference_data.csv
Current: current_data_no_drift.csv
Expected: All features show NO_DRIFT
```

### Test Scenario 2: Moderate Drift
```
Reference: reference_data.csv
Current: current_data_moderate_drift.csv
Expected: Some features show MODERATE_DRIFT
```

### Test Scenario 3: Severe Drift
```
Reference: reference_data.csv
Current: current_data_severe_drift.csv
Expected: Most features show SEVERE_DRIFT
```

### Test Scenario 4: Progressive Drift
```
Reference: reference_data.csv
Current: current_data_progressive_drift.csv
Expected: Drift increases over batches
```

## 📈 Best Practices

1. **Set Realistic Thresholds**
   - Start conservative, adjust based on false positives
   - Different features may need different thresholds

2. **Monitor Regularly**
   - Daily for high-stakes models
   - Weekly for stable environments
   - After any upstream data changes

3. **Investigate Alerts**
   - Don't ignore moderate drift
   - Drill into feature-level details
   - Check for data quality issues first

4. **Document Findings**
   - Export reports regularly
   - Share with data science team
   - Track resolution actions

5. **Combine with Model Monitoring**
   - Drift detection complements performance monitoring
   - Check both input drift and prediction quality

## 🤝 Contributing

This is a production-style reference implementation. Key areas for extension:

- Database integration for persistent storage
- Scheduled batch processing
- Slack/email alert integration
- Multi-model monitoring
- Custom drift metrics
- A/B test integration

## 📄 License

This is an educational/reference implementation for MLOps practitioners.

## 🙏 Acknowledgments

Built using:
- **Streamlit**: Interactive web app framework
- **SciPy**: Statistical tests (KS test)
- **Pandas/NumPy**: Data manipulation
- **Matplotlib**: Visualization

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review sample data generation
3. Verify all dependencies installed
4. Check Streamlit version compatibility

---

**Built with ❤️ for MLOps Engineers**
