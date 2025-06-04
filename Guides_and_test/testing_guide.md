# Testing Guide

This guide explains how to test the Subject-Aware Model Validation Pipeline using synthetic data and automated testing tools.

## 📚 Table of Contents

1. [Overview](#overview)
2. [Quick Start Testing](#quick-start-testing)
3. [Synthetic Data Generation](#synthetic-data-generation)
4. [Test Scenarios](#test-scenarios)
5. [Automated Testing](#automated-testing)
6. [Manual Testing](#manual-testing)
7. [Troubleshooting Tests](#troubleshooting-tests)

## 🎯 Overview

The testing framework provides:

- **Synthetic Data Generation**: Realistic test datasets with known properties
- **Automated Test Runner**: Complete pipeline testing with validation
- **Multiple Test Scenarios**: From minimal smoke tests to comprehensive validation
- **Result Validation**: Automated verification of outputs and MLflow tracking

### Testing Philosophy

1. **Synthetic Data**: Use controlled synthetic data to test specific scenarios
2. **Graduated Testing**: Start with minimal tests, scale up to comprehensive
3. **Validation Coverage**: Test all CV strategies, models, and edge cases
4. **Reproducibility**: All tests use fixed random seeds

## 🚀 Quick Start Testing

### 1. Generate Test Data and Run Pipeline

```bash
# Generate synthetic data and run quick test
python run_tests.py --quick

# Or run minimal test (fastest)
python run_tests.py --minimal

# Or run comprehensive test (thorough)
python run_tests.py --full
```

### 2. Manual Quick Test

```bash
# Step 1: Generate test data
python generate_test_data.py --output_dir ./test_data/

# Step 2: Start MLflow server
mlflow ui --port 5000 &

# Step 3: Run pipeline with test config
python main.py --config config_test.yaml

# Step 4: Check results
ls ./test_results/
open http://localhost:5000  # MLflow UI
```

## 🧪 Synthetic Data Generation

### Basic Data Generation

```bash
# Generate default test scenarios
python generate_test_data.py

# Generate single dataset
python generate_test_data.py --single_dataset 10 --n_participants 20

# Custom output directory
python generate_test_data.py --output_dir /path/to/test/data/
```

### Data Characteristics

The synthetic data generator creates realistic biomechanics data with:

#### **Participant Characteristics**
- Age distribution (18-65 years)
- Baseline fear levels (beta distribution)
- Individual recovery rates
- Participant-specific noise patterns

#### **Feature Types**
- **Kinematic features** (15): Joint angles, velocities
- **Kinetic features** (15): Forces, moments  
- **EMG features** (10): Muscle activation patterns
- **Balance features** (10): Stability measures

#### **Realistic Patterns**
- Fear-related movement modifications
- Recovery progression over sessions
- Age-related baseline differences
- Participant-specific characteristics that could cause data leakage

### Generated Dataset Structure

```python
# Example dataset structure
df = pd.DataFrame({
    'kinematic_00': [...],     # Joint angle 1
    'kinematic_01': [...],     # Joint angle 2
    # ... 15 kinematic features
    'kinetic_00': [...],       # Force 1  
    'kinetic_01': [...],       # Force 2
    # ... 15 kinetic features
    'emg_00': [...],           # Muscle 1 activation
    # ... 10 EMG features
    'balance_00': [...],       # Balance measure 1
    # ... 10 balance features
    'target': [0, 1, 0, ...]   # Binary classification target
}, index=[
    '106_1', '106_2', '106_3',  # Participant 106, sessions 1-3
    '108_1', '108_2', '108_3',  # Participant 108, sessions 1-3
    # ...
])
```

## 🎭 Test Scenarios

### Data Generation Scenarios

The framework includes 5 predefined scenarios:

| Scenario | Participants | Sessions | Features | Total Samples | Purpose |
|----------|-------------|----------|-----------|---------------|---------|
| 10 | 5 | 3 | 20 | 15 | Edge case testing |
| 20 | 15 | 4 | 30 | 60 | Medium balanced |
| 30 | 25 | 5 | 50 | 125 | Realistic size |
| 40 | 30 | 2 | 25 | 60 | Many participants |
| 50 | 8 | 8 | 40 | 64 | Many sessions |

### Test Configurations

#### Minimal Test
```yaml
# Fastest test for smoke testing
models: ["Logistic Regression"]
file_indices: [10]
outer_n_jobs: 1
internal_n_jobs: 1
```

#### Quick Test  
```yaml
# Balanced test for development
models: ["Random Forest", "Logistic Regression"]
file_indices: [10, 20]
outer_n_jobs: 1
internal_n_jobs: 2
```

#### Full Test
```yaml
# Comprehensive test for validation
models: ["Random Forest", "Logistic Regression", "XGBoost"]
file_indices: [10, 20, 30]
outer_n_jobs: 2
internal_n_jobs: 4
```

## 🤖 Automated Testing

### Test Runner Features

The `run_tests.py` script provides comprehensive automated testing:

```bash
# Available test modes
python run_tests.py --minimal    # ~2-5 minutes
python run_tests.py --quick      # ~10-15 minutes  
python run_tests.py --full       # ~30-60 minutes

# Individual components
python run_tests.py --generate-only   # Only generate data
python run_tests.py --validate-only   # Only validate results
```

### What Gets Tested

#### 1. **Data Generation Validation**
- Correct file formats and naming
- Valid participant IDs
- Proper index structure
- Target variable distribution
- Feature correlation patterns

#### 2. **Pipeline Execution**
- All CV strategies (LOPOCV, Group 3-Fold, 10-Fold)
- Multiple models and hyperparameter tuning
- Error handling and edge cases
- Resource usage and timeouts

#### 3. **Output Validation**
- CSV file generation and content
- MLflow experiment logging
- Model artifact storage
- Metric calculation accuracy

#### 4. **Result Analysis**
- Performance metric ranges
- CV strategy comparisons
- Model ranking consistency
- Computational efficiency

### Test Result Structure

```python
test_results = {
    'overall_success': True/False,
    'data_generation': {'success': True},
    'mlflow_server': {'success': True},
    'pipeline_runs': {
        'config_name': {
            'success': True,
            'duration': 45.2,
            'stdout': '...',
            'stderr': '...'
        }
    },
    'validation': {
        'config_results': {...},
        'summary': {
            'total_configs': 2,
            'successful_configs': 2,
            'success_rate': 1.0
        }
    }
}
```

## 🔧 Manual Testing

### Step-by-Step Manual Testing

#### 1. **Environment Setup**
```bash
# Activate environment
conda activate validation-pipeline

# Verify installation
python -c "from src import build_models; print('✅ Pipeline imports OK')"

# Check MLflow
mlflow --version
```

#### 2. **Generate Test Data**
```bash
# Create small test dataset
python generate_test_data.py \
    --single_dataset 99 \
    --n_participants 10 \
    --output_dir ./manual_test_data/

# Inspect generated data
python -c "
import pandas as pd
df = pd.read_pickle('./manual_test_data/filtered_df_99GBC.pkl')
print(f'Shape: {df.shape}')
print(f'Participants: {df.index.str.split(\"_\").str[0].nunique()}')
print(f'Target distribution: {df[\"target\"].value_counts().to_dict()}')
"
```

#### 3. **Create Test Configuration**
```yaml
# manual_test_config.yaml
experiment:
  name: "Manual_Test_Experiment"
  tracking_uri: "http://localhost:5000"

paths:
  data_dir: "./manual_test_data/"
  result_dir: "./manual_test_results/"

models:
  selected: ["Random Forest"]

file_indices: [99]

job_control:
  outer_n_jobs: 1
  internal_n_jobs: 1

hardware:
  use_gpu_xgb: false
  use_gpu_lgbm: false
```

#### 4. **Run Pipeline**
```bash
# Start MLflow server
mlflow ui --port 5000 &

# Run pipeline
python main.py --config manual_test_config.yaml

# Check outputs
ls ./manual_test_results/
```

#### 5. **Validate Results**
```bash
# Check CSV files
python -c "
import pandas as pd
import os

result_dir = './manual_test_results/'
csv_files = [f for f in os.listdir(result_dir) if f.endswith('.csv')]
print(f'Generated CSV files: {csv_files}')

for csv_file in csv_files:
    df = pd.read_csv(os.path.join(result_dir, csv_file))
    print(f'{csv_file}: {df.shape[0]} rows, {df.shape[1]} columns')
"

# Check MLflow
open http://localhost:5000
```

### Testing Specific Features

#### Test GPU Support
```bash
# Generate test data
python generate_test_data.py --single_dataset 88 --n_participants 15

# Test with GPU config
cat > gpu_test_config.yaml << EOF
experiment:
  name: "GPU_Test"
  tracking_uri: "http://localhost:5000"
paths:
  data_dir: "./"
  result_dir: "./gpu_test_results/"
models:
  selected: ["XGBoost", "LightGBM"]
file_indices: [88]
hardware:
  use_gpu_xgb: true
  use_gpu_lgbm: true
EOF

python main.py --config gpu_test_config.yaml
```

#### Test Large Dataset
```bash
# Generate larger dataset
python generate_test_data.py \
    --single_dataset 77 \
    --n_participants 50 \
    --output_dir ./large_test_data/

# Test with parallel processing
cat > parallel_test_config.yaml << EOF
experiment:
  name: "Parallel_Test"
  tracking_uri: "http://localhost:5000"
paths:
  data_dir: "./large_test_data/"
  result_dir: "./parallel_test_results/"
models:
  selected: ["Random Forest", "XGBoost", "Logistic Regression"]
file_indices: [77]
job_control:
  outer_n_jobs: 1
  internal_n_jobs: 4
EOF

python main.py --config parallel_test_config.yaml
```

#### Test Edge Cases
```bash
# Very small dataset (edge case)
python generate_test_data.py \
    --single_dataset 66 \
    --n_participants 3 \
    --output_dir ./edge_test_data/

# This should test minimum data requirements
python main.py --config edge_test_config.yaml
```

## 🚨 Troubleshooting Tests

### Common Test Failures

#### 1. **Data Generation Issues**

**Problem**: `ValueError: Invalid participant IDs`
```bash
# Solution: Check LIST_ID in data_loader.py
python -c "from src.data_loader import LIST_ID; print(f'Valid IDs: {len(LIST_ID)}')"
```

**Problem**: `Index format incorrect`
```bash
# Solution: Verify index format
python -c "
import pandas as pd
df = pd.read_pickle('test_data/filtered_df_10GBC.pkl')
print('Index samples:', df.index[:5].tolist())
print('Expected format: participant_session (e.g., 106_1)')
"
```

#### 2. **Pipeline Execution Issues**

**Problem**: `MLflow connection failed`
```bash
# Check MLflow server
curl -s http://localhost:5000 || echo "MLflow server not running"

# Start server
mlflow ui --host 0.0.0.0 --port 5000 &
sleep 5
curl -s http://localhost:5000 && echo "✅ MLflow server running"
```

**Problem**: `Insufficient data for validation strategies`
```bash
# Check data requirements
python -c "
import pandas as pd
df = pd.read_pickle('test_data/filtered_df_10GBC.pkl')
participants = df.index.str.split('_').str[0].nunique()
min_class = df['target'].value_counts().min()
print(f'Participants: {participants} (need ≥2 for LOPOCV, ≥3 for Group3Fold)')
print(f'Min samples per class: {min_class} (need ≥10 for 10Fold)')
"
```

#### 3. **Import and Environment Issues**

**Problem**: `ModuleNotFoundError: No module named 'src'`
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

**Problem**: Missing dependencies
```bash
# Check installations
python -c "
import sys
required = ['numpy', 'pandas', 'sklearn', 'mlflow', 'xgboost', 'lightgbm']
for pkg in required:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg} - run: pip install {pkg}')
"
```

### Test Performance Issues

#### Slow Tests
```bash
# Reduce test scope
models:
  selected: ["Logistic Regression"]  # Fastest model

job_control:
  outer_n_jobs: 1      # Sequential processing
  internal_n_jobs: 1   # Minimal parallelization

file_indices: [10]     # Single small dataset
```

#### Memory Issues
```bash
# Monitor memory usage
python -c "
import psutil
print(f'Available memory: {psutil.virtual_memory().available / 1e9:.1f} GB')
print('Reduce dataset size or parallelization if memory is low')
"

# Reduce memory usage
python generate_test_data.py \
    --single_dataset 55 \
    --n_participants 8 \
    --output_dir ./small_test_data/
```

### Debugging Test Failures

#### Enable Verbose Logging
```python
# Add to main.py or test scripts
import logging
logging.basicConfig(level=logging.DEBUG)

# Add to config
logging:
  level: DEBUG
  file: pipeline_debug.log
```

#### Inspect Test Data
```python
# Create data inspection script
import pandas as pd
import numpy as np

def inspect_test_data(pkl_path):
    df = pd.read_pickle(pkl_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index samples: {df.index[:5].tolist()}")
    
    # Check participant extraction
    try:
        pids = df.index.str.split('_').str[0].astype(int)
        print(f"Unique participants: {pids.nunique()}")
        print(f"Participant IDs: {sorted(pids.unique())}")
    except Exception as e:
        print(f"❌ Participant extraction failed: {e}")
    
    # Check target
    if 'target' in df.columns:
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")
        print(f"Target type: {df['target'].dtype}")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing} ({missing/df.size*100:.2f}%)")
    
    return df

# Use it
df = inspect_test_data('test_data/filtered_df_10GBC.pkl')
```

## 📊 Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Pipeline Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        
    - name: Generate test data
      run: python generate_test_data.py --output_dir ./ci_test_data/
    
    - name: Start MLflow server
      run: |
        mlflow ui --port 5000 &
        sleep 10
        
    - name: Run minimal test
      run: python run_tests.py --minimal --test-dir ./ci_