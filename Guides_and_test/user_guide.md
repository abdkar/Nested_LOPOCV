# User Guide

This comprehensive guide walks you through using the Subject-Aware Model Validation Pipeline from basic usage to advanced customization.

## 📚 Table of Contents

1. [Getting Started](#getting-started)
2. [Data Preparation](#data-preparation)
3. [Configuration](#configuration)
4. [Running Experiments](#running-experiments)
5. [Understanding Results](#understanding-results)
6. [Advanced Usage](#advanced-usage)
7. [Best Practices](#best-practices)
8. [FAQ](#faq)

## 🚀 Getting Started

### Prerequisites Checklist

Before starting, ensure you have:
- [ ] Completed installation (see [INSTALLATION.md](INSTALLATION.md))
- [ ] MLflow server running (`mlflow ui --port 5000`)
- [ ] Your data in the correct format
- [ ] Configured `config/config.yaml` with your paths

### Your First Experiment

1. **Start with sample data**:
```bash
# Copy example configuration
cp config/config.yaml config/my_first_experiment.yaml
```

2. **Edit basic settings**:
```yaml
experiment:
  name: "My_First_Validation_Study"
  
paths:
  data_dir: "/path/to/your/data/"
  result_dir: "/path/to/results/"
  
models:
  selected: ["Random Forest", "Logistic Regression"]  # Start simple
  
file_indices: [10]  # Single dataset first
```

3. **Run the pipeline**:
```bash
python main.py
```

4. **View results**:
   - Open browser to `http://localhost:5000`
   - Navigate to your experiment
   - Explore the generated metrics and artifacts

## 📊 Data Preparation

### Input Data Format

Your data must be in pickle format with specific structure:

#### File Naming Convention
- `filtered_df_{index}.pkl` or `filtered_df_{index}GBC.pkl`
- Examples: `filtered_df_10.pkl`, `filtered_df_20GBC.pkl`

#### Required Data Structure
```python
import pandas as pd
import numpy as np

# Example dataframe structure
df = pd.DataFrame({
    # Features (any number of numerical columns)
    'feature_1': [1.2, 2.3, 3.4, 1.5, 2.8],
    'feature_2': [0.5, 1.6, 2.7, 0.8, 1.9],
    # ... more features
    
    # Target variable (binary: 0 or 1)
    'target': [0, 1, 0, 1, 0],
    
}, index=[
    # Index format: {participant_id}_{segment_order}
    '106_1', '106_2', '108_1', '108_2', '110_1'
])

# Save as pickle
df.to_pickle('filtered_df_10.pkl')
```

#### Participant ID Requirements
Participant IDs must be from this predefined list:
```python
VALID_PARTICIPANT_IDS = [
    106, 108, 109, 110, 114, 115, 119, 125, 133, 135, 138, 148, 150,
    151, 152, 155, 156, 157, 164, 190, 191, 192, 197, 51102, 51103,
    # ... (see complete list in src/data_loader.py)
]
```

### Data Quality Checks

Before running experiments, verify your data:

```python
from src.data_loader import load_data
import pandas as pd

# Load and inspect your data
df = load_data('path/to/your/filtered_df_10.pkl')

print("📊 Data Summary:")
print(f"Shape: {df.shape}")
print(f"Participants: {df['Test_ID'].nunique()}")
print(f"Target distribution: {df['target'].value_counts()}")
print(f"Features: {df.shape[1] - 3}")  # Excluding target, Test_ID, seg_order

# Check minimum requirements
participants = df['Test_ID'].nunique()
min_class_samples = df['target'].value_counts().min()

print("\n✅ Validation Strategy Feasibility:")
print(f"LOPOCV: {'✅' if participants >= 2 else '❌'} (need ≥2 participants, have {participants})")
print(f"Group 3-Fold: {'✅' if participants >= 3 else '❌'} (need ≥3 participants, have {participants})")
print(f"10-Fold: {'✅' if min_class_samples >= 10 else '❌'} (need ≥10 samples/class, have {min_class_samples})")
```

## ⚙️ Configuration

### Basic Configuration

The `config/config.yaml` file controls all pipeline behavior:

```yaml
# Experiment identification
experiment:
  name: "ACL_Recovery_Study_v1"           # MLflow experiment name
  tracking_uri: "http://localhost:5000"   # MLflow server URL

# File paths
paths:
  data_dir: "/home/user/data/"            # Where your .pkl files are
  result_dir: "/home/user/results/"       # Where to save CSV outputs

# Data preprocessing
scaling:
  scaler: "StandardScaler"                # StandardScaler|MinMaxScaler|Normalizer

# Model selection
models:
  selected:                               # Leave empty for all models
    - "Random Forest"
    - "XGBoost" 
    - "Logistic Regression"

# Data files to process
file_indices: [10, 20, 30]               # Corresponds to filtered_df_10.pkl, etc.
```

### Advanced Configuration

```yaml
# Parallel processing control
job_control:
  outer_n_jobs: 2        # Parallel processing of different datasets
  internal_n_jobs: 4     # Parallel processing within models (GridSearchCV, etc.)

# Hardware acceleration
hardware:
  use_gpu_xgb: true      # Enable XGBoost GPU (requires CUDA)
  use_gpu_lgbm: true     # Enable LightGBM GPU (requires CUDA)

# Validation strategy thresholds
validation:
  min_participants_lopo: 2     # Minimum participants for LOPOCV
  min_participants_group3: 3   # Minimum participants for Group 3-Fold
  min_samples_per_class: 10    # Minimum samples per class for 10-Fold

# Reproducibility
random_state:
  global_seed: 42        # Global random seed
  cv_seed: 42           # Cross-validation random seed
  model_seed: 42        # Model random seed
```

### Configuration Validation

Validate your configuration before running:

```python
import yaml
from pathlib import Path

# Load and validate config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Check required fields
required_fields = [
    'experiment.name', 'experiment.tracking_uri',
    'paths.data_dir', 'paths.result_dir',
    'file_indices'
]

print("🔍 Configuration Validation:")
for field in required_fields:
    keys = field.split('.')
    value = config
    try:
        for key in keys:
            value = value[key]
        print(f"✅ {field}: {value}")
    except KeyError:
        print(f"❌ {field}: MISSING")

# Check data files exist
data_dir = Path(config['paths']['data_dir'])
for idx in config['file_indices']:
    file_paths = [
        data_dir / f"filtered_df_{idx}.pkl",
        data_dir / f"filtered_df_{idx}GBC.pkl"
    ]
    exists = any(p.exists() for p in file_paths)
    print(f"{'✅' if exists else '❌'} Data file for index {idx}")
```

## 🏃 Running Experiments

### Basic Execution

```bash
# Simple run with default config
python main.py

# Run with custom config
python main.py --config config/my_experiment.yaml

# Run with environment variables
MLFLOW_TRACKING_URI=http://remote-server:5000 python main.py
```

### Monitoring Progress

The pipeline provides detailed console output:

```
🚀 Launching pipeline → Experiment='My_Study' | Indices=[10, 20]

===== Processing data index 10 (PID 12345) =====
Loading data from: /path/to/filtered_df_10GBC.pkl
  Parent MLflow run for index 10: abc123-def456-ghi789

    Running CV Strategy: LOPOCV for index 10
      Starting MLflow run: RandomForest_LOPOCV_Fixed_idx10
      Starting MLflow run: RandomForest_LOPOCV_NestedCV_idx10
      Starting MLflow run: XGBoost_LOPOCV_Fixed_idx10
      Starting MLflow run: XGBoost_LOPOCV_NestedCV_idx10

    Running CV Strategy: 10Fold for index 10
      Starting MLflow run: RandomForest_10Fold_Fixed_idx10
      ...

===== Completed data index 10 =====
✅ All indices processed.
🏁 Pipeline finished in 245.67 seconds.
```

### Parallel Execution

For multiple datasets, the pipeline can process them in parallel:

```yaml
# config.yaml
job_control:
  outer_n_jobs: 3  # Process 3 datasets simultaneously

file_indices: [10, 20, 30, 40, 50, 60]  # 6 datasets total
```

### Resuming Interrupted Experiments

MLflow automatically handles experiment resumption. If your pipeline is interrupted:

1. **Check MLflow UI** to see which runs completed
2. **Modify file_indices** in config to exclude completed indices
3. **Rerun** with same experiment name

```yaml
# Original run
file_indices: [10, 20, 30, 40]

# After interruption (assuming 10, 20 completed)
file_indices: [30, 40]
```

## 📈 Understanding Results

### MLflow UI Navigation

#### Experiment Structure
```
📁 My_Validation_Study
├── 📄 idx_10_processing (Parent Run)
│   ├── 📄 RandomForest_LOPOCV_Fixed_idx10 (Child Run)
│   ├── 📄 RandomForest_LOPOCV_NestedCV_idx10 (Child Run)
│   ├── 📄 XGBoost_LOPOCV_Fixed_idx10 (Child Run)
│   └── ... (more child runs)
├── 📄 idx_20_processing (Parent Run)
│   └── ... (child runs for idx 20)
```

#### Key Metrics Explained

**Performance Metrics:**
- `overall_accuracy`: Accuracy across all CV folds
- `overall_f1_score`: F1-score across all CV folds  
- `avg_fold_test_accuracy`: Average test accuracy per fold
- `avg_fold_train_accuracy`: Average training accuracy per fold

**Overfitting Indicators:**
- `Train-Test Gap`: Difference between training and test performance
- `Overfitting Indicator`: "+" if gap > 10%, "-" otherwise

**Efficiency Metrics:**
- `avg_fold_train_time_s`: Average training time per fold
- `avg_sample_infer_time_s`: Average inference time per sample
- `avg_fold_model_size_mb`: Average model size in MB

### CSV Output Files

The pipeline generates several CSV files in your `result_dir`:

#### 1. Performance Summary
`combined_performance_idx{N}.csv` - All models and CV strategies for dataset N

| Model | CV_Strategy | Tuning | Accuracy | F1_Score | Precision | Recall | MCC |
|-------|-------------|---------|----------|----------|-----------|---------|-----|
| Random Forest | LOPOCV | Fixed | 0.85 | 0.83 | 0.87 | 0.79 | 0.71 |
| Random Forest | LOPOCV | NestedCV | 0.82 | 0.80 | 0.84 | 0.76 | 0.65 |

#### 2. Computational Efficiency
`computational_efficiency_idx{N}.csv` - Resource usage analysis

| Model | Technique | Train_Time_s | Inference_Time_s | Model_Size_MB |
|-------|-----------|--------------|------------------|---------------|
| Random Forest | LOPOCV - Fixed | 2.34 | 0.001 | 15.2 |
| XGBoost | LOPOCV - NestedCV | 45.67 | 0.0005 | 8.9 |

#### 3. Detailed Fold Results  
`{CV_Strategy}/perf_{CV_Strategy}_idx{N}.csv` - Per-fold detailed metrics

### Comparing Validation Strategies

Use these guidelines to interpret results:

#### 1. **Standard vs. Subject-Aware Performance Gap**
```python
# Large gap indicates data leakage in standard CV
standard_acc = 0.95  # 10-Fold CV accuracy
lopocv_acc = 0.78    # LOPOCV accuracy
gap = standard_acc - lopocv_acc  # 0.17 = significant data leakage!
```

#### 2. **Model Ranking Stability**
Compare model rankings across CV strategies:
- **Stable ranking**: Same best models across all CV strategies
- **Unstable ranking**: Different best models → be cautious of overfitting

#### 3. **Computational Trade-offs**
- **LOPOCV**: Most realistic but computationally expensive
- **Group 3-Fold**: Good balance of realism and efficiency  
- **10-Fold**: Fastest but potentially overoptimistic

### Visualization and Analysis

Create visualizations from your results:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results = pd.read_csv('results/combined_performance_idx10.csv')

# Performance comparison across CV strategies
plt.figure(figsize=(12, 6))
sns.boxplot(data=results, x='CV_Strategy', y='Test_Accuracy', hue='Model')
plt.title('Model Performance Across Validation Strategies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Train-test gap analysis
plt.figure(figsize=(10, 6))
results['Train_Test_Gap'] = results['Training_Accuracy'] - results['Test_Accuracy']
sns.barplot(data=results, x='Model', y='Train_Test_Gap', hue='CV_Strategy')
plt.title('Overfitting Analysis: Train-Test Gap by Model and CV Strategy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 🔧 Advanced Usage

### Custom Model Integration

Add your own models to the pipeline:

```python
# In src/models.py, add to build_models function:

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def build_models(internal_n_jobs=1, use_gpu_xgb=False, use_gpu_lgbm=False, selected_model_keys=None):
    all_models = {
        # ... existing models ...
        
        # Add custom models
        "SVM": SVC(random_state=42, probability=True),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            random_state=42,
            max_iter=1000
        ),
    }
    
    # Also add to MODEL_PARAM_GRIDS:
    MODEL_PARAM_GRIDS.update({
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        },
        "Neural Network": {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "learning_rate_init": [0.001, 0.01]
        }
    })
```

### Custom Metrics

Add custom evaluation metrics:

```python
# In src/metrics.py, modify metric_row function:

from sklearn.metrics import roc_auc_score, balanced_accuracy_score

def metric_row(pred, y_true, pid, train_acc, cv_acc):
    # ... existing code ...
    
    # Add custom metrics
    balanced_acc = balanced_accuracy_score(y_true, pred) if len(y_true) > 0 else np.nan
    
    try:
        auc_score = roc_auc_score(y_true, pred) if len(np.unique(y_true)) > 1 else np.nan
    except ValueError:
        auc_score = np.nan
    
    # Add to return dictionary
    result_dict.update({
        "Balanced_Accuracy": balanced_acc,
        "AUC_Score": auc_score,
    })
    
    return result_dict
```

### Batch Processing

Process multiple datasets with different configurations:

```python
# batch_runner.py
import yaml
from pathlib import Path
from src.training import parallel_run
from src.mlflow_utils import setup_mlflow

# Define batch configurations
batch_configs = [
    {
        "name": "ACL_Study_StandardScaling",
        "scaler": "StandardScaler",
        "models": ["Random Forest", "XGBoost"],
        "indices": [10, 20, 30]
    },
    {
        "name": "ACL_Study_MinMaxScaling", 
        "scaler": "MinMaxScaler",
        "models": ["Random Forest", "XGBoost"],
        "indices": [10, 20, 30]
    }
]

# Run batch experiments
for config in batch_configs:
    print(f"Running experiment: {config['name']}")
    
    exp_id = setup_mlflow(config['name'], "http://localhost:5000")
    
    parallel_run(
        indices=config['indices'],
        exp_id=exp_id,
        data_dir="/path/to/data/",
        result_dir=f"/path/to/results/{config['name']}/",
        scaler_name=config['scaler'],
        selected_models_list=config['models'],
        n_jobs_outer=1,
        internal_n_jobs=4,
        use_gpu_xgb=False,
        use_gpu_lgbm=False
    )
```

### Remote MLflow Server

Use a remote MLflow server for team collaboration:

```yaml
# config.yaml
experiment:
  tracking_uri: "http://mlflow-server.company.com:5000"
  
# Or set environment variable
export MLFLOW_TRACKING_URI=http://mlflow-server.company.com:5000
```

## 📋 Best Practices

### 1. Experiment Organization

```yaml
# Use descriptive experiment names
experiment:
  name: "ACL_Recovery_Study_v2_StandardizedFeatures_20241201"
```

### 2. Reproducibility

```yaml
# Always set random seeds
random_state:
  global_seed: 42
  cv_seed: 42
  model_seed: 42

# Document your configuration
# Use version control for config files
git add config/experiment_v2.yaml
git commit -m "Add config for ACL study v2"
```

### 3. Resource Management

```yaml
# Start conservative with parallelization
job_control:
  outer_n_jobs: 1      # Increase gradually
  internal_n_jobs: 2   # Monitor CPU usage

# Monitor memory usage
# Use smaller datasets for initial testing
file_indices: [10]  # Start with one dataset
```

### 4. Incremental Development

1. **Start Simple**: Use 2-3 fast models (Logistic Regression, Random Forest)
2. **Validate Setup**: Ensure everything works with small dataset
3. **Scale Up**: Add more models and datasets gradually
4. **Optimize**: Tune parallelization and hardware usage

### 5. Result Validation

Always sanity-check your results:

- **Performance ranges**: Are accuracies reasonable for your domain?
- **CV consistency**: Do different CV strategies show logical relationships?
- **Model behavior**: Do complex models show higher train-test gaps?
- **Computational efficiency**: Are timing results reasonable?

## ❓ FAQ

### General Questions

**Q: How long does a typical experiment take?**
A: Depends on data size and model complexity. For 1000 samples with 50 features:
- Single model/dataset: 2-10 minutes
- Full pipeline (10 models, 3 CV strategies): 30-120 minutes

**Q: Can I stop and resume experiments?**
A: Yes, MLflow tracks completed runs. Modify `file_indices` to exclude completed datasets.

**Q: How much disk space do results require?**
A: Typical space usage per experiment:
- MLflow artifacts: 50-200 MB
- CSV files: 5-20 MB
- Model files: 10-100 MB per model

### Technical Questions

**Q: My GPU isn't being used. Why?**
A: Check:
1. CUDA installation: `nvidia-smi`  
2. GPU-enabled packages: `pip install xgboost[gpu]`
3. Configuration: `use_gpu_xgb: true`

**Q: I'm getting memory errors. What should I do?**
A: Try:
1. Reduce parallelization: `outer_n_jobs: 1, internal_n_jobs: 1`
2. Use smaller datasets initially
3. Close other applications
4. Consider cloud computing for large experiments

**Q: Results seem inconsistent across runs. Why?**
A: Ensure:
1. Random seeds are set in configuration
2. Same data preprocessing steps
3. Same model parameters
4. Check for data randomization issues

### Data Questions

**Q: My participant IDs aren't recognized. What should I do?**
A: Update the `LIST_ID` in `src/data_loader.py` with your participant IDs.

**Q: Can I use multi-class classification?**
A: Currently supports binary classification only. Multi-class support is planned for v2.0.

**Q: What if I have missing values?**
A: Handle missing values before creating pickle files. The pipeline doesn't perform imputation.

### Troubleshooting

**Q: MLflow server won't start. What's wrong?**
A: Try:
```bash
# Kill existing processes
pkill -f "mlflow ui"

# Start with specific host/port
mlflow ui --host 0.0.0.0 --port 5000

# Check port availability
netstat -tlnp | grep 5000
```

**Q: Pipeline fails with import errors. How to fix?**
A: Ensure:
1. Virtual environment is activated
2. All dependencies installed: `pip install -r requirements.txt`
3. Python path includes project root: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

Need more help? Check our [GitHub Issues](https://github.com/abdkar/Nested_LOPOCV/issues) or create a new issue with your specific problem!
