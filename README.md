# Subject-Aware Model Validation Pipeline for Repeated-Measures Data

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)

##  Overview

This pipeline addresses a critical challenge in **medical AI validation**: evaluating machine learning models on repeated-measures data without data leakage. Standard cross-validation techniques can lead to inflated performance estimates when multiple samples from the same participant appear in both training and test sets.

### The Problem
- **Data Leakage**: Traditional CV puts samples from same participant in train/test splits
- **Inflated Performance**: Models learn participant-specific patterns, not generalizable features  
- **Poor Generalization**: High validation scores don't translate to real-world performance

### Our Solution
- **Subject-Aware Validation**: Ensures no participant appears in both training and test sets
- **Multiple CV Strategies**: Compare standard vs. subject-aware approaches
- **Comprehensive Evaluation**: Performance, overfitting, and computational efficiency metrics
- **Reproducible Research**: Complete MLflow experiment tracking

##  Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd subject-aware-validation-pipeline
conda create -n validation-env python=3.11 -y
conda activate validation-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start MLflow server
mlflow ui --host 0.0.0.0 --port 5000 &

# 4. Configure your paths
cp config/config.yaml config/my_config.yaml
# Edit paths in my_config.yaml

# 5. Run pipeline
python main.py
```

##  Key Features

###  **Validation Strategies**
- **Stratified 10-Fold CV**: Traditional approach (baseline)
- **Leave-One-Participant-Out (LOPOCV)**: Ultimate subject-aware validation
- **Group 3-Fold CV**: Balanced subject-aware approach
- **Nested CV**: Proper hyperparameter tuning with subject-aware validation

###  **Supported Models**
- **Tree-based**: Random Forest, XGBoost, LightGBM, Extra Trees, Gradient Boosting
- **Linear**: Logistic Regression, LDA, QDA
- **Instance-based**: K-Nearest Neighbors
- **Ensemble**: AdaBoost

### **Comprehensive Metrics**
- **Performance**: Accuracy, F1, Precision, Recall, MCC
- **Overfitting**: Train-test gap analysis
- **Efficiency**: Training time, inference speed, model size
- **Stability**: Cross-validation variance analysis

###  **Advanced Features**
- **GPU Support**: XGBoost and LightGBM GPU acceleration
- **Parallel Processing**: Multi-core training and evaluation
- **Experiment Tracking**: Complete MLflow integration
- **Reproducibility**: Seeded random states and version tracking

## 📁 Project Structure

```
subject-aware-validation-pipeline/
├── 📄 main.py                    # Pipeline entry point
├── 📁 config/
│   └── 📄 config.yaml           # Configuration settings
├── 📁 src/                      # Core modules
│   ├── 📄 __init__.py
│   ├── 📄 data_loader.py        # Data loading and preprocessing
│   ├── 📄 models.py             # Model definitions and grids
│   ├── 📄 training.py           # Training and validation logic
│   ├── 📄 metrics.py            # Performance calculations
│   ├── 📄 mlflow_utils.py       # Experiment tracking
│   └── 📄 utils.py              # Utility functions
├── 📁 docs/                     # Documentation
├── 📁 tests/                    # Test suite
├── 📄 requirements.txt          # Dependencies
└── 📄 README.md                 # This file
```

## 🛠️ Installation

### Prerequisites
- **Python**: 3.9 or higher
- **System Memory**: 8GB+ recommended for large datasets
- **Storage**: 2GB+ for MLflow artifacts and results

### Option 1: Conda (Recommended)
```bash
# Create isolated environment
conda create -n validation-env python=3.11 -y
conda activate validation-env

# Install core dependencies
conda install numpy pandas scikit-learn=1.3.0 -y
pip install mlflow==2.9.2 joblib pyyaml

# Install optional GPU libraries
pip install xgboost lightgbm  # CPU versions
# For GPU: follow XGBoost/LightGBM GPU installation guides
```

### Option 2: Virtual Environment
```bash
python -m venv validation-env
source validation-env/bin/activate  # Linux/Mac
# validation-env\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Option 3: Docker
```bash
docker build -t validation-pipeline .
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results validation-pipeline
```

## ⚙️ Configuration

### Basic Configuration
Edit `config/config.yaml`:

```yaml
experiment:
  name: "My_Validation_Study"
  tracking_uri: "http://localhost:5000"

paths:
  data_dir: "/path/to/your/data/"
  result_dir: "/path/to/results/"

models:
  selected: ["Random Forest", "XGBoost", "Logistic Regression"]
  
file_indices: [10, 20, 30]  # Your data file indices
```

### Advanced Options
```yaml
job_control:
  outer_n_jobs: 2        # Parallel data processing
  internal_n_jobs: 4     # Model internal parallelization

hardware:
  use_gpu_xgb: true      # Enable XGBoost GPU
  use_gpu_lgbm: true     # Enable LightGBM GPU

validation:
  min_participants_lopo: 2     # Minimum for LOPOCV
  min_participants_group3: 3   # Minimum for Group 3-fold
  min_samples_per_class: 10    # Minimum for 10-fold
```

##  Usage Examples

### Basic Usage
```python
# Simple pipeline run
python main.py
```

### Custom Configuration
```python
# Use specific config file
python main.py --config config/my_custom_config.yaml
```

### Programmatic Usage
```python
from src import parallel_run, setup_mlflow

# Setup experiment
exp_id = setup_mlflow("My_Experiment", "http://localhost:5000")

# Run validation
parallel_run(
    indices=[10, 20],
    exp_id=exp_id,
    data_dir="/path/to/data",
    result_dir="/path/to/results",
    scaler_name="StandardScaler"
)
```

## 📊 Understanding Results

### MLflow UI Navigation
1. **Experiments**: Each study gets its own experiment
2. **Parent Runs**: One per data index (e.g., `idx_10_processing`)
3. **Child Runs**: One per model/CV combination (e.g., `XGBoost_LOPOCV_NestedCV_idx10`)

### Key Metrics to Monitor
- **Overall Accuracy**: Performance across all folds
- **Train-Test Gap**: Overfitting indicator
- **CV Variance**: Model stability
- **Computational Efficiency**: Resource usage

### Output Files
```
results/
├── computational_efficiency_idx10.csv     # Timing and resource usage
├── combined_performance_idx10.csv         # All metrics combined
└── LOPOCV/
    └── perf_LOPOCV_idx10.csv             # LOPOCV-specific results
```

## 🔍 Data Requirements

### Input Format
Your data should be pickle files named: `filtered_df_{index}.pkl` or `filtered_df_{index}GBC.pkl`

### Required Columns
- **Features**: Numerical columns for ML
- **target**: Binary classification labels (0/1)
- **Index**: Format `{participant_id}_{segment_order}` (e.g., `106_1`, `108_2`)

### Example Data Structure
```python
import pandas as pd

# Your dataframe should look like:
df = pd.DataFrame({
    'feature_1': [1.2, 2.3, 3.4, ...],
    'feature_2': [0.5, 1.6, 2.7, ...],
    # ... more features
    'target': [0, 1, 0, ...],  # Binary labels
}, index=['106_1', '106_2', '108_1', '108_2', ...])  # participant_segment format
```

## 🚨 Troubleshooting

### Common Issues

#### MLflow Connection
```bash
# If MLflow server isn't accessible
mlflow ui --host 0.0.0.0 --port 5000
# Check firewall settings
```

#### GPU Issues
```python
# Disable GPU if causing problems
hardware:
  use_gpu_xgb: false
  use_gpu_lgbm: false
```

#### Memory Issues
```yaml
# Reduce parallel jobs
job_control:
  outer_n_jobs: 1
  internal_n_jobs: 2
```

#### Data Loading Errors
```python
# Check data file paths and format
from src.data_loader import load_data
df = load_data("your_file.pkl")
print(df.head())
print(df.columns)
```

### Error Messages and Solutions

| Error | Solution |
|-------|----------|
| `FileNotFoundError: No such file` | Check `data_dir` path in config |
| `ValueError: single class in training` | Ensure balanced target distribution |
| `MLflow connection failed` | Start MLflow server: `mlflow ui` |
| `GPU not available` | Set `use_gpu_*: false` in config |
| `Insufficient participants` | Need ≥2 participants for LOPOCV |

## 🧪 Testing

Run the test suite to verify installation:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_pipeline.py::TestPipeline::test_build_models -v

# Check test coverage
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=html
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd subject-aware-validation-pipeline

# Create development environment
conda create -n validation-dev python=3.11 -y
conda activate validation-dev

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install
```

## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{karbalaie2024subject,
  title={Subject-Aware Model Validation for Repeated-Measures Data: A Nested Approach for Trustworthy Medical AI Applications},
  author={Karbalaie, Abdolamir and Abtahi, Farhad and others},
  journal={Journal Name},
  year={2024},
  note={Paper under review}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](https://github.com/abdkar/Nested_LOPOCV/blob/main/Guides_and_test/Documentation%20Summary.pdf)
- **Issues**: [GitHub Issues](https://github.com/abdkar/Nested_LOPOCV/issues)
- **Discussions**: [GitHub Discussions](https://github.com/abdkar/Nested_LOPOCV/discussions)
- **Email**: abdolamir.karbalaie@umu.se, farhad.abtahi@ki.se

## 🎯 Roadmap

- [ ] **v2.0**: Multi-class classification support
- [ ] **v2.1**: Time-series specific validation strategies
- [ ] **v2.2**: Automated hyperparameter optimization
- [ ] **v2.3**: Real-time monitoring dashboard
- [ ] **v3.0**: Deep learning model support

---

**Built with ❤️ for trustworthy medical AI research**
