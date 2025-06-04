# Installation Guide

This guide provides detailed installation instructions for the Subject-Aware Model Validation Pipeline.

## 📋 System Requirements

### Minimum Requirements
- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: 3.9 or higher
- **RAM**: 4GB (8GB+ recommended for large datasets)
- **Storage**: 2GB free space for dependencies and artifacts
- **CPU**: Multi-core recommended for parallel processing

### Recommended Requirements
- **RAM**: 16GB+ for large-scale experiments
- **CPU**: 8+ cores for optimal parallel processing
- **GPU**: NVIDIA GPU with CUDA support (optional, for XGBoost/LightGBM acceleration)
- **Storage**: SSD for faster I/O operations

## 🐍 Python Environment Setup

### Option 1: Conda (Recommended)

Conda provides the most reliable environment management and dependency resolution.

```bash
# Install Miniconda (if not already installed)
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Create new environment
conda create -n validation-pipeline python=3.11 -y

# Activate environment
conda activate validation-pipeline

# Verify Python version
python --version  # Should show Python 3.11.x
```

### Option 2: Virtual Environment

```bash
# Create virtual environment
python -m venv validation-pipeline

# Activate environment
# On Linux/macOS:
source validation-pipeline/bin/activate
# On Windows:
validation-pipeline\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Option 3: Poetry (For Developers)

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate environment
poetry shell
```

## 📦 Core Dependencies Installation

### Method 1: Requirements File (Recommended)

```bash
# Install all dependencies at once
pip install -r requirements.txt

# Verify installation
python -c "import sklearn, mlflow, pandas, numpy; print('All core packages installed successfully')"
```

### Method 2: Manual Installation

```bash
# Core scientific computing
pip install numpy==1.24.3 pandas==2.0.3

# Machine learning
pip install scikit-learn==1.3.0

# Experiment tracking
pip install mlflow==2.9.2

# Parallel processing and utilities
pip install joblib==1.3.2 pyyaml==6.0.1

# Optional visualization
pip install matplotlib==3.7.2 seaborn==0.12.2
```

### Method 3: Conda Installation

```bash
# Install from conda-forge (often more stable)
conda install -c conda-forge numpy pandas scikit-learn joblib pyyaml -y

# MLflow from pip (better integration)
pip install mlflow==2.9.2
```

## 🚀 Optional Dependencies

### GPU Support (XGBoost & LightGBM)

#### For CPU-only (Default)
```bash
pip install xgboost==1.7.6 lightgbm==4.0.0
```

#### For GPU Support (NVIDIA CUDA required)

**Prerequisites:**
- NVIDIA GPU with compute capability 3.5+
- CUDA Toolkit 11.0+ installed
- cuDNN library

```bash
# Install CUDA-enabled versions
pip install xgboost[gpu]==1.7.6
pip install lightgbm[gpu]==4.0.0

# Verify GPU availability
python -c "import xgboost as xgb; print('XGBoost GPU:', xgb.dask.get_device_info())"
```

### Development Dependencies

```bash
# Testing framework
pip install pytest==7.4.0 pytest-cov==4.1.0

# Code formatting and linting
pip install black==23.7.0 flake8==6.0.0 isort==5.12.0

# Pre-commit hooks
pip install pre-commit==3.3.3
```

## 🐳 Docker Installation

### Using Pre-built Image

```bash
# Pull image
docker pull your-registry/validation-pipeline:latest

# Run container
docker run -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -p 5000:5000 \
  your-registry/validation-pipeline:latest
```

### Building from Source

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose MLflow port
EXPOSE 5000

# Default command
CMD ["python", "main.py"]
```

```bash
# Build image
docker build -t validation-pipeline .

# Run with volume mounts
docker run -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -p 5000:5000 \
  validation-pipeline
```

## 🧪 Installation Verification

### Quick Test Script

Create a file `test_installation.py`:

```python
#!/usr/bin/env python3
"""Test script to verify installation."""

import sys
import subprocess
from importlib import import_module

# Required packages
REQUIRED_PACKAGES = [
    'numpy', 'pandas', 'sklearn', 'mlflow', 
    'joblib', 'yaml', 'pathlib'
]

OPTIONAL_PACKAGES = [
    'xgboost', 'lightgbm', 'matplotlib', 'seaborn'
]

def test_package_import(package_name, optional=False):
    """Test if a package can be imported."""
    try:
        import_module(package_name)
        print(f"✅ {package_name}: OK")
        return True
    except ImportError as e:
        if optional:
            print(f"⚠️  {package_name}: Not installed (optional)")
        else:
            print(f"❌ {package_name}: FAILED - {e}")
        return not optional

def test_python_version():
    """Test Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}: OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro}: Requires 3.9+")
        return False

def test_mlflow_server():
    """Test if MLflow server can start."""
    try:
        import mlflow
        # This just tests if mlflow can be imported, not if server starts
        print("✅ MLflow: Available")
        print("ℹ️  To test MLflow server: mlflow ui --port 5000")
        return True
    except Exception as e:
        print(f"❌ MLflow: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Testing Installation...\n")
    
    all_passed = True
    
    # Test Python version
    all_passed &= test_python_version()
    print()
    
    # Test required packages
    print("📦 Testing Required Packages:")
    for package in REQUIRED_PACKAGES:
        all_passed &= test_package_import(package)
    print()
    
    # Test optional packages
    print("📦 Testing Optional Packages:")
    for package in OPTIONAL_PACKAGES:
        test_package_import(package, optional=True)
    print()
    
    # Test MLflow
    print("🔧 Testing MLflow:")
    all_passed &= test_mlflow_server()
    print()
    
    # Final result
    if all_passed:
        print("🎉 Installation verification passed!")
        print("Run 'python main.py' to start the pipeline.")
    else:
        print("❌ Installation verification failed!")
        print("Please fix the issues above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Run the test:
```bash
python test_installation.py
```

### Manual Verification

```bash
# Test core imports
python -c "
import numpy as np
import pandas as pd
import sklearn
import mlflow
print('Core packages: OK')
"

# Test MLflow server startup
mlflow ui --port 5000 &
sleep 5
curl -s http://localhost:5000 > /dev/null && echo "MLflow server: OK" || echo "MLflow server: FAILED"
pkill -f "mlflow ui"

# Test pipeline import
python -c "
from src import build_models, load_data, metric_row
print('Pipeline modules: OK')
"
```

## 🔧 Environment Configuration

### Setting Environment Variables

Create a `.env` file in your project root:

```bash
# .env file
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=validation_pipeline
PYTHONPATH=${PYTHONPATH}:$(pwd)
OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0
```

Load environment variables:

```bash
# Linux/macOS
export $(cat .env | xargs)

# Or use python-dotenv
pip install python-dotenv
```

### Jupyter Notebook Setup (Optional)

```bash
# Install Jupyter
pip install jupyter ipykernel

# Add environment to Jupyter
python -m ipykernel install --user --name=validation-pipeline --display-name="Validation Pipeline"

# Start Jupyter
jupyter notebook
```

## 🚨 Troubleshooting

### Common Installation Issues

#### Issue: `pip install` fails with permission errors
**Solution:**
```bash
# Use --user flag
pip install --user -r requirements.txt

# Or create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

#### Issue: MLflow installation fails
**Solution:**
```bash
# Try installing with specific version
pip install mlflow==2.9.2 --no-cache-dir

# If still fails, install dependencies separately
pip install click flask gunicorn numpy pandas pyyaml
pip install mlflow==2.9.2
```

#### Issue: XGBoost/LightGBM GPU installation fails
**Solution:**
```bash
# Install CPU versions first
pip install xgboost lightgbm

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install GPU versions only if CUDA is available
pip uninstall xgboost lightgbm
pip install xgboost[gpu] lightgbm[gpu]
```

#### Issue: Import errors for local modules
**Solution:**
```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or create __init__.py files
touch src/__init__.py
```

### System-Specific Notes

#### macOS
```bash
# Install Xcode command line tools (if needed)
xcode-select --install

# Use Homebrew for system dependencies
brew install python@3.11
```

#### Windows
```bash
# Use Anaconda Prompt for best compatibility
# Install Microsoft C++ Build Tools if needed
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### Linux
```bash
# Install build essentials
sudo apt-get update
sudo apt-get install python3-dev build-essential

# For CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

## ✅ Next Steps

After successful installation:

1. **Configure the pipeline**: Edit `config/config.yaml`
2. **Prepare your data**: Follow the data format requirements
3. **Start MLflow server**: `mlflow ui --port 5000`
4. **Run the pipeline**: `python main.py`
5. **View results**: Open `http://localhost:5000` in your browser

For detailed usage instructions, see the [User Guide](docs/USER_GUIDE.md).

## 🆘 Getting Help

If you encounter issues:

1. Check the [Troubleshooting section](#🚨-troubleshooting)
2. Search [existing issues](https://github.com/your-repo/issues)
3. Create a [new issue](https://github.com/your-repo/issues/new) with:
   - Your operating system and Python version
   - Complete error message
   - Steps to reproduce the problem
   - Output of `pip list` and `python --version`