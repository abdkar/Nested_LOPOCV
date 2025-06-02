
Markdown
# Subject-Aware Model Validation Pipeline for Repeated-Measures Data

## 1\. Overview

This project provides the codebase for the study titled: **"Subject-Aware Model Validation for Repeated-Measures Data: A Nested Approach for Trustworthy Medical AI Applications."**

The primary challenge addressed is the evaluation of machine learning (ML) models on repeated-measures data, common in clinical biomechanics and digital health. Such data often exhibit dependencies between trials from the same participant, which can lead to inflated performance estimates if standard cross-validation (CV) techniques are used due to data leakage.

This pipeline was developed to systematically compare four distinct validation strategies in the context of predicting fear of re-injury from a dataset of 623 movement trials from 72 individuals post-ACL reconstruction. The strategies evaluated are:

  * Stratified 10-Fold CV
  * Leave-One-Participant-Out CV (LOPOCV)
  * Group 3-Fold CV
  * A nested LOPOCV (outer loop) with Group 3-Fold CV (inner loop for hyperparameter tuning) framework.

The study utilizes this pipeline to evaluate ten different classifiers across various metrics, including classification performance, the train-test gap, model ranking stability, and computational runtime. The findings aim to highlight the necessity of subject-aware evaluation methodologies for developing robust and trustworthy ML models in clinical and behavioral settings, thereby supporting transparent and reproducible evaluation standards in medical AI applications. This codebase facilitates such evaluations using MLflow for comprehensive experiment tracking.

## 2\. Features

  * **Multiple Classifier Evaluation:** Supports training and evaluation of ten distinct classifiers relevant to the study.
  * **Subject-Aware Cross-Validation Strategies:** Implements and compares:
      * Standard Stratified 10-Fold CV.
      * Leave-One-Participant-Out CV (LOPOCV) using `GroupKFold`.
      * Group K-Fold CV (e.g., Group 3-Fold).
  * **Nested Cross-Validation for Hyperparameter Tuning:** Features a nested LOPOCV (outer) + Group 3-Fold CV (inner) design using `GridSearchCV` for robust hyperparameter optimization and model generalization assessment.
  * **Comprehensive MLflow Integration:**
      * Logs experiment parameters: model configurations, CV strategy specifics, data scaling methods.
      * Tracks detailed performance metrics: Accuracy, F1-score, Precision, Recall, MCC, Confusion Matrix per fold, aggregated overall metrics, and average fold metrics.
      * Records **train-test gap** to assess model overfitting.
      * Logs computational efficiency metrics: training time, inference time, and model size, facilitating runtime comparisons.
      * Saves trained `sklearn` model artifacts and input examples.
      * Organizes experiments with parent runs (per data index) and deeply nested child runs (per model/CV strategy/tuning mode) for clarity.
  * **Configurable & Reproducible Pipeline:**
      * Utilizes a `config.yaml` file for managing all critical experiment settings, paths, model selections, scaling choices, and hardware preferences (CPU/GPU).
      * Designed to support transparent and reproducible evaluation standards.
  * **Data Handling:** Includes functions for loading preprocessed data (from pickle files) and applying feature scaling.
  * **Parallel Processing:** Leverages `joblib` for efficient parallel execution of experiments across different data indices (if applicable) or internal model fitting processes.
  * **Structured Output Generation:** Produces detailed CSV files for performance and efficiency metrics, which are also logged as MLflow artifacts for easy access and further analysis (contributing to model ranking stability assessment).

## 3\. Project Structure


Nested_GPT/
├── main.py                     # Main entry point to run the pipeline
├── config/
│   └── config.yaml             # Configuration file for the pipeline
├── src/
│   ├── init.py             # Makes 'src' a package and exposes modules
│   ├── data_loader.py          # Functions for loading and initial data prep
│   ├── metrics.py              # Defines the metric_row function for performance calculation
│   ├── mlflow_utils.py         # Utilities for MLflow setup
│   ├── models.py               # Defines models and their hyperparameter grids
│   ├── training.py             # Core training, CV, and MLflow logging logic
│   └── utils.py                # Utility functions (e.g., saving temporary CSVs)
├── requirements.txt            # (Recommended) Python dependencies
└── README.md                   # This file

## 4\. Prerequisites

  * Python (e.g., 3.9, 3.10, 3.11)
  * Conda or a virtual environment manager (recommended)
  * An MLflow tracking server running and accessible (if `tracking_uri` in `config.yaml` is not local file-based).
  * Necessary system libraries for specific models (e.g., for GPU support if enabled for XGBoost/LightGBM).

## 5\. Installation & Setup

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <your-repository-url>
    cd Nested_GPT
    ```

2.  **Create and activate a Conda environment (recommended):**

    ```bash
    conda create -n your_env_name python=3.11 -y  # Replace your_env_name
    conda activate your_env_name
    ```

3.  **Install dependencies:**
    It's highly recommended to create and use a `requirements.txt` file based on your working environment.
    If you have one:

    ```bash
    pip install -r requirements.txt
    ```

    If not, install the core libraries manually (example versions, adjust as needed):

    ```bash
    pip install numpy pandas scikit-learn=="1.3.0" mlflow=="2.9.2" joblib pyyaml # Core
    pip install xgboost lightgbm # Optional, install if used
    pip install pyarrow --no-cache-dir # If you faced pyarrow issues
    ```

    *Note: For GPU support with XGBoost/LightGBM, you will need to install them following their specific GPU installation guides which may involve CUDA toolkit setup and specific compilation flags.*

4.  **Set up MLflow Tracking Server:**
    Ensure your MLflow tracking server is running and accessible at the URI specified in `config/config.yaml` (e.g., `http://localhost:5000`). You can start a local server by running `mlflow ui` in a separate terminal from your Conda environment where MLflow is installed.

## 6\. Configuration

The main configuration for the pipeline is done through the `config/config.yaml` file. Key sections include:

  * **`experiment`**:
      * `name`: Name of the MLflow experiment.
      * `tracking_uri`: URI of the MLflow tracking server.
  * **`paths`**:
      * `data_dir`: Directory where input data (e.g., `filtered_df_{idx}GBC.pkl`) is located.
      * `result_dir`: Directory where output CSV files will be saved locally.
  * **`scaling`**:
      * `scaler`: Type of scaler to use (e.g., `StandardScaler`, `MinMaxScaler`, `Normalizer`).
  * **`job_control`**:
      * `outer_n_jobs`: Number of parallel jobs for processing different data indices (used in `main.py`). Set to `1` for sequential processing of indices.
      * `internal_n_jobs`: Number of parallel jobs for internal tasks like `GridSearchCV` and `cross_val_score` (used in `training.py`).
  * **`hardware`**:
      * `use_gpu_xgb`: `true` or `false` to enable/disable GPU for XGBoost. Requires XGBoost compiled with GPU support.
      * `use_gpu_lgbm`: `true` or `false` to enable/disable GPU for LightGBM. Requires LightGBM compiled with GPU support.
  * **`models`**:
      * `selected`: (Optional) A list of model names (keys from `build_models` in `src/models.py`) to run. If omitted or `null`/empty, all models defined in `build_models` will be run.
  * **`file_indices`**: A list of integers representing the data indices to process (e.g., `[10, 20]`). Each index typically corresponds to a specific preprocessed data file.

**Example `config.yaml` snippet:**

```yaml
experiment:
  name: Subject_Aware_ACL_Validation_v1
  tracking_uri: http://localhost:5000

paths:
  data_dir: /path/to/your/data/ # e.g., /home/amir/datasets/acl_study/
  result_dir: /path/to/your/results/ # e.g., /home/amir/outputs/acl_study_results/

scaling:
  scaler: StandardScaler

job_control:
  outer_n_jobs: 1
  internal_n_jobs: 4 # Set based on your CPU cores

hardware:
  use_gpu_xgb: false # Set to true if XGBoost GPU version is installed
  use_gpu_lgbm: false # Set to true if LightGBM GPU version is installed

models:
  selected: # Leave empty or remove to run all models from src/models.py
    # - "Random Forest"
    # - "XGBoost"
    # - "LightGBM"
    # - "KNN"
    # - "Logistic Regression"
    # - "AdaBoost"
    # - "LDA"
    # - "QDA"
    # - "Gradient Boosting Classifier"
    # - "Extra Trees"

file_indices: [10] # The specific dataset index for filtered_df_10GBC.pkl
```

## 7. Usage

To run the pipeline:

1.  Ensure your Conda environment is activated:
    ```bash
    conda activate lopocv_env # Or your chosen environment name
    ```

2.  Navigate to the project root directory (e.g., `Nested_LOPOCV/`):
    ```bash
    cd path/to/your/Nested_LOPOCV
    ```

3.  Verify that your `config/config.yaml` is correctly set up, especially the `paths` and `file_indices`. Pay close attention to `data_dir` and `result_dir`.

4.  Run the main script:
    ```bash
    python main.py
    ```

The script will output progress to the console and log all experiments to your configured MLflow server.

## 8. MLflow Integration

* **Experiment Tracking:** All runs are logged under the experiment name specified in `config.yaml`.
* **Run Hierarchy:**
    * A parent MLflow run is created for each `data_idx` (e.g., `idx_10_processing`).
    * Nested child runs are created for each combination of (model, CV strategy, tuning mode) (e.g., `XGBoost_LOPOCV_NestedCV_idx10`).
* **Logged Information per Child Run:**
    * **Parameters:** Hyperparameters (initial fixed ones or best found during tuning), scaler type, CV details, data index, `model_name_for_plot` (if you added it).
    * **Metrics:**
        * Per-fold metrics: Test Accuracy, Training Accuracy, Group CV Validation Accuracy (inner loop score), F1, Precision, Recall, MCC.
        * Aggregated metrics: `overall_accuracy`, `overall_f1_score`, etc., across all folds of the child run.
        * Average fold metrics: `avg_fold_test_accuracy`, `avg_fold_train_accuracy`, etc.
        * Efficiency metrics: `avg_fold_train_time_s`, `avg_sample_infer_time_s`, `avg_fold_model_size_mb`.
    * **Tags:** `model_name`, `outer_cv_strategy`, `tuning_mode`, `data_idx`, `scaler`.
    * **Artifacts:**
        * Trained `sklearn` model (from the last successfully processed fold of the child run).
        * Input example (sample of training data) for the model.
        * CSV tables (logged to MLflow artifact store):
            * `tables/{run_name}_fold_details.csv`
* **Logged Information per Parent Run (per data index):**
    * **Parameters:** `data_idx`, `scaler`, `n_samples`, `n_features`, `n_participants`, `internal_n_jobs_config`.
    * **Artifacts:**
        * `dataset_info/dataset_summary_idx{idx}_...json`
        * `performance_tables_idx{idx}_by_cv/{tag}/perf_{tag}_idx{idx}.csv`
        * `efficiency_tables_idx_summary/computational_efficiency_idx{idx}.csv`
        * `performance_tables_idx_summary/combined_performance_all_cv_idx{idx}.csv`

You can view, compare, and analyze these runs using the MLflow UI.

## 9. Output

Besides MLflow, the pipeline also saves CSV files to the local filesystem in the directory specified by `result_dir` in `config.yaml`. The structure typically mirrors the artifacts logged to MLflow for tables.

* **`{result_dir}/computational_efficiency_idx{idx}.csv`**
* **`{result_dir}/combined_performance_all_cv_idx{idx}.csv`**
* **`{result_dir}/{CV_STRATEGY_TAG}/perf_{CV_STRATEGY_TAG}_idx{idx}.csv`** (e.g., `LOPOCV/perf_LOPOCV_idx10.csv`)

## 10. Troubleshooting Common Issues

* **`LightGBMError: GPU Tree Learner was not enabled...` / XGBoost GPU issues:**
    * Ensure LightGBM/XGBoost are compiled with GPU support if `use_gpu_lgbm: true` or `use_gpu_xgb: true` in `config.yaml`.
    * Alternatively, set these flags to `false` to use CPU.
* **`ValueError: pyarrow.lib.IpcWriteOptions size changed...`:**
    * This often indicates a PyArrow binary incompatibility. Try reinstalling:
        ```bash
        pip uninstall pyarrow -y && pip install pyarrow --no-cache-dir
        ```
* **`NameError: name 'some_function' is not defined`:**
    * Check for missing `import` statements at the top of the relevant Python file (e.g., `from sklearn.metrics import accuracy_score` in `src/training.py`).
* **`TypeError: function() missing X required positional arguments...` or `got an unexpected keyword argument`:**
    * Ensure the function definition (parameters it accepts) and the function call (arguments being passed) match exactly.
* **No artifacts for specific models (e.g., XGBoost, LightGBM):**
    * Check the console output carefully for messages like `"No valid folds completed for {run_name}. Skipping..."`. If this appears, it means all cross-validation folds for that specific model configuration failed.
    * To debug, add more `print()` statements within the fold loop in `src/training.py` to see data shapes (e.g., `X_tr.shape`, `y_tr.shape`) and any exceptions caught for problematic models.
* **`ValueError: The truth value of a Series is ambiguous...` in `_mean` function:**
    * Ensure the `_mean` function in `src/training.py` is correctly implemented to handle both Python `list`s and pandas `Series` as input.

## 11. Citation (To be added)

Once the associated journal paper is published, please add citation details here.
**Title:** "Subject-Aware Model Validation for Repeated-Measures Data: A Nested Approach for Trustworthy Medical AI Applications"
**Authors:** Abdolamir Karbalaie, *et al.* (adjust as per final publication)
*(Consider adding a link to preprint or published paper when available)*

---

*This README is a template. Please review and customize it to accurately reflect all specifics of your project, including paths, exact library versions if critical, and any custom logic or features you've implemented.*


