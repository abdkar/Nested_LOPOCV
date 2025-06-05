# mlflow_pipeline/src/training.py
import os, time, pickle, json  # Add json
from pathlib import Path
import tempfile
from typing import List, Tuple, Dict, Any # Add Dict, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import mlflow
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     GridSearchCV, cross_val_score)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
# Ensure these imports are correct based on your file structure
from .data_loader import load_data # LIST_ID will be used via load_data -> extract_pid
from .models import build_models, MODEL_PARAM_GRIDS # MODEL_PARAM_GRIDS is now defined in models.py
from .metrics import metric_row
from .mlflow_utils import setup_mlflow # Ensure this is the correct import path
from .utils import save_tmp_csv # Keep this if you like it, or use monolithic's tempfile approach

from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     GridSearchCV, cross_val_score)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics import accuracy_score # <<< Add this import

# Helper from monolithic (or keep in utils.py)
def _get_model_size_bytes(model) -> int:
    """Gets the size of a model in bytes after pickling."""
    try:
        return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) # Match monolithic
    except Exception as e:
        print(f"Warning: Could not get model size: {e}")
        return 0

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _get_scaler(name: str): # Keep as is
    if name == "StandardScaler": return StandardScaler()
    if name == "MinMaxScaler": return MinMaxScaler()
    return Normalizer()


def _mean(data_input) -> float:
    """
    Calculates the mean of a list or pandas Series,
    handling empty cases or lists/series containing NaNs.
    """
    if isinstance(data_input, pd.Series):
        if data_input.empty:
            return np.nan
        # np.mean on a pandas Series handles NaNs appropriately (by skipping them by default)
        return float(np.mean(data_input))
    elif isinstance(data_input, list):
        if not data_input:  # Check if the list is empty
            return np.nan
        # np.mean on a list containing NaNs will result in np.nan
        # np.mean on an empty list raises a warning and returns np.nan
        # This check for emptiness handles it cleanly.
        return float(np.mean(data_input))
    else:
        # Handle other unexpected types if necessary, or raise an error
        print(f"Warning: _mean called with an unsupported type: {type(data_input)}. Returning np.nan.")
        return np.nan


# run_outer_cv to match monolithic closely
def run_outer_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    pids: np.ndarray,
    outer_cv, # CV splitter instance
    inner_cv, # CV splitter instance for GridSearchCV and cross_val_score
    tag: str, # CV strategy tag (e.g., "LOPOCV")
    idx: int, # Data index
    parent_run_id: str, # For MLflow nested runs
    exp_id: str, # MLflow experiment ID
    scaler_name: str, # For logging
    internal_n_jobs: int, # For GridSearchCV, cross_val_score, some models
    use_gpu_xgb: bool,
    use_gpu_lgbm: bool,
    selected_models_list: list = None # From config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (performance_table, efficiency_table) for one CV strategy."""
    tables: List[pd.DataFrame] = []
    eff_rows: List[dict] = []

    # Models are now built with potential GPU/n_jobs config
    # And potentially filtered by selected_models_list
    models_dict = build_models(internal_n_jobs, use_gpu_xgb, use_gpu_lgbm, selected_models_list)

    # Param grids are now globally defined in models.py as MODEL_PARAM_GRIDS
    # Ensure MODEL_PARAM_GRIDS is comprehensive and matches monolithic logic.

    for model_name, model_prototype in models_dict.items():
        tuning_modes = [False] # Always run Fixed
        if model_name in MODEL_PARAM_GRIDS and MODEL_PARAM_GRIDS[model_name]: # Check if grid exists and is not empty
            tuning_modes.append(True) # Add NestedCV if tuning params exist

        for tuning in tuning_modes:
            mode_tag = "NestedCV" if tuning else "Fixed"
            technique_label = f"{tag} - {mode_tag}" if not (tag == "10Fold" and mode_tag == "Fixed") else "10Fold" # Monolithic technique label
            run_name = f"{model_name}_{tag}_{mode_tag}_idx{idx}"

            with mlflow.start_run(run_name=run_name, experiment_id=exp_id, nested=True) as child_run:
                print(f"      Starting MLflow run: {run_name} (Child of {parent_run_id})")
                mlflow.set_tag("mlflow.parentRunId", parent_run_id) # Explicitly set parent
                mlflow.set_tag("outer_cv_strategy", tag)
                mlflow.set_tag("tuning_mode", mode_tag)
                mlflow.set_tag("data_idx", idx)
                mlflow.set_tag("model_name", model_name)
                mlflow.log_param("model_name_for_plot", model_name) # <<< ADD THIS LINE
                mlflow.set_tag("scaler", scaler_name)

                # Log initial model parameters if not tuning (Fixed mode)
                if not tuning:
                    try:
                        initial_params = model_prototype.get_params()
                        loggable_params = {k:v for k,v in initial_params.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                        mlflow.log_params({f"initial_{k}": v for k, v in loggable_params.items()})
                    except Exception as e:
                        print(f"Warning: Could not log initial params for {model_name} ({mode_tag}): {e}")

                fold_metrics_rows = []
                train_times, infer_times, model_sizes_fold = [], [], []
                all_y_true_run, all_y_pred_run = [], [] # For overall metrics for this child run

                for fold, (tr_indices, te_indices) in enumerate(outer_cv.split(X, y, groups=pids)):
                    X_tr, X_te = X.iloc[tr_indices], X.iloc[te_indices]
                    y_tr, y_te = y[tr_indices], y[te_indices]
                    pids_tr, pids_te = pids[tr_indices], pids[te_indices]

                    # Monolithic script's robustness checks
                    if len(np.unique(y_tr)) < 2:
                        print(f"Skipping fold {fold+1} for {model_name} ({mode_tag}) due to single class in training data (train set size: {len(y_tr)}).")
                        continue
                    if len(X_te) == 0 : # Test set is empty
                        print(f"Skipping fold {fold+1} for {model_name} ({mode_tag}) due to empty test set.")
                        continue
                    if len(np.unique(y_te)) < 1: # No labels in test set (should not happen if X_te not empty)
                         print(f"Warning: Fold {fold+1} for {model_name} ({mode_tag}) has no labels in test data (test set size: {len(y_te)}).")
                    elif len(np.unique(y_te)) < 2:
                        print(f"Warning: Fold {fold+1} for {model_name} ({mode_tag}) has single class in test data (test set size: {len(y_te)}). Metrics might be misleading.")

                    # Deep clone the model prototype for each fold (monolithic approach)
                    current_model = pickle.loads(pickle.dumps(model_prototype))
                    fold_train_start_time = time.perf_counter()
                    fold_cv_acc = np.nan # CV accuracy from inner loop on this fold's training data
                    best_params_fold = {}

                    try:
                        if tuning:
                            grid_params = MODEL_PARAM_GRIDS.get(model_name, {})
                            # Ensure inner_cv uses groups correctly if it's GroupKFold
                            grid = GridSearchCV(current_model, grid_params, cv=inner_cv, scoring="accuracy", n_jobs=internal_n_jobs, verbose=0)
                            grid.fit(X_tr, y_tr, groups=pids_tr if isinstance(inner_cv, GroupKFold) else None)
                            current_model = grid.best_estimator_
                            fold_cv_acc = grid.best_score_
                            best_params_fold = grid.best_params_
                            mlflow.log_params({f"fold{fold+1}_best_{k}": v for k, v in best_params_fold.items()})
                        else: # Fixed hyper-parameters
                            # Calculate cross_val_score on training data for comparison
                            cv_scores = cross_val_score(current_model, X_tr, y_tr, cv=inner_cv, groups=pids_tr if isinstance(inner_cv, GroupKFold) else None, scoring="accuracy", n_jobs=internal_n_jobs)
                            fold_cv_acc = cv_scores.mean()
                            current_model.fit(X_tr, y_tr) # Fit on full training part of the fold

                    except Exception as e:
                        print(f"Error during training for fold {fold+1}, model {model_name} ({mode_tag}): {e}")
                        train_times.append(time.perf_counter() - fold_train_start_time)
                        continue # Skip to next fold

                    current_train_time = time.perf_counter() - fold_train_start_time
                    train_times.append(current_train_time)
                    model_sizes_fold.append(_get_model_size_bytes(current_model))

                    # Inference
                    current_infer_time_total = np.nan
                    if len(X_te) > 0:
                        infer_start_time = time.perf_counter()
                        try:
                            pred_te = current_model.predict(X_te)
                            current_infer_time_total = time.perf_counter() - infer_start_time
                            infer_times.append(current_infer_time_total / len(X_te) if len(X_te) > 0 else 0) # Per sample
                        except Exception as e:
                            print(f"Error during prediction for fold {fold+1}, model {model_name} ({mode_tag}): {e}")
                            pred_te = np.array([])
                            infer_times.append(np.nan)
                    else: # Empty test set
                        pred_te = np.array([])
                        infer_times.append(np.nan)


                    # Fold's training accuracy (on this fold's training data)
                    fold_train_acc = np.nan
                    try:
                        if len(X_tr) > 0: # Ensure training data is not empty
                            pred_tr_for_acc = current_model.predict(X_tr) # Predict on training data for train_acc
                            fold_train_acc = accuracy_score(y_tr, pred_tr_for_acc)
                    except Exception as e:
                        print(f"Error calculating train accuracy for fold {fold+1}: {e}")


                    # PID for metric_row (monolithic logic for single vs multiple PIDs in test fold)
                    unique_pids_te = np.unique(pids_te)
                    pid_for_row_val = "|".join(map(str, unique_pids_te)) if len(unique_pids_te) > 0 else "N/A"

                    # Use the metric_row function (ensure it's the monolithic version)
                    row_data = metric_row(pred_te, y_te, pid_for_row_val, fold_train_acc, fold_cv_acc)
                    row_data["Fold"] = fold + 1
                    # Add other specific fields if your metric_row doesn't include them directly from monolithic
                    # These are added to eff_rows in monolithic, but let's add to fold_metrics_rows for now for the CSV.
                    # This part differs from your original refactored code's row.update()
                    row_data["Fold Training Time (s)"] = current_train_time
                    row_data["Fold Inference Time (s/sample)"] = current_infer_time_total / len(X_te) if len(X_te) > 0 else np.nan
                    row_data["Fold Model Size (bytes)"] = _get_model_size_bytes(current_model) # Already in model_sizes_fold
                    row_data["Tuning Mode"] = mode_tag # Already present in your original refactored code

                    fold_metrics_rows.append(row_data)

                    all_y_true_run.extend(y_te) # For overall run metrics
                    all_y_pred_run.extend(pred_te)


                # After all folds for this model/tuning_mode run
                if not fold_metrics_rows:
                    print(f"No valid folds completed for {run_name}. Skipping metric aggregation and model logging.")
                    continue # Skip to next tuning mode or model

                # Log overall metrics for this child run (monolithic style)
                if all_y_true_run and all_y_pred_run:
                    try:
                        mlflow.log_metric("overall_accuracy", accuracy_score(all_y_true_run, all_y_pred_run))
                        mlflow.log_metric("overall_f1_score", f1_score(all_y_true_run, all_y_pred_run, zero_division=0))
                        mlflow.log_metric("overall_precision", precision_score(all_y_true_run, all_y_pred_run, zero_division=0))
                        mlflow.log_metric("overall_recall", recall_score(all_y_true_run, all_y_pred_run, zero_division=0))
                        if len(np.unique(all_y_true_run)) > 1 and len(np.unique(all_y_pred_run)) > 1: # Check for MCC
                            mlflow.log_metric("overall_mcc", matthews_corrcoef(all_y_true_run, all_y_pred_run))
                    except Exception as e:
                         print(f"Error logging overall run metrics for {run_name}: {e}")

                # Log average fold metrics (monolithic style)
                df_fold_metrics_temp = pd.DataFrame(fold_metrics_rows)
                for m_key in ["Test Accuracy", "Training Accuracy", "Group CV Validation Accuracy", "Test F1 Score", "Test Precision", "Test Recall", "Test MCC"]:
                    if m_key in df_fold_metrics_temp.columns:
                         mlflow.log_metric(f"avg_fold_{m_key.lower().replace(' ', '_')}", _mean(df_fold_metrics_temp[m_key].dropna()))


                # Save fold-level metrics table for this specific run (model_name, mode_tag)
                df_model_perf_run = pd.DataFrame(fold_metrics_rows)
                # Suffixing columns like monolithic: df_model_perf.add_suffix(f"_{model_name}_{mode_tag}")
                # Your save_tmp_csv handles logging to "tables/{run_name}.csv"
                # For parity, ensure the table structure and naming matches.
                # Monolithic saved one table per child run.
                save_tmp_csv(df_model_perf_run, f"{run_name}_fold_details") # This will be tables/{run_name}_fold_details.csv
                tables.append(df_model_perf_run.add_suffix(f"_{model_name}_{mode_tag}")) # For combined table later in process_idx


                # Log model from the LAST fold (monolithic style)
                # 'current_model' is from the last successful fold here.
                if 'current_model' in locals() and current_model is not None:
                    input_example_df = X_tr.head(5) if not X_tr.empty else None # Example from last fold's X_tr
                    try:
                        mlflow.sklearn.log_model(
                            sk_model=current_model,
                            artifact_path=f"{model_name}_{mode_tag}_model", # Simpler path
                            input_example=input_example_df
                        )
                    except Exception as e:
                        print(f"Error logging model artifact for {run_name}: {e}")

                # Efficiency data for this run (model_name, mode_tag)
                eff_rows.append({
                    "Model": model_name,
                    "Technique": technique_label, # Monolithic label
                    "Avg. Train Time (s/fold)": _mean(train_times),
                    "Avg. Inference Time (s/sample)": _mean(infer_times), # This is per sample
                    "Model Size (MB - Avg Fold)": _mean(model_sizes_fold) / 1_048_576 if model_sizes_fold else np.nan, # Avg over folds
                    "Data Index": idx,
                    "Tuning Mode": mode_tag,
                    "CV Strategy": tag,
                    "Completed Folds": len(train_times),
                })
                # Log average efficiency metrics to MLflow for the child run
                mlflow.log_metric("avg_fold_train_time_s", _mean(train_times))
                mlflow.log_metric("avg_sample_infer_time_s", _mean(infer_times)) # Per sample
                mlflow.log_metric("avg_fold_model_size_mb", _mean(model_sizes_fold) / 1_048_576 if model_sizes_fold else np.nan)


    # This combines tables from ALL models and tuning modes under ONE CV strategy (e.g. LOPOCV)
    # Monolithic script saves one CSV per (model,tuning_mode) and then a combined one per index.
    # Your refactored code structure already creates tables per run.
    # The list 'tables' now contains DFs from each child run.
    # For `process_idx` to create a single combined CSV for the CV strategy:
    # perf_df_cv_strategy = pd.concat(tables, axis=1) if tables else pd.DataFrame()
    # This `perf_df_cv_strategy` might become very wide. Monolithic logic for saving this:
    # it saved perf_<CV>_idxXX.csv which seemed to be this wide table.

    # This return matches your original refactored code structure.
    # `tables` contains individual run DFs, `eff_rows` contains summary dicts.
    return tables, pd.DataFrame(eff_rows)


def process_idx(
    idx: int,
    exp_id: str,
    data_dir: str,
    result_dir: str,
    internal_n_jobs: int, # New
    scaler_name: str,
    use_gpu_xgb: bool, # New
    use_gpu_lgbm: bool, # New
    selected_models_list: list = None # New
) -> None:
    # Monolithic: Re-set tracking URI and experiment in the worker process
    # This is good practice for joblib workers.
    try:
        # Assuming TRACKING_URI is accessible globally or passed if not part of exp_id context
        # For simplicity, if your `setup_mlflow` in main sets it globally for the main process,
        # joblib *might* inherit. If not, TRACKING_URI needs to be passed or re-read from config.
        # For robust parity:
        # cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        # config = yaml.safe_load(cfg_path.read_text())
        # tracking_uri_worker = config["experiment"]["tracking_uri"]
        # mlflow.set_tracking_uri(tracking_uri_worker)
        mlflow.set_experiment(experiment_id=exp_id)
    except Exception as e:
        print(f"Error setting up MLflow in worker for index {idx}: {e}. Skipping.")
        return

    print(f"\n===== Processing data index {idx} (PID {os.getpid()}) =====")

    # Data loading logic to match monolithic more closely regarding path finding
    # Monolithic used: f"/.../filtered_df_{idx}GBC.pkl"
    # Your refactored used base and alt. Let's stick to your refactored flexible way.
    base_path = Path(data_dir) / f"filtered_df_{idx}.pkl"
    alt_path = Path(data_dir) / f"filtered_df_{idx}GBC.pkl" # Common GBC suffix
    pkl_to_load = None
    if alt_path.exists(): # Prefer GBC path if exists, similar to monolithic intent
        pkl_to_load = alt_path
    elif base_path.exists():
        pkl_to_load = base_path
    else:
        print(f"❌ Data file not found for idx {idx} at {base_path} or {alt_path}. Skipping.")
        return

    print(f"Loading data from: {pkl_to_load}")
    try:
        df = load_data(str(pkl_to_load)) # load_data expects string path
    except FileNotFoundError: # Should be caught by above check, but good to have
        print(f"Error: Data file not found at {pkl_to_load}. Skipping index {idx}.")
        return
    except Exception as e:
        print(f"Error loading/processing data from {pkl_to_load}: {e}. Skipping index {idx}.")
        return

    if "target" not in df.columns:
        print(f"Error: 'target' column missing in {pkl_to_load}. Skipping index {idx}.")
        return

    cols_to_drop = [col for col in ["target", "Test_ID", "seg_order"] if col in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df["target"].values
    pids = df["Test_ID"].values

    if X.empty: print(f"Warning: Features X are empty for index {idx}. Skipping."); return
    if len(np.unique(y)) < 2: print(f"Warning: Target y has < 2 classes for index {idx}. Skipping."); return

    num_participants = len(np.unique(pids))
    min_samples_per_class = min((np.sum(y == c) for c in np.unique(y)), default=0)

    # Conditional CV execution flags (from monolithic)
    run_lopo = num_participants >= 2
    run_g3 = num_participants >= 3
    run_10fold = min_samples_per_class >= 10 # Or n_splits for StratifiedKFold

    if not run_lopo: print(f"Skipping LOPOCV for idx {idx}: requires >=2 participants, found {num_participants}")
    if not run_g3: print(f"Skipping Group3Fold for idx {idx}: requires >=3 participants, found {num_participants}")
    if not run_10fold: print(f"Skipping 10Fold for idx {idx}: requires >=10 samples/class, found min {min_samples_per_class}")


    scaler_obj = _get_scaler(scaler_name)
    X = pd.DataFrame(scaler_obj.fit_transform(X), columns=X.columns, index=X.index)

    inner_cv = GroupKFold(3) # As per monolithic (inner_cv for GridSearchCV)
    res_root = Path(result_dir) # Output directory for CSVs

    # Parent run for this data index
    with mlflow.start_run(run_name=f"idx_{idx}_processing", experiment_id=exp_id) as parent_run_obj:
        parent_run_id_for_children = parent_run_obj.info.run_id
        print(f"  Parent MLflow run for index {idx}: {parent_run_id_for_children}")
        mlflow.log_params({
            "data_idx": idx, "scaler": scaler_name, "n_samples": len(df),
            "n_features": X.shape[1], "n_participants": num_participants,
            "internal_n_jobs_config": internal_n_jobs # Log this config
        })

        # Log dataset summary JSON (monolithic)
        try:
            summary_dict = {"participants": num_participants, "samples": X.shape[0], "features": X.shape[1], "classes": len(np.unique(y)), "min_samples_per_class": min_samples_per_class}
            # Use a unique temp name or save directly to result_dir then log
            summary_json_path = Path(tempfile.gettempdir()) / f"dataset_summary_idx{idx}_{parent_run_id_for_children}.json"
            with open(summary_json_path, 'w') as f:
                json.dump(summary_dict, f, indent=4)
            mlflow.log_artifact(summary_json_path, "dataset_info")
            os.remove(summary_json_path)
        except Exception as e:
            print(f"Error logging dataset summary artifact for index {idx}: {e}")


        cv_strategies = {}
        if run_lopo: cv_strategies["LOPOCV"] = GroupKFold(n_splits=num_participants)
        if run_10fold: cv_strategies["10Fold"] = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        if run_g3: cv_strategies["Group3Fold"] = GroupKFold(n_splits=3) # Monolithic used "Group 3-fold CV" as tag

        all_cv_perf_dfs_for_idx = [] # To combine all model perf tables for this idx
        all_cv_eff_dfs_for_idx = []  # To combine all model eff tables for this idx

        for tag, outer_cv_splitter in cv_strategies.items():
            cv_tag_dir = res_root / tag
            _ensure_dir(cv_tag_dir) # Ensure subdir exists for this CV type

            print(f"    Running CV Strategy: {tag} for index {idx}")
            # run_outer_cv returns list of DFs (one per model/mode) and one DF for efficiency
            list_of_model_perf_dfs, df_eff_for_cv_strat = run_outer_cv(
                X, y, pids, outer_cv_splitter, inner_cv,
                tag, idx, parent_run_id_for_children, exp_id, scaler_name,
                internal_n_jobs, use_gpu_xgb, use_gpu_lgbm, selected_models_list
            )

            if df_eff_for_cv_strat is not None and not df_eff_for_cv_strat.empty:
                all_cv_eff_dfs_for_idx.append(df_eff_for_cv_strat)

            # Monolithic saves one perf_<CV>_idxXX.csv which is a wide table of all models for that CV.
            # list_of_model_perf_dfs contains DFs for each model run under this CV strategy.
            if list_of_model_perf_dfs:
                # Create the wide combined table for this CV strategy
                combined_perf_df_for_cv_strat = pd.concat(list_of_model_perf_dfs, axis=1) if list_of_model_perf_dfs else pd.DataFrame()
                if not combined_perf_df_for_cv_strat.empty:
                    perf_path = cv_tag_dir / f"perf_{tag}_idx{idx}.csv"
                    combined_perf_df_for_cv_strat.to_csv(perf_path, index=False)
                    mlflow.log_artifact(str(perf_path), artifact_path=f"performance_tables_idx{idx}/{tag}")
                    all_cv_perf_dfs_for_idx.append(combined_perf_df_for_cv_strat) # Collect for the super-combined table

        # After all CV strategies for this index idx
        if all_cv_eff_dfs_for_idx:
            eff_all_idx = pd.concat(all_cv_eff_dfs_for_idx, ignore_index=True)
            eff_path_idx = res_root / f"computational_efficiency_idx{idx}.csv"
            eff_all_idx.to_csv(eff_path_idx, index=False)
            mlflow.log_artifact(str(eff_path_idx), artifact_path="efficiency_tables_idx_summary")

        # Monolithic combined performance table (very wide, concatenates results from all CV strategies for this idx)
        if all_cv_perf_dfs_for_idx:
            # This might be extremely wide. Ensure this is the desired behavior.
            # Each element in all_cv_perf_dfs_for_idx is already a wide table for a CV strategy.
            # Concatenating them side-by-side might not be what monolithic did for the "combined_performance_idxXX.csv".
            # Monolithic seemed to save perf_LOPOCV_idx, perf_10Fold_idx separately.
            # The "combined_performance_idxXX.csv" in monolithic script might have been a different aggregation.
            # For now, let's assume we log the individual CV strategy summary performance tables.
            # If a single grand combined table is needed, the logic for its structure must be clarified from monolithic.
            # The previous `combo = pd.concat([df.reset_index(drop=True) for df in perf_all], axis=1)`
            # in your refactored code suggests combining the wide tables from each CV strategy side-by-side.
            grand_combo_perf_df = pd.concat([df.reset_index(drop=True) for df in all_cv_perf_dfs_for_idx], axis=1)
            if not grand_combo_perf_df.empty:
                combo_path = res_root / f"combined_performance_idx{idx}.csv"
                grand_combo_perf_df.to_csv(combo_path, index=False)
                mlflow.log_artifact(str(combo_path), artifact_path="performance_tables_idx_summary")

    print(f"===== Completed data index {idx} =====")


# Update parallel_run signature to accept and pass new configs
def parallel_run(
    indices: list[int],
    exp_id: str,
    data_dir: str,
    result_dir: str,
    n_jobs_outer: int,      # Renamed from n_jobs for clarity
    internal_n_jobs: int,
    scaler_name: str,
    use_gpu_xgb: bool,
    use_gpu_lgbm: bool,
    selected_models_list: list = None
) -> None:
    # Create result_dir if it doesn't exist (monolithic did this for subdirs)
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    Parallel(n_jobs=n_jobs_outer)(
        delayed(process_idx)(
            i, exp_id, data_dir, result_dir,
            internal_n_jobs, scaler_name,
            use_gpu_xgb, use_gpu_lgbm, selected_models_list
        ) for i in indices
    )
    print("✅ All indices processed.")