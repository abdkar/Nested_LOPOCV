# mlflow_pipeline/main.py
import sys, yaml, time # Add time
from pathlib import Path
from joblib import cpu_count # Keep if you want to override config, or remove

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.mlflow_utils import setup_mlflow
from src.training import parallel_run # Ensure this is 'training' not 'training0'
# import warnings # If you want to add warning suppression
# warnings.filterwarnings("ignore") # Add if parity with monolithic is desired

if __name__ == "__main__":
    start_time_total = time.time() # For total runtime

    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    config = yaml.safe_load(cfg_path.read_text())

    exp_name = config["experiment"]["name"]
    tracking_uri = config["experiment"]["tracking_uri"]
    data_dir = config["paths"]["data_dir"]
    result_dir = config["paths"]["result_dir"]
    indices = config.get("file_indices", [10])
    scaler = config["scaling"]["scaler"]

    # Get job control and hardware settings
    outer_n_jobs = config.get("job_control", {}).get("outer_n_jobs", 1)
    internal_n_jobs = config.get("job_control", {}).get("internal_n_jobs", 4) # Default to 4 if not in config
    use_gpu_xgb = config.get("hardware", {}).get("use_gpu_xgb", False) # Default to False
    use_gpu_lgbm = config.get("hardware", {}).get("use_gpu_lgbm", False) # Default to False
    selected_models_list = config.get("models", {}).get("selected", None)


    exp_id = setup_mlflow(exp_name, tracking_uri)

    if exp_id: # Proceed only if MLflow setup was successful
        print(f"🚀 Launching pipeline → Experiment='{exp_name}' | Indices={indices}")
        parallel_run(
            indices=indices,
            exp_id=exp_id,
            data_dir=data_dir,
            result_dir=result_dir,
            n_jobs_outer=outer_n_jobs, # Pass this
            internal_n_jobs=internal_n_jobs, # Pass this
            scaler_name=scaler,
            use_gpu_xgb=use_gpu_xgb, # Pass this
            use_gpu_lgbm=use_gpu_lgbm, # Pass this
            selected_models_list=selected_models_list # Pass this
        )
        print(f"🏁 Pipeline finished in {time.time() - start_time_total:.2f} seconds.")
    else:
        print("MLflow setup failed. Pipeline not started.")