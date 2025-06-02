# mlflow_pipeline/src/mlflow_utils.py
import mlflow
from mlflow.tracking import MlflowClient

def setup_mlflow(exp_name: str, tracking_uri: str) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        print(f"Creating MLflow experiment → {exp_name}")
        exp_id = client.create_experiment(exp_name)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(exp_name)
    return exp_id
