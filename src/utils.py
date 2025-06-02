# mlflow_pipeline/src/utils.py
import os, tempfile, re
import pandas as pd
import mlflow

def save_tmp_csv(df: pd.DataFrame, name: str) -> None:
    """Save → log → remove temp file."""
    safe_name = re.sub(r"[\\/*?:\"<>|]", "_", name)
    tmp = os.path.join(tempfile.gettempdir(), f"{safe_name}.csv")
    df.to_csv(tmp, index=False)
    mlflow.log_artifact(tmp, artifact_path="tables")
    os.remove(tmp)