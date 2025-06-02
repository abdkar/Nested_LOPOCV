# mlflow_pipeline/src/data_loader.py
from pathlib import Path
import pandas as pd
import re

LIST_ID = [106, 108, 109, 110, 114, 115, 119, 125, 133, 135, 138, 148, 150,
           151, 152, 155, 156, 157, 164, 190, 191, 192, 197, 51102, 51103,
           51104, 51107, 51108, 51110, 51111, 51113, 51114, 51115, 51118,
           51127, 51128, 103, 107, 116, 118, 121, 127, 129, 134, 137, 139,
           140, 141, 142, 143, 144, 145, 149, 160, 162, 165, 170, 174, 181,
           198, 199, 51101, 51112, 51119, 51120, 51121, 51122, 51123, 51124,
           51125, 51126, 51129]

def clean_name(col: str) -> str: # Keep as is
    return re.sub(r"\W+", "_", col)

def extract_pid(segment_id: str) -> int:
    """Extract participant ID prefix from raw segment id string."""
    s = str(segment_id)
    # Match monolithic script's check
    if len(s) >= 5 and s[:5].isdigit() and int(s[:5]) in LIST_ID:
        return int(s[:5])
    if len(s) >= 3 and s[:3].isdigit() and int(s[:3]) in LIST_ID:
        return int(s[:3])
    raise ValueError(f"Unexpected segment ID format or ID not in LIST_ID: {segment_id}")

# load_data function can remain largely the same,
# but ensure it correctly uses extract_pid and handles columns.
# The monolithic script's seg_order extraction was slightly more robust in process_idx,
# consider if that logic needs to be here or in training.py's process_idx.
# For now, keeping load_data as previously defined by you.
def load_data(pkl_path: str) -> pd.DataFrame:
    """Load pickle → add Test_ID & seg_order → sort."""
    df = pd.read_pickle(Path(pkl_path))
    df["Test_ID"]  = df.index.to_series().apply(extract_pid)
    # Ensure robust seg_order extraction, similar to monolithic if issues arise
    seg_order_extract = df.index.to_series().astype(str).str.extract(r"(\d+)$")
    if seg_order_extract.empty or seg_order_extract.iloc[:, 0].isnull().any():
        print(f"Warning: Could not reliably extract segment order from index for {pkl_path}. Some indices might not end with digits. Defaulting to 0.")
        df["seg_order"] = 0
    else:
        df["seg_order"] = seg_order_extract.iloc[:, 0].astype(int)

    df.sort_values(["Test_ID", "seg_order"], inplace=True)
    return df