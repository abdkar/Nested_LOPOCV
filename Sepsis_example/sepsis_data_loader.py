# Enhanced data loader for sepsis detection (replace or supplement existing data_loader.py)

from pathlib import Path
import pandas as pd
import re
import numpy as np

# Valid ICU patient IDs (matches the sepsis data generator)
SEPSIS_PATIENT_IDS = [
    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
    1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020,
    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
    2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,
    3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010,
    3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020,
    4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010,
    5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010
]

# Use either the original LIST_ID or SEPSIS_PATIENT_IDS based on data type
try:
    from src.data_loader import LIST_ID
    VALID_IDS = LIST_ID + SEPSIS_PATIENT_IDS  # Combine both for flexibility
except ImportError:
    VALID_IDS = SEPSIS_PATIENT_IDS

def extract_patient_id_sepsis(segment_id: str) -> int:
    """
    Extract patient ID from sepsis data segment identifier.
    
    Args:
        segment_id: Segment identifier (e.g., "1001_12" for patient 1001, hour 12)
        
    Returns:
        int: Patient ID
        
    Raises:
        ValueError: If segment ID format is invalid or patient not in valid list
    """
    s = str(segment_id)
    
    # Split by underscore to get patient_id and hour
    parts = s.split('_')
    if len(parts) != 2:
        raise ValueError(f"Invalid sepsis segment ID format: {segment_id}. Expected format: patient_id_hour")
    
    try:
        patient_id = int(parts[0])
        hour = int(parts[1])  # Validate hour is also numeric
    except ValueError:
        raise ValueError(f"Invalid sepsis segment ID format: {segment_id}. Both patient_id and hour must be numeric")
    
    if patient_id not in VALID_IDS:
        raise ValueError(f"Patient ID {patient_id} not in valid sepsis patient list")
    
    return patient_id

def load_sepsis_data(pkl_path: str) -> pd.DataFrame:
    """
    Load and preprocess sepsis detection data from pickle file.
    
    Args:
        pkl_path: Path to pickle file containing sepsis data
        
    Returns:
        pd.DataFrame: Processed dataframe with added columns:
            - Test_ID: Patient ID (for compatibility with existing pipeline)
            - seg_order: Hour number within ICU stay
            - Original clinical features and target
    """
    df = pd.read_pickle(Path(pkl_path))
    
    # Extract patient IDs and hours from index
    try:
        df["Test_ID"] = df.index.to_series().apply(extract_patient_id_sepsis)
        
        # Extract hour (seg_order) from index
        hours = df.index.to_series().str.split('_').str[1].astype(int)
        df["seg_order"] = hours
        
    except Exception as e:
        raise ValueError(f"Error processing sepsis data index format: {e}")
    
    # Validate required columns
    if "target" not in df.columns:
        raise ValueError("Sepsis data must contain 'target' column (0=no sepsis, 1=sepsis)")
    
    # Clinical data validation
    clinical_features = [
        'heart_rate', 'systolic_bp', 'temperature', 'respiratory_rate',
        'oxygen_saturation', 'white_blood_cells', 'lactate'
    ]
    
    missing_features = [f for f in clinical_features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing some clinical features: {missing_features}")
    
    # Handle missing values (common in ICU data)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col not in ['Test_ID', 'seg_order', 'target']]
    
    # Forward fill missing values within each patient (realistic for continuous monitoring)
    df[numeric_columns] = df.groupby('Test_ID')[numeric_columns].fillna(method='ffill')
    
    # Backward fill any remaining missing values
    df[numeric_columns] = df.groupby('Test_ID')[numeric_columns].fillna(method='bfill')
    
    # If still missing, fill with median (last resort)
    for col in numeric_columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Warning: Filled {col} missing values with median: {median_val:.2f}")
    
    # Sort by patient ID and hour for temporal consistency
    df.sort_values(["Test_ID", "seg_order"], inplace=True)
    
    # Clinical data quality checks
    print(f"Loaded sepsis data: {len(df)} measurements from {df['Test_ID'].nunique()} patients")
    print(f"Sepsis prevalence: {(df['target'] == 1).mean():.1%}")
    print(f"Average ICU stay: {df.groupby('Test_ID').size().mean():.1f} hours")
    
    return df

# For compatibility with existing pipeline, provide both functions
def clean_name(col: str) -> str:
    """Clean column names by replacing non-alphanumeric characters."""
    return re.sub(r"\W+", "_", col)

def extract_pid(segment_id: str) -> int:
    """
    Extract participant ID - works with both original and sepsis data formats.
    
    This function provides backward compatibility while supporting sepsis data.
    """
    try:
        # Try sepsis format first (patient_id_hour)
        return extract_patient_id_sepsis(segment_id)
    except ValueError:
        # Fall back to original format if available
        try:
            from src.data_loader import extract_pid as original_extract_pid
            return original_extract_pid(segment_id)
        except ImportError:
            raise ValueError(f"Cannot extract patient ID from: {segment_id}")

def load_data(pkl_path: str) -> pd.DataFrame:
    """
    Load data - automatically detects sepsis vs original format.
    
    This function provides backward compatibility while supporting sepsis data.
    """
    # First, try to load and detect format
    df = pd.read_pickle(Path(pkl_path))
    
    # Check if this looks like sepsis data (has clinical features)
    sepsis_indicators = ['heart_rate', 'temperature', 'white_blood_cells']
    is_sepsis_data = any(col in df.columns for col in sepsis_indicators)
    
    if is_sepsis_data:
        print("Detected sepsis data format")
        return load_sepsis_data(pkl_path)
    else:
        print("Detected original data format")
        # Use original load_data function if available
        try:
            from src.data_loader import load_data as original_load_data
            return original_load_data(pkl_path) 
        except ImportError:
            # Fallback implementation
            df["Test_ID"] = df.index.to_series().apply(extract_pid)
            seg_order_extract = df.index.to_series().astype(str).str.extract(r"(\d+)$")
            if seg_order_extract.empty or seg_order_extract.iloc[:, 0].isnull().any():
                df["seg_order"] = 0
            else:
                df["seg_order"] = seg_order_extract.iloc[:, 0].astype(int)
            df.sort_values(["Test_ID", "seg_order"], inplace=True)
            return df