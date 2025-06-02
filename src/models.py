# mlflow_pipeline/src/models.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, RandomForestClassifier)
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)

# Conditionally import XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    print("XGBoost imported.")
except ImportError:
    XGBClassifier = None
    print("Warning: xgboost not installed. Skipping XGBoost.")

try:
    from lightgbm import LGBMClassifier
    print("LightGBM imported.")
except ImportError:
    LGBMClassifier = None
    print("Warning: lightgbm not installed. Skipping LightGBM.")

# internal_n_jobs will be passed from training.py or globally set if preferred
# For now, let's assume it's passed to build_models or models take it from their params if set there
def build_models(internal_n_jobs: int = 1, use_gpu_xgb: bool = False, use_gpu_lgbm: bool = False, selected_model_keys=None):
    """Return dict of models. If *selected_model_keys* list supplied, keep only those."""
    all_models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, penalty="l1", solver="saga"),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "LDA": LinearDiscriminantAnalysis(), # Solver can be part of tuning grid
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.5), # As per monolithic
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            subsample=0.8, min_samples_split=10, random_state=42), # As per monolithic
        "Extra Trees": ExtraTreesClassifier( # Parameters from monolithic
            n_estimators=100, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, max_features="sqrt", bootstrap=False, # Monolithic has "sqrt"
            random_state=123, n_jobs=internal_n_jobs),
        "Random Forest": RandomForestClassifier( # Parameters from monolithic
            n_estimators=200, max_depth=10, min_samples_split=2,
            min_samples_leaf=2, max_features="sqrt", bootstrap=True, # Monolithic has "sqrt"
            random_state=42, n_jobs=internal_n_jobs),
    }

    if XGBClassifier is not None:
        xgb_params = {
            "eval_metric": "logloss", "random_state": 42,
            "colsample_bytree": 0.8, "learning_rate": 0.15, # From monolithic build_models
            "max_depth": 3, "n_estimators": 200, "reg_alpha": 0.1, "reg_lambda": 1,
            "subsample": 1.0 # From monolithic build_models
        }
        if use_gpu_xgb:
            print("Attempting to use GPU for XGBoost in build_models.")
            xgb_params["tree_method"] = "gpu_hist"
        # The 'use_label_encoder=False' is deprecated and defaults to False.
        # If your XGBoost version is very old, you might need it. Otherwise, remove.
        # xgb_params["use_label_encoder"] = False # if needed for old versions
        all_models["XGBoost"] = XGBClassifier(**xgb_params)

    if LGBMClassifier is not None:
        lgbm_params = {
            "random_state": 42, "class_weight": "balanced", "learning_rate": 0.01, # From monolithic
            "reg_alpha": 0.1, "reg_lambda": 0.1, # From monolithic
            # num_leaves could be part of tuning or a fixed default
        }
        if use_gpu_lgbm:
            print("Attempting to use GPU for LightGBM in build_models.")
            lgbm_params["device"] = "gpu"
        all_models["LightGBM"] = LGBMClassifier(**lgbm_params)

    if selected_model_keys:
        return {k: v for k, v in all_models.items() if k in selected_model_keys}
    return all_models

# This needs to exactly match the structure and content of `param_grids` from the monolithic script
MODEL_PARAM_GRIDS = {
    "KNN": {"n_neighbors": [3, 5, 7]},
    "Logistic Regression": {"C": [0.1, 1, 10]},
    "AdaBoost": {"n_estimators": [50, 100]},
    # For LDA, monolithic `param_grids` included solver.
    # If shrinkage is tuned, solver should ideally be 'lsqr' or 'eigen'.
    # 'svd' does not use shrinkage. Monolithic had: "solver": ["svd"] which means shrinkage wouldn't apply.
    # For parity with monolithic `param_grids`:
    "LDA": {"shrinkage": [None, "auto"], "solver": ["svd"]},
    "QDA": {"reg_param": [0.0, 0.5, 1.0]},
    "Gradient Boosting Classifier": {"n_estimators": [100, 200], "max_depth": [3, 5]},
    "Extra Trees": {"n_estimators": [100, 200], "max_depth": [None, 10]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10]},
    # Conditionally add grids if models are available
    **({"XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 5]}} if XGBClassifier is not None else {}),
    **({"LightGBM": {"num_leaves": [31, 64], "learning_rate": [0.01, 0.1]}} if LGBMClassifier is not None else {}), # learning_rate added as per your first example, check monolithic.
}
# Note: Monolithic LightGBM grid was {"num_leaves": [31, 64]}. Verify against your specific monolithic version.