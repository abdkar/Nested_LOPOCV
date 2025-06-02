# mlflow_pipeline/tests/test_pipeline.py
import unittest
import pandas as pd
from src.__init__1 import build_models, get_hyperparams, load_data, metric_row, save_tmp_csv
from sklearn.datasets import make_classification
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

class TestPipeline(unittest.TestCase):
    def test_build_models(self):
        models = build_models()
        self.assertTrue("KNN" in models)
        self.assertTrue(hasattr(models["KNN"], 'fit'))
    
    def test_get_hyperparams(self):
        params = get_hyperparams()
        self.assertIn("KNN", params)
        self.assertIsInstance(params["KNN"], dict)
    
    def test_load_data(self):
        # Create a dummy pickle file
        df = pd.DataFrame({"target": [0,1], "col1": [1,2], "index": ["106_1", "108_2"]})
        pkl_path = "/tmp/test.pkl"
        df.to_pickle(pkl_path)
        loaded_df = load_data(pkl_path)
        self.assertIn("Test_ID", loaded_df.columns)
        os.remove(pkl_path)
    
    def test_metric_row(self):
        pred = pd.Series([1,0,1])
        y_true = pd.Series([1,0,0])
        row = metric_row(pred, y_true, "test_pid", 0.9, 0.8, 1, 0.5, 1234)
        self.assertIn("Test Accuracy", row)
    
    def test_save_tmp_csv(self):
        df = pd.DataFrame({"A": [1,2], "B": [3,4]})
        path = save_tmp_csv(df, "test")
        self.assertTrue(os.path.exists(path))
        os.remove(path)

if __name__ == '__main__':
    unittest.main()
