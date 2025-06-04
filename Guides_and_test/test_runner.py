#!/usr/bin/env python3
"""
Test Runner for Subject-Aware Model Validation Pipeline

This script provides comprehensive testing utilities for the validation pipeline
using synthetic data. It includes data generation, pipeline testing, and 
result validation.

Usage:
    python run_tests.py --quick          # Quick test with minimal data
    python run_tests.py --full           # Full test with all scenarios
    python run_tests.py --generate-only  # Only generate test data
    python run_tests.py --validate-only  # Only validate existing results
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import mlflow
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class PipelineTestRunner:
    """Comprehensive test runner for the validation pipeline."""
    
    def __init__(self, test_dir: str = "./test_pipeline/"):
        """
        Initialize test runner.
        
        Args:
            test_dir: Directory for all test files and results
        """
        self.test_dir = Path(test_dir)
        self.data_dir = self.test_dir / "data"
        self.results_dir = self.test_dir / "results"
        self.config_dir = self.test_dir / "configs"
        
        # Create directories
        for dir_path in [self.test_dir, self.data_dir, self.results_dir, self.config_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_test_data(self, scenarios: List[Dict]) -> bool:
        """
        Generate synthetic test data for all scenarios.
        
        Args:
            scenarios: List of data generation scenarios
            
        Returns:
            True if successful, False otherwise
        """
        print("🧪 Generating synthetic test data...")
        
        try:
            # Import the generator
            sys.path.append(str(Path(__file__).parent))
            from generate_test_data import SyntheticDataGenerator, validate_generated_data
            
            generator = SyntheticDataGenerator(random_state=42)
            
            for scenario in scenarios:
                print(f"\n📊 Generating scenario {scenario['dataset_idx']}...")
                
                df = generator.generate_dataset(**scenario)
                
                # Save in multiple formats
                idx = scenario['dataset_idx']
                
                # Primary format
                pickle_path = self.data_dir / f"filtered_df_{idx}GBC.pkl"
                df.to_pickle(pickle_path)
                
                # Alternative format
                alt_pickle_path = self.data_dir / f"filtered_df_{idx}.pkl"
                df.to_pickle(alt_pickle_path)
                
                # CSV for inspection
                csv_path = self.data_dir / f"synthetic_dataset_{idx}.csv"
                df.to_csv(csv_path)
                
                # Validate generated data
                validation = validate_generated_data(df)
                if not validation['valid']:
                    print(f"❌ Validation failed for scenario {idx}:")
                    for issue in validation['issues']:
                        print(f"    {issue}")
                    return False
                
                print(f"✅ Generated scenario {idx}: {validation['stats']['n_samples']} samples, "
                      f"{validation['stats']['n_participants']} participants")
            
            print("\n✅ All test data generated successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error generating test data: {e}")
            return False
    
    def create_test_configs(self, test_scenarios: List[Dict]) -> List[Path]:
        """
        Create test configuration files for different scenarios.
        
        Args:
            test_scenarios: List of test scenario configurations
            
        Returns:
            List of paths to created config files
        """
        print("⚙️ Creating test configuration files...")
        
        config_paths = []
        
        for scenario in test_scenarios:
            config_name = scenario['name']
            config_path = self.config_dir / f"{config_name}.yaml"
            
            config = {
                'experiment': {
                    'name': f"Test_{config_name}",
                    'tracking_uri': "http://localhost:5000"
                },
                'paths': {
                    'data_dir': str(self.data_dir) + "/",
                    'result_dir': str(self.results_dir / config_name) + "/"
                },
                'scaling': {
                    'scaler': scenario.get('scaler', 'StandardScaler')
                },
                'job_control': {
                    'outer_n_jobs': scenario.get('outer_n_jobs', 1),
                    'internal_n_jobs': scenario.get('internal_n_jobs', 2)
                },
                'hardware': {
                    'use_gpu_xgb': False,
                    'use_gpu_lgbm': False
                },
                'models': {
                    'selected': scenario.get('models', [
                        "Random Forest", "Logistic Regression", "XGBoost"
                    ])
                },
                'file_indices': scenario.get('file_indices', [10, 20, 30]),
                'validation': {
                    'min_participants_lopo': 2,
                    'min_participants_group3': 3,
                    'min_samples_per_class': 5
                },
                'random_state': {
                    'global_seed': 42,
                    'cv_seed': 42,
                    'model_seed': 42
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            config_paths.append(config_path)
            print(f"✅ Created config: {config_path}")
        
        return config_paths
    
    def check_mlflow_server(self) -> bool:
        """
        Check if MLflow server is running.
        
        Returns:
            True if server is accessible, False otherwise
        """
        try:
            import requests
            response = requests.get("http://localhost:5000", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_mlflow_server(self) -> Optional[subprocess.Popen]:
        """
        Start MLflow server if not running.
        
        Returns:
            Popen object if server started, None if already running
        """
        if self.check_mlflow_server():
            print("✅ MLflow server already running")
            return None
        
        try:
            print("🚀 Starting MLflow server...")
            process = subprocess.Popen(
                ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            for _ in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if self.check_mlflow_server():
                    print("✅ MLflow server started successfully")
                    return process
            
            print("❌ MLflow server failed to start within 30 seconds")
            process.kill()
            return None
            
        except Exception as e:
            print(f"❌ Error starting MLflow server: {e}")
            return None
    
    def run_pipeline_test(self, config_path: Path) -> Dict:
        """
        Run pipeline test with specified configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary with test results
        """
        print(f"\n🏃 Running pipeline test: {config_path.stem}")
        
        start_time = time.time()
        
        try:
            # Run the main pipeline
            result = subprocess.run(
                [sys.executable, "main.py", "--config", str(config_path)],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            duration = time.time() - start_time
            
            success = result.returncode == 0
            
            test_result = {
                'config': config_path.stem,
                'success': success,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            if success:
                print(f"✅ Test passed in {duration:.1f}s: {config_path.stem}")
            else:
                print(f"❌ Test failed after {duration:.1f}s: {config_path.stem}")
                print(f"Error: {result.stderr}")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Test timed out after 30 minutes: {config_path.stem}")
            return {
                'config': config_path.stem,
                'success': False,
                'duration': 1800,
                'error': 'Timeout',
                'return_code': -1
            }
        except Exception as e:
            print(f"❌ Test error: {e}")
            return {
                'config': config_path.stem,
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'return_code': -1
            }
    
    def validate_results(self, config_paths: List[Path]) -> Dict:
        """
        Validate pipeline results for all test configurations.
        
        Args:
            config_paths: List of configuration file paths
            
        Returns:
            Validation results dictionary
        """
        print("\n🔍 Validating pipeline results...")
        
        validation_results = {
            'overall_success': True,
            'config_results': {},
            'summary': {}
        }
        
        for config_path in config_paths:
            config_name = config_path.stem
            
            try:
                # Load configuration
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                result_dir = Path(config['paths']['result_dir'])
                file_indices = config['file_indices']
                
                config_validation = {
                    'csv_files_found': 0,
                    'expected_csv_files': 0,
                    'mlflow_runs_found': 0,
                    'issues': []
                }
                
                # Check CSV output files
                expected_files = []
                for idx in file_indices:
                    expected_files.extend([
                        f"computational_efficiency_idx{idx}.csv",
                        f"combined_performance_idx{idx}.csv"
                    ])
                
                config_validation['expected_csv_files'] = len(expected_files)
                
                for filename in expected_files:
                    file_path = result_dir / filename
                    if file_path.exists():
                        config_validation['csv_files_found'] += 1
                        
                        # Basic validation of CSV content
                        try:
                            df = pd.read_csv(file_path)
                            if df.empty:
                                config_validation['issues'].append(f"Empty CSV: {filename}")
                        except Exception as e:
                            config_validation['issues'].append(f"Invalid CSV {filename}: {e}")
                    else:
                        config_validation['issues'].append(f"Missing CSV: {filename}")
                
                # Check MLflow runs
                try:
                    mlflow.set_tracking_uri("http://localhost:5000")
                    experiment_name = config['experiment']['name']
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    
                    if experiment:
                        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                        config_validation['mlflow_runs_found'] = len(runs)
                        
                        if len(runs) == 0:
                            config_validation['issues'].append("No MLflow runs found")
                    else:
                        config_validation['issues'].append("MLflow experiment not found")
                        
                except Exception as e:
                    config_validation['issues'].append(f"MLflow validation error: {e}")
                
                # Overall success for this config
                config_success = (
                    config_validation['csv_files_found'] > 0 and
                    config_validation['mlflow_runs_found'] > 0 and
                    len(config_validation['issues']) == 0
                )
                
                config_validation['success'] = config_success
                validation_results['config_results'][config_name] = config_validation
                
                if not config_success:
                    validation_results['overall_success'] = False
                
                # Print summary
                print(f"{'✅' if config_success else '❌'} {config_name}:")
                print(f"    CSV files: {config_validation['csv_files_found']}/{config_validation['expected_csv_files']}")
                print(f"    MLflow runs: {config_validation['mlflow_runs_found']}")
                if config_validation['issues']:
                    for issue in config_validation['issues']:
                        print(f"    ⚠️  {issue}")
                
            except Exception as e:
                print(f"❌ Validation error for {config_name}: {e}")
                validation_results['config_results'][config_name] = {
                    'success': False,
                    'error': str(e)
                }
                validation_results['overall_success'] = False
        
        # Summary statistics
        total_configs = len(config_paths)
        successful_configs = sum(1 for result in validation_results['config_results'].values() 
                               if result.get('success', False))
        
        validation_results['summary'] = {
            'total_configs': total_configs,
            'successful_configs': successful_configs,
            'success_rate': successful_configs / total_configs if total_configs > 0 else 0
        }
        
        return validation_results
    
    def run_comprehensive_test(self, test_type: str = "quick") -> Dict:
        """
        Run comprehensive pipeline test.
        
        Args:
            test_type: Type of test ("quick", "full", "minimal")
            
        Returns:
            Complete test results
        """
        print(f"🧪 Starting comprehensive pipeline test: {test_type}")
        print("=" * 60)
        
        overall_start_time = time.time()
        
        # Define test scenarios based on test type
        if test_type == "minimal":
            data_scenarios = [
                {'dataset_idx': 10, 'n_participants': 5, 'n_sessions': 2, 'n_features': 10}
            ]
            test_configs = [
                {
                    'name': 'minimal_test',
                    'models': ['Logistic Regression'],
                    'file_indices': [10],
                    'outer_n_jobs': 1,
                    'internal_n_jobs': 1
                }
            ]
        elif test_type == "quick":
            data_scenarios = [
                {'dataset_idx': 10, 'n_participants': 8, 'n_sessions': 3, 'n_features': 20},
                {'dataset_idx': 20, 'n_participants': 12, 'n_sessions': 4, 'n_features': 25}
            ]
            test_configs = [
                {
                    'name': 'quick_test',
                    'models': ['Random Forest', 'Logistic Regression'],
                    'file_indices': [10, 20],
                    'scaler': 'StandardScaler'
                }
            ]
        else:  # full
            data_scenarios = [
                {'dataset_idx': 10, 'n_participants': 5, 'n_sessions': 3, 'n_features': 20},
                {'dataset_idx': 20, 'n_participants': 15, 'n_sessions': 4, 'n_features': 30},
                {'dataset_idx': 30, 'n_participants': 25, 'n_sessions': 5, 'n_features': 50}
            ]
            test_configs = [
                {
                    'name': 'standard_scaler_test',
                    'models': ['Random Forest', 'Logistic Regression', 'XGBoost'],
                    'file_indices': [10, 20, 30],
                    'scaler': 'StandardScaler'
                },
                {
                    'name': 'minmax_scaler_test',
                    'models': ['Random Forest', 'XGBoost'],
                    'file_indices': [10, 20],
                    'scaler': 'MinMaxScaler'
                }
            ]
        
        # Test results
        test_results = {
            'test_type': test_type,
            'start_time': time.time(),
            'data_generation': {'success': False},
            'mlflow_server': {'success': False},
            'pipeline_runs': {},
            'validation': {},
            'overall_success': False,
            'total_duration': 0
        }
        
        try:
            # Step 1: Generate test data
            print("\n📊 Step 1: Generating test data")
            data_success = self.generate_test_data(data_scenarios)
            test_results['data_generation']['success'] = data_success
            
            if not data_success:
                print("❌ Test data generation failed. Aborting.")
                return test_results
            
            # Step 2: Start MLflow server
            print("\n🚀 Step 2: Setting up MLflow server")
            mlflow_process = self.start_mlflow_server()
            mlflow_success = self.check_mlflow_server()
            test_results['mlflow_server']['success'] = mlflow_success
            
            if not mlflow_success:
                print("❌ MLflow server setup failed. Aborting.")
                return test_results
            
            # Step 3: Create test configurations
            print("\n⚙️ Step 3: Creating test configurations")
            config_paths = self.create_test_configs(test_configs)
            
            # Step 4: Run pipeline tests
            print("\n🏃 Step 4: Running pipeline tests")
            for config_path in config_paths:
                pipeline_result = self.run_pipeline_test(config_path)
                test_results['pipeline_runs'][config_path.stem] = pipeline_result
            
            # Step 5: Validate results
            print("\n🔍 Step 5: Validating results")
            validation_results = self.validate_results(config_paths)
            test_results['validation'] = validation_results
            
            # Overall success
            pipeline_success = all(
                result['success'] for result in test_results['pipeline_runs'].values()
            )
            test_results['overall_success'] = (
                data_success and mlflow_success and 
                pipeline_success and validation_results['overall_success']
            )
            
            # Cleanup MLflow process if we started it
            if mlflow_process:
                print("\n🧹 Cleaning up MLflow server")
                mlflow_process.terminate()
            
        except Exception as e:
            print(f"❌ Comprehensive test error: {e}")
            test_results['error'] = str(e)
        
        finally:
            test_results['total_duration'] = time.time() - overall_start_time
        
        return test_results
    
    def print_test_summary(self, test_results: Dict):
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        success_icon = "✅" if test_results['overall_success'] else "❌"
        print(f"{success_icon} Overall Test Result: {'PASSED' if test_results['overall_success'] else 'FAILED'}")
        print(f"⏱️  Total Duration: {test_results['total_duration']:.1f} seconds")
        print(f"🧪 Test Type: {test_results['test_type']}")
        
        # Data generation
        data_icon = "✅" if test_results['data_generation']['success'] else "❌"
        print(f"{data_icon} Data Generation: {'Success' if test_results['data_generation']['success'] else 'Failed'}")
        
        # MLflow server
        mlflow_icon = "✅" if test_results['mlflow_server']['success'] else "❌"
        print(f"{mlflow_icon} MLflow Server: {'Running' if test_results['mlflow_server']['success'] else 'Failed'}")
        
        # Pipeline runs
        print(f"\n🏃 Pipeline Runs:")
        for config_name, result in test_results['pipeline_runs'].items():
            icon = "✅" if result['success'] else "❌"
            print(f"  {icon} {config_name}: {result['duration']:.1f}s")
        
        # Validation
        if 'validation' in test_results:
            validation = test_results['validation']
            summary = validation.get('summary', {})
            print(f"\n🔍 Result Validation:")
            print(f"  Success Rate: {summary.get('successful_configs', 0)}/{summary.get('total_configs', 0)}")
            
            for config_name, result in validation.get('config_results', {}).items():
                icon = "✅" if result.get('success', False) else "❌"
                print(f"  {icon} {config_name}")
                if 'issues' in result and result['issues']:
                    for issue in result['issues'][:3]:  # Show first 3 issues
                        print(f"      ⚠️  {issue}")

def main():
    """Main function for test runner."""
    parser = argparse.ArgumentParser(description="Test runner for validation pipeline")
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    parser.add_argument('--full', action='store_true', help='Run full test suite')
    parser.add_argument('--minimal', action='store_true', help='Run minimal test')
    parser.add_argument('--generate-only', action='store_true', help='Only generate test data')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing results')
    parser.add_argument('--test-dir', type=str, default='./test_pipeline/', help='Test directory')
    
    args = parser.parse_args()
    
    # Determine test type
    if args.minimal:
        test_type = "minimal"
    elif args.full:
        test_type = "full"
    elif args.quick or not (args.generate_only or args.validate_only):
        test_type = "quick"
    else:
        test_type = "quick"  # default
    
    # Initialize test runner
    runner = PipelineTestRunner(test_dir=args.test_dir)
    
    if args.generate_only:
        # Only generate test data
        scenarios = [
            {'dataset_idx': 10, 'n_participants': 8, 'n_sessions': 3, 'n_features': 20},
            {'dataset_idx': 20, 'n_participants': 12, 'n_sessions': 4, 'n_features': 25}
        ]
        success = runner.generate_test_data(scenarios)
        print(f"{'✅' if success else '❌'} Data generation {'completed' if success else 'failed'}")
        
    elif args.validate_only:
        # Only validate existing results
        print("🔍 Validating existing results...")
        # This would need existing config files - simplified for now
        print("⚠️  Validation-only mode requires existing config files")
        
    else:
        # Run comprehensive test
        test_results = runner.run_comprehensive_test(test_type)
        runner.print_test_summary(test_results)
        
        # Exit with appropriate code
        sys.exit(0 if test_results['overall_success'] else 1)

if __name__ == "__main__":
    main()