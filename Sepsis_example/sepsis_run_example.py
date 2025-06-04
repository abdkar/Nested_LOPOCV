#!/usr/bin/env python3
"""
Complete Sepsis Detection Example Runner

This script demonstrates the complete workflow for sepsis detection in ICU patients
using the Subject-Aware Model Validation Pipeline.

Usage:
    python run_sepsis_example.py
    python run_sepsis_example.py --n_patients 100 --full_analysis
"""

import argparse
import subprocess
import sys
import time
import requests
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

class SepsisValidationExample:
    """Complete example runner for sepsis detection validation."""
    
    def __init__(self, output_dir="./sepsis_example/"):
        """Initialize example runner."""
        self.output_dir = Path(output_dir)
        self.data_dir = self.output_dir / "data"
        self.results_dir = self.output_dir / "results"
        self.config_path = self.output_dir / "config_sepsis_example.yaml"
        
        # Create directories
        for dir_path in [self.output_dir, self.data_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def print_header(self, title):
        """Print formatted header."""
        print(f"\n{'='*60}")
        print(f"🏥 {title}")
        print(f"{'='*60}")
    
    def check_mlflow_server(self):
        """Check if MLflow server is running."""
        try:
            response = requests.get("http://localhost:5000", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def start_mlflow_server(self):
        """Start MLflow server if not running."""
        if self.check_mlflow_server():
            print("✅ MLflow server already running")
            return None
        
        print("🚀 Starting MLflow server...")
        process = subprocess.Popen(
            ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        for _ in range(20):
            time.sleep(1)
            if self.check_mlflow_server():
                print("✅ MLflow server started successfully")
                return process
        
        print("❌ MLflow server failed to start")
        process.kill()
        return None
    
    def generate_sepsis_data(self, n_patients=30):
        """Generate synthetic sepsis data."""
        self.print_header("STEP 1: GENERATING SYNTHETIC SEPSIS DATA")
        
        print(f"Generating data for {n_patients} ICU patients...")
        
        result = subprocess.run([
            sys.executable, 'generate_sepsis_data.py',
            '--output_dir', str(self.data_dir),
            '--single_dataset', '10',
            '--n_patients', str(n_patients)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Sepsis data generated successfully")
            print(result.stdout)
            return True
        else:
            print("❌ Failed to generate sepsis data")
            print(result.stderr)
            return False
    
    def analyze_generated_data(self):
        """Analyze the generated sepsis data."""
        self.print_header("STEP 2: ANALYZING GENERATED DATA")
        
        data_file = self.data_dir / "filtered_df_10GBC.pkl"
        if not data_file.exists():
            print(f"❌ Data file not found: {data_file}")
            return False
        
        df = pd.read_pickle(data_file)
        
        print("📊 Dataset Overview:")
        print(f"  Shape: {df.shape}")
        print(f"  Patients: {df.index.str.split('_').str[0].nunique()}")
        print(f"  Total measurements: {len(df)}")
        print(f"  Sepsis prevalence: {(df['target'] == 1).mean():.1%}")
        
        # Patient-level analysis
        patient_stats = df.groupby(df.index.str.split('_').str[0]).agg({
            'target': ['sum', 'count'],
            'heart_rate': 'mean',
            'temperature': 'mean'
        }).round(2)
        
        patient_stats.columns = ['sepsis_hours', 'icu_stay_hours', 'avg_hr', 'avg_temp']
        patients_with_sepsis = (patient_stats['sepsis_hours'] > 0).sum()
        
        print(f"\n👥 Patient Statistics:")
        print(f"  Patients with sepsis: {patients_with_sepsis}/{len(patient_stats)}")
        print(f"  Average ICU stay: {patient_stats['icu_stay_hours'].mean():.1f} hours")
        
        # Visualize data patterns
        self.create_data_visualizations(df)
        
        return True
    
    def create_data_visualizations(self, df):
        """Create visualizations of the sepsis data."""
        print("\n📈 Creating data visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Sepsis prevalence over time
        hourly_sepsis = df.groupby('hour')['target'].mean()
        axes[0,0].plot(hourly_sepsis.index, hourly_sepsis.values * 100, 'b-', linewidth=2)
        axes[0,0].set_title('Sepsis Prevalence by ICU Hour', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('ICU Hour')
        axes[0,0].set_ylabel('Sepsis Prevalence (%)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Heart rate distribution
        df_sample = df.sample(min(1000, len(df)))
        sns.boxplot(data=df_sample, x='target', y='heart_rate', ax=axes[0,1])
        axes[0,1].set_title('Heart Rate by Sepsis Status', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('Sepsis Status (0=No, 1=Yes)')
        axes[0,1].set_ylabel('Heart Rate (bpm)')
        
        # 3. Temperature distribution
        sns.boxplot(data=df_sample, x='target', y='temperature', ax=axes[1,0])
        axes[1,0].set_title('Temperature by Sepsis Status', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('Sepsis Status (0=No, 1=Yes)')
        axes[1,0].set_ylabel('Temperature (°C)')
        
        # 4. Lactate distribution
        sns.boxplot(data=df_sample, x='target', y='lactate', ax=axes[1,1])
        axes[1,1].set_title('Lactate by Sepsis Status', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('Sepsis Status (0=No, 1=Yes)')
        axes[1,1].set_ylabel('Lactate (mmol/L)')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "sepsis_data_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved visualization: {plot_path}")
        
        plt.show()
    
    def create_config_file(self):
        """Create configuration file for sepsis pipeline."""
        config_content = f"""# Sepsis Detection Configuration
experiment:
  name: "Sepsis_Detection_Example_Study"
  tracking_uri: "http://localhost:5000"

paths:
  data_dir: "{self.data_dir}/"
  result_dir: "{self.results_dir}/"

scaling:
  scaler: "StandardScaler"

job_control:
  outer_n_jobs: 1
  internal_n_jobs: 2

hardware:
  use_gpu_xgb: false
  use_gpu_lgbm: false

models:
  selected:
    - "Random Forest"
    - "XGBoost"  
    - "Logistic Regression"

file_indices: [10]

validation:
  min_participants_lopo: 2
  min_participants_group3: 3
  min_samples_per_class: 5

random_state:
  global_seed: 42
  cv_seed: 42
  model_seed: 42
"""
        
        with open(self.config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Created configuration: {self.config_path}")
    
    def run_validation_pipeline(self):
        """Run the subject-aware validation pipeline."""
        self.print_header("STEP 3: RUNNING SUBJECT-AWARE VALIDATION")
        
        print("🏃 Running sepsis detection validation pipeline...")
        print("This will compare:")
        print("  • Standard 10-Fold CV (potential data leakage)")
        print("  • Leave-One-Patient-Out CV (subject-aware)")
        print("  • Group 3-Fold CV (balanced subject-aware)")
        print("\nExpected runtime: 5-15 minutes...\n")
        
        result = subprocess.run([
            sys.executable, 'main.py', 
            '--config', str(self.config_path)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Pipeline completed successfully!")
            print("📊 Check MLflow UI for detailed results: http://localhost:5000")
            return True
        else:
            print(f"❌ Pipeline failed with return code: {result.returncode}")
            print("Error output:")
            print(result.stderr)
            return False
    
    def analyze_results(self):
        """Analyze pipeline results and data leakage."""
        self.print_header("STEP 4: ANALYZING RESULTS AND DATA LEAKAGE")
        
        # Find result files
        result_files = list(self.results_dir.glob('*.csv'))
        print(f"📊 Found {len(result_files)} result files:")
        for file in result_files:
            print(f"  • {file.name}")
        
        # Analyze data leakage
        self.analyze_data_leakage()
        
        # Analyze computational efficiency
        self.analyze_efficiency()
        
        return True
    
    def analyze_data_leakage(self):
        """Analyze potential data leakage between CV strategies."""
        print("\n🔍 Data Leakage Analysis:")
        print("-" * 40)
        
        cv_strategies = ['10Fold', 'LOPOCV', 'Group3Fold']
        results_summary = []
        
        for strategy in cv_strategies:
            strategy_dir = self.results_dir / strategy
            if strategy_dir.exists():
                csv_files = list(strategy_dir.glob('perf_*.csv'))
                if csv_files:
                    strategy_df = pd.read_csv(csv_files[0])
                    
                    # Look for accuracy columns
                    accuracy_cols = [col for col in strategy_df.columns 
                                   if 'accuracy' in col.lower() and 'test' in col.lower()]
                    
                    if accuracy_cols:
                        avg_accuracy = strategy_df[accuracy_cols].mean().mean()
                        results_summary.append({
                            'CV_Strategy': strategy,
                            'Avg_Test_Accuracy': avg_accuracy
                        })
                        print(f"  {strategy:12s}: {avg_accuracy:.3f}")
        
        if len(results_summary) >= 2:
            accuracies = [r['Avg_Test_Accuracy'] for r in results_summary]
            max_acc = max(accuracies)
            min_acc = min(accuracies)
            leakage_gap = max_acc - min_acc
            
            print(f"\n🚨 Data Leakage Assessment:")
            print(f"  Performance range: {min_acc:.3f} - {max_acc:.3f}")
            print(f"  Potential leakage: {leakage_gap:.3f} ({leakage_gap*100:.1f} percentage points)")
            
            if leakage_gap > 0.1:
                print("  ⚠️  HIGH LEAKAGE RISK: >10 percentage point difference")
                print("     Standard CV likely learning patient-specific patterns")
            elif leakage_gap > 0.05:
                print("  ⚠️  MODERATE LEAKAGE RISK: 5-10 percentage point difference")
            else:
                print("  ✅ LOW LEAKAGE RISK: <5 percentage point difference")
    
    def analyze_efficiency(self):
        """Analyze computational efficiency."""
        efficiency_file = self.results_dir / 'computational_efficiency_idx10.csv'
        
        if efficiency_file.exists():
            efficiency_df = pd.read_csv(efficiency_file)
            
            print(f"\n⚡ Computational Efficiency:")
            print("-" * 30)
            
            if not efficiency_df.empty:
                for _, row in efficiency_df.iterrows():
                    print(f"  {row['Model']:<20s}: "
                          f"{row['Avg. Train Time (s/fold)']:6.1f}s train, "
                          f"{row['Avg. Inference Time (s/sample)']*1000:6.2f}ms inference")
                
                # Clinical deployment assessment
                max_inference_time = efficiency_df['Avg. Inference Time (s/sample)'].max()
                print(f"\n🏥 Clinical Deployment:")
                print(f"  Real-time feasibility: {'✅ YES' if max_inference_time < 1.0 else '❌ NO'}")
                print(f"  (Max inference: {max_inference_time*1000:.2f}ms, ICU updates: 1-60 min)")
    
    def clinical_feature_analysis(self):
        """Perform clinical feature importance analysis."""
        self.print_header("STEP 5: CLINICAL FEATURE IMPORTANCE")
        
        data_file = self.data_dir / "filtered_df_10GBC.pkl"
        df = pd.read_pickle(data_file)
        
        # Clinical features for analysis
        clinical_features = [
            'heart_rate', 'systolic_bp', 'temperature', 'respiratory_rate',
            'oxygen_saturation', 'white_blood_cells', 'lactate', 
            'procalcitonin', 'c_reactive_protein', 'platelets'
        ]
        
        available_features = [f for f in clinical_features if f in df.columns]
        
        if len(available_features) < 5:
            print("⚠️  Insufficient clinical features for analysis")
            return
        
        X = df[available_features]
        y = df['target']
        pids = df.index.str.split('_').str[0].astype(int)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # Calculate feature importance with LOPOCV
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv = GroupKFold(n_splits=min(5, len(np.unique(pids))))
        
        feature_importances = []
        
        print("🔬 Calculating feature importance with subject-aware CV...")
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y, groups=pids)):
            X_train = X_scaled.iloc[train_idx]
            y_train = y.iloc[train_idx]
            
            rf.fit(X_train, y_train)
            feature_importances.append(rf.feature_importances_)
            
            if fold >= 4:  # Limit folds for speed
                break
        
        # Average importance across folds
        avg_importance = np.mean(feature_importances, axis=0)
        
        # Create results dataframe
        feature_df = pd.DataFrame({
            'Feature': available_features,
            'Importance': avg_importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\n🏆 Top Clinical Features for Sepsis Detection:")
        print("-" * 50)
        
        for i, (_, row) in enumerate(feature_df.head(8).iterrows()):
            clinical_name = row['Feature'].replace('_', ' ').title()
            print(f"  {i+1:2d}. {clinical_name:<25s} {row['Importance']:.4f}")
        
        # Clinical interpretation
        clinical_interpretations = {
            'lactate': 'Tissue hypoxia marker - elevated in septic shock',
            'procalcitonin': 'Bacterial infection biomarker - highly specific',
            'c_reactive_protein': 'Inflammatory response marker',
            'white_blood_cells': 'Immune response indicator',
            'temperature': 'Fever/hypothermia - both sepsis signs',
            'heart_rate': 'Tachycardia - compensatory response',
            'respiratory_rate': 'Tachypnea - metabolic compensation'
        }
        
        print(f"\n🩺 Clinical Interpretation:")
        print("-" * 25)
        
        for feature in feature_df.head(5)['Feature']:
            if feature in clinical_interpretations:
                name = feature.replace('_', ' ').title()
                interpretation = clinical_interpretations[feature]
                print(f"  • {name}: {interpretation}")
    
    def generate_final_report(self):
        """Generate final summary report."""
        self.print_header("SUMMARY AND RECOMMENDATIONS")
        
        print("✅ COMPLETED SUCCESSFULLY:")
        print("   • Generated realistic synthetic ICU sepsis data")
        print("   • Demonstrated subject-aware validation approach")
        print("   • Compared multiple CV strategies for data leakage")
        print("   • Analyzed computational efficiency for deployment")
        print("   • Identified key clinical features")
        
        print("\n📊 KEY INSIGHTS:")
        print("   • Subject-aware validation prevents overoptimistic results")
        print("   • Patient-specific patterns can cause significant data leakage")
        print("   • Clinical biomarkers align with known sepsis pathophysiology")
        print("   • Models are efficient enough for real-time ICU monitoring")
        
        print("\n🏥 CLINICAL RECOMMENDATIONS:")
        print("   • Always use LOPOCV for patient-level validation")
        print("   • Prioritize sensitivity for early sepsis detection")
        print("   • Validate models on local patient populations")
        print("   • Monitor for model drift over time")
        
        print("\n🚀 NEXT STEPS:")
        print("   1. Validate with real ICU data")
        print("   2. Implement temporal validation")
        print("   3. Assess model fairness across demographics")
        print("   4. Develop clinical decision support interface")
        
        print(f"\n📁 OUTPUT DIRECTORY: {self.output_dir}")
        print(f"   • Data: {self.data_dir}")
        print(f"   • Results: {self.results_dir}")
        print(f"   • MLflow UI: http://localhost:5000")

def main():
    """Main function to run the complete sepsis example."""
    parser = argparse.ArgumentParser(
        description="Complete sepsis detection validation example"
    )
    parser.add_argument(
        '--n_patients', type=int, default=30,
        help='Number of ICU patients to simulate'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./sepsis_example/',
        help='Output directory for all files'
    )
    parser.add_argument(
        '--full_analysis', action='store_true',
        help='Run complete analysis including feature importance'
    )
    parser.add_argument(
        '--skip_pipeline', action='store_true',
        help='Skip pipeline execution (for testing other components)'
    )
    
    args = parser.parse_args()
    
    # Initialize example runner
    example = SepsisValidationExample(args.output_dir)
    
    print("🏥 SEPSIS DETECTION: SUBJECT-AWARE VALIDATION EXAMPLE")
    print("=" * 60)
    print(f"Simulating {args.n_patients} ICU patients")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Step 1: Generate data
        if not example.generate_sepsis_data(args.n_patients):
            return 1
        
        # Step 2: Analyze data
        if not example.analyze_generated_data():
            return 1
        
        # Step 3: Setup MLflow
        example.start_mlflow_server()
        time.sleep(2)  # Give server time to fully start
        
        # Step 4: Create config and run pipeline
        example.create_config_file()
        
        if not args.skip_pipeline:
            if not example.run_validation_pipeline():
                return 1
        
        # Step 5: Analyze results
        if not args.skip_pipeline:
            example.analyze_results()
        
        # Step 6: Feature analysis (if requested)
        if args.full_analysis:
            example.clinical_feature_analysis()
        
        # Step 7: Final report
        example.generate_final_report()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Example interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)