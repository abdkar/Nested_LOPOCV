#!/usr/bin/env python3
"""
Sepsis Detection Data Generator for ICU Patients

This script generates realistic synthetic ICU patient data for sepsis detection
using repeated measures (hourly vital signs and lab values over ICU stay).

The generated data simulates:
- Multiple ICU patients with different baseline health status
- Hourly measurements during ICU stay (12-72 hours)
- Realistic vital signs and laboratory values
- Sepsis onset patterns with physiological changes
- Patient-specific baseline variations that could lead to data leakage

Clinical Context:
- Sepsis is a life-threatening condition requiring early detection
- Patients have different baseline vital signs and lab values
- Sepsis causes characteristic changes in vital signs and inflammatory markers
- Early detection (within 6 hours) improves outcomes significantly

Usage:
    python generate_sepsis_data.py
    python generate_sepsis_data.py --n_patients 100 --output_dir ./sepsis_data/
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Valid patient IDs (using realistic ICU patient ID format)
VALID_PATIENT_IDS = [
    # ICU Patient IDs (format: ICU unit + patient number)
    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
    1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020,
    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
    2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,
    3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010,
    3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020,
    # Add more as needed for larger studies
    4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010,
    5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010
]

class SepsisDataGenerator:
    """Generates realistic synthetic ICU sepsis detection data."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the sepsis data generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define normal ranges for vital signs and lab values
        self.normal_ranges = {
            # Vital Signs
            'heart_rate': (60, 100),           # beats/min
            'systolic_bp': (90, 140),          # mmHg
            'diastolic_bp': (60, 90),          # mmHg
            'respiratory_rate': (12, 20),      # breaths/min
            'temperature': (36.1, 37.2),       # Celsius
            'oxygen_saturation': (95, 100),    # %
            
            # Laboratory Values
            'white_blood_cells': (4.0, 11.0),  # x10^3/µL
            'lactate': (0.5, 2.0),             # mmol/L
            'procalcitonin': (0.0, 0.1),       # ng/mL
            'c_reactive_protein': (0, 3),       # mg/L
            'platelets': (150, 450),           # x10^3/µL
            'creatinine': (0.6, 1.2),          # mg/dL
            'bilirubin': (0.2, 1.0),           # mg/dL
            
            # Derived Measures
            'mean_arterial_pressure': (70, 105), # mmHg
            'shock_index': (0.5, 0.7),          # HR/SBP
        }
        
    def generate_patient_characteristics(self, n_patients: int) -> Dict:
        """
        Generate baseline characteristics for ICU patients.
        
        Args:
            n_patients: Number of patients to generate
            
        Returns:
            Dictionary with patient characteristics
        """
        if n_patients > len(VALID_PATIENT_IDS):
            raise ValueError(f"Cannot generate {n_patients} patients. "
                           f"Maximum available: {len(VALID_PATIENT_IDS)}")
        
        patient_ids = np.random.choice(
            VALID_PATIENT_IDS, 
            size=n_patients, 
            replace=False
        )
        
        # Generate patient characteristics
        characteristics = {
            'patient_ids': patient_ids,
            'age': np.random.normal(65, 15, n_patients).clip(18, 95),
            'gender': np.random.choice(['M', 'F'], n_patients),
            'apache_ii_score': np.random.normal(15, 8, n_patients).clip(0, 40),  # Severity score
            'charlson_index': np.random.poisson(2, n_patients).clip(0, 10),      # Comorbidity score
            
            # Baseline health status (affects normal ranges)
            'baseline_health': np.random.beta(3, 2, n_patients),  # 0=poor, 1=excellent
            
            # Sepsis risk factors
            'diabetes': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
            'immunocompromised': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
            'chronic_kidney_disease': np.random.choice([0, 1], n_patients, p=[0.75, 0.25]),
            
            # ICU stay characteristics
            'icu_stay_hours': np.random.normal(48, 24, n_patients).clip(12, 168),  # 12h to 7 days
            'sepsis_onset_probability': np.random.beta(1, 4, n_patients),  # Most patients don't develop sepsis
            
            # Individual variation in vital signs (patient-specific normals)
            'hr_baseline_offset': np.random.normal(0, 10, n_patients),
            'sbp_baseline_offset': np.random.normal(0, 15, n_patients),
            'temp_baseline_offset': np.random.normal(0, 0.3, n_patients),
            'noise_level': np.random.gamma(1, 0.1, n_patients),  # Individual measurement noise
        }
        
        return characteristics
    
    def generate_sepsis_trajectory(self, patient_char: Dict, patient_idx: int) -> Tuple[np.ndarray, List[int]]:
        """
        Generate sepsis onset trajectory for a patient.
        
        Args:
            patient_char: Patient characteristics dictionary
            patient_idx: Index of current patient
            
        Returns:
            Tuple of (sepsis_status_over_time, sepsis_hours)
        """
        icu_hours = int(patient_char['icu_stay_hours'][patient_idx])
        sepsis_prob = patient_char['sepsis_onset_probability'][patient_idx]
        
        # Determine if patient develops sepsis
        develops_sepsis = np.random.random() < sepsis_prob
        
        sepsis_status = np.zeros(icu_hours, dtype=int)
        sepsis_hours = []
        
        if develops_sepsis:
            # Sepsis onset typically occurs 6-48 hours into ICU stay
            onset_hour = int(np.random.uniform(6, min(48, icu_hours * 0.7)))
            
            # Sepsis persists once it starts (with treatment, some patients recover)
            recovery_prob = 0.6  # 60% recover during ICU stay
            if np.random.random() < recovery_prob:
                # Patient recovers after 12-48 hours of treatment
                recovery_duration = int(np.random.uniform(12, 48))
                recovery_hour = min(onset_hour + recovery_duration, icu_hours)
                sepsis_status[onset_hour:recovery_hour] = 1
                sepsis_hours = list(range(onset_hour, recovery_hour))
            else:
                # Sepsis persists for remainder of ICU stay
                sepsis_status[onset_hour:] = 1
                sepsis_hours = list(range(onset_hour, icu_hours))
        
        return sepsis_status, sepsis_hours
    
    def generate_vital_signs_and_labs(
        self, 
        patient_char: Dict, 
        patient_idx: int,
        sepsis_status: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate realistic vital signs and lab values with sepsis-related changes.
        
        Args:
            patient_char: Patient characteristics
            patient_idx: Current patient index
            sepsis_status: Array indicating sepsis status for each hour
            
        Returns:
            DataFrame with hourly measurements
        """
        icu_hours = len(sepsis_status)
        
        # Patient-specific baseline adjustments
        age = patient_char['age'][patient_idx]
        baseline_health = patient_char['baseline_health'][patient_idx]
        apache_score = patient_char['apache_ii_score'][patient_idx]
        noise_level = patient_char['noise_level'][patient_idx]
        
        # Individual baseline offsets
        hr_offset = patient_char['hr_baseline_offset'][patient_idx]
        sbp_offset = patient_char['sbp_baseline_offset'][patient_idx]
        temp_offset = patient_char['temp_baseline_offset'][patient_idx]
        
        # Initialize arrays for each measurement
        measurements = {}
        
        for hour in range(icu_hours):
            is_septic = sepsis_status[hour]
            
            # Time-based trends (patient condition changes over ICU stay)
            time_factor = hour / icu_hours
            fatigue_factor = 1 + time_factor * 0.1  # Gradual deterioration
            
            # Generate vital signs
            # Heart Rate (increases with sepsis, age, and severity)
            hr_base = 75 + hr_offset + (age - 65) * 0.2 + apache_score * 0.5
            if is_septic:
                hr_base += np.random.normal(25, 10)  # Tachycardia in sepsis
            hr = hr_base * fatigue_factor + np.random.normal(0, 5 * noise_level)
            hr = np.clip(hr, 40, 180)
            
            # Blood Pressure (decreases with sepsis)
            sbp_base = 120 + sbp_offset - (age - 65) * 0.3 - apache_score * 0.8
            if is_septic:
                sbp_base -= np.random.normal(20, 10)  # Hypotension in sepsis
            sbp = sbp_base + np.random.normal(0, 8 * noise_level)
            sbp = np.clip(sbp, 60, 200)
            
            dbp = sbp * 0.65 + np.random.normal(0, 5 * noise_level)
            dbp = np.clip(dbp, 40, 120)
            
            # Temperature (fever in sepsis, but can be hypothermia in severe cases)
            temp_base = 36.7 + temp_offset
            if is_septic:
                if np.random.random() < 0.8:  # 80% have fever
                    temp_base += np.random.normal(2.0, 0.8)  # Fever
                else:  # 20% have hypothermia (worse prognosis)
                    temp_base -= np.random.normal(1.5, 0.5)  # Hypothermia
            temperature = temp_base + np.random.normal(0, 0.2 * noise_level)
            temperature = np.clip(temperature, 32, 42)
            
            # Respiratory Rate (increases with sepsis)
            rr_base = 16 + apache_score * 0.2
            if is_septic:
                rr_base += np.random.normal(8, 3)  # Tachypnea
            respiratory_rate = rr_base + np.random.normal(0, 2 * noise_level)
            respiratory_rate = np.clip(respiratory_rate, 8, 45)
            
            # Oxygen Saturation (decreases with sepsis)
            spo2_base = 98 - apache_score * 0.3
            if is_septic:
                spo2_base -= np.random.normal(4, 2)  # Hypoxemia
            oxygen_saturation = spo2_base + np.random.normal(0, 1 * noise_level)
            oxygen_saturation = np.clip(oxygen_saturation, 70, 100)
            
            # Laboratory Values (measured every 6-12 hours, so some interpolation)
            lab_measurement_prob = 0.2 if hour % 6 != 0 else 1.0  # Labs every 6 hours mostly
            
            if np.random.random() < lab_measurement_prob or hour == 0:
                # White Blood Cells (elevated in sepsis)
                wbc_base = 7.0 - baseline_health * 2
                if is_septic:
                    if np.random.random() < 0.9:  # Usually elevated
                        wbc_base += np.random.normal(8, 4)  # Leukocytosis
                    else:  # Sometimes low (worse prognosis)
                        wbc_base -= np.random.normal(3, 1)  # Leukopenia
                wbc = wbc_base + np.random.normal(0, 1 * noise_level)
                wbc = np.clip(wbc, 0.5, 50)
                
                # Lactate (elevated in sepsis due to tissue hypoxia)
                lactate_base = 1.0 + apache_score * 0.05
                if is_septic:
                    lactate_base += np.random.normal(3, 1.5)  # Elevated lactate
                lactate = lactate_base + np.random.normal(0, 0.3 * noise_level)
                lactate = np.clip(lactate, 0.3, 15)
                
                # Procalcitonin (biomarker for bacterial sepsis)
                pct_base = 0.05
                if is_septic:
                    pct_base += np.random.lognormal(1, 1)  # Highly elevated in sepsis
                procalcitonin = pct_base + np.random.normal(0, 0.1 * noise_level)
                procalcitonin = np.clip(procalcitonin, 0, 50)
                
                # C-Reactive Protein (inflammatory marker)
                crp_base = 2 + apache_score * 0.3
                if is_septic:
                    crp_base += np.random.normal(80, 30)  # Highly elevated
                c_reactive_protein = crp_base + np.random.normal(0, 5 * noise_level)
                c_reactive_protein = np.clip(c_reactive_protein, 0, 300)
                
                # Platelets (often decreased in sepsis)
                platelets_base = 250 - apache_score * 5
                if is_septic:
                    platelets_base -= np.random.normal(80, 40)  # Thrombocytopenia
                platelets = platelets_base + np.random.normal(0, 20 * noise_level)
                platelets = np.clip(platelets, 10, 600)
                
                # Store lab values for interpolation
                if hour == 0:
                    last_labs = {
                        'wbc': wbc, 'lactate': lactate, 'procalcitonin': procalcitonin,
                        'c_reactive_protein': c_reactive_protein, 'platelets': platelets
                    }
                else:
                    # Interpolate from last measurement
                    for lab_name in ['wbc', 'lactate', 'procalcitonin', 'c_reactive_protein', 'platelets']:
                        last_labs[lab_name] = locals()[lab_name]
            else:
                # Use last measured values with small random variation
                wbc = last_labs['wbc'] + np.random.normal(0, 0.3)
                lactate = last_labs['lactate'] + np.random.normal(0, 0.1)
                procalcitonin = last_labs['procalcitonin'] + np.random.normal(0, 0.05)
                c_reactive_protein = last_labs['c_reactive_protein'] + np.random.normal(0, 2)
                platelets = last_labs['platelets'] + np.random.normal(0, 10)
            
            # Derived measures
            mean_arterial_pressure = (sbp + 2 * dbp) / 3
            shock_index = hr / sbp if sbp > 0 else 5  # HR/SBP ratio
            
            # Store measurements for this hour
            hour_data = {
                'heart_rate': hr,
                'systolic_bp': sbp,
                'diastolic_bp': dbp,
                'mean_arterial_pressure': mean_arterial_pressure,
                'respiratory_rate': respiratory_rate,
                'temperature': temperature,
                'oxygen_saturation': oxygen_saturation,
                'shock_index': shock_index,
                'white_blood_cells': wbc,
                'lactate': lactate,
                'procalcitonin': procalcitonin,
                'c_reactive_protein': c_reactive_protein,
                'platelets': platelets,
                'sepsis_status': is_septic
            }
            
            for key, value in hour_data.items():
                if key not in measurements:
                    measurements[key] = []
                measurements[key].append(value)
        
        return pd.DataFrame(measurements)
    
    def generate_dataset(
        self,
        n_patients: int = 50,
        dataset_idx: int = 10
    ) -> pd.DataFrame:
        """
        Generate a complete synthetic sepsis detection dataset.
        
        Args:
            n_patients: Number of ICU patients
            dataset_idx: Dataset index for filename
            
        Returns:
            Complete synthetic sepsis dataset
        """
        print(f"Generating synthetic sepsis dataset {dataset_idx}...")
        print(f"  ICU Patients: {n_patients}")
        
        # Generate patient characteristics
        patient_chars = self.generate_patient_characteristics(n_patients)
        
        all_data = []
        total_sepsis_hours = 0
        total_hours = 0
        
        for p_idx, patient_id in enumerate(patient_chars['patient_ids']):
            # Generate sepsis trajectory for this patient
            sepsis_status, sepsis_hours = self.generate_sepsis_trajectory(patient_chars, p_idx)
            
            # Generate vital signs and labs
            patient_data = self.generate_vital_signs_and_labs(patient_chars, p_idx, sepsis_status)
            
            # Add patient metadata
            patient_data['patient_id'] = patient_id
            patient_data['age'] = patient_chars['age'][p_idx]
            patient_data['gender'] = patient_chars['gender'][p_idx]
            patient_data['apache_ii_score'] = patient_chars['apache_ii_score'][p_idx]
            patient_data['hour'] = range(len(sepsis_status))
            
            # Create index in expected format (patient_id_hour)
            patient_data.index = [f"{patient_id}_{hour}" for hour in range(len(sepsis_status))]
            
            all_data.append(patient_data)
            total_sepsis_hours += sum(sepsis_status)
            total_hours += len(sepsis_status)
        
        # Combine all patient data
        df = pd.concat(all_data, ignore_index=False)
        
        # Rename sepsis_status to target for pipeline compatibility
        df['target'] = df['sepsis_status'].astype(int)
        df = df.drop('sepsis_status', axis=1)
        
        # Add some realistic missing values (ICU equipment failures, etc.)
        missing_rate = 0.01  # 1% missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['target', 'patient_id', 'hour', 'age', 'apache_ii_score']]
        
        for col in numeric_cols:
            mask = np.random.random(len(df)) < missing_rate
            df.loc[mask, col] = np.nan
        
        # Forward fill missing values (realistic for continuous monitoring)
        df[numeric_cols] = df.groupby('patient_id')[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        sepsis_prevalence = (df['target'] == 1).mean()
        
        print(f"  Total measurements: {len(df)}")
        print(f"  Sepsis prevalence: {sepsis_prevalence:.1%} ({sum(df['target'])} septic hours)")
        print(f"  Patients with sepsis: {df[df['target']==1]['patient_id'].nunique()}")
        print(f"  Average ICU stay: {df.groupby('patient_id').size().mean():.1f} hours")
        
        return df

def create_sepsis_scenarios() -> List[Dict]:
    """
    Create different sepsis detection scenarios for testing.
    
    Returns:
        List of dataset configurations
    """
    scenarios = [
        {
            # Scenario 1: Small ICU (pilot study)
            'dataset_idx': 10,
            'n_patients': 25,
        },
        {
            # Scenario 2: Medium ICU study
            'dataset_idx': 20,
            'n_patients': 50,
        },
        {
            # Scenario 3: Large multi-center study
            'dataset_idx': 30,
            'n_patients': 100,
        },
        {
            # Scenario 4: Very small dataset (edge case)
            'dataset_idx': 40,
            'n_patients': 15,
        },
        {
            # Scenario 5: Large single-center study
            'dataset_idx': 50,
            'n_patients': 75,
        }
    ]
    
    return scenarios

def validate_sepsis_data(df: pd.DataFrame) -> Dict:
    """
    Validate generated sepsis detection data.
    
    Args:
        df: Generated sepsis dataset
        
    Returns:
        Validation results dictionary
    """
    results = {
        'valid': True,
        'issues': [],
        'stats': {}
    }
    
    # Check required columns
    required_cols = ['target', 'heart_rate', 'systolic_bp', 'temperature', 'white_blood_cells']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        results['valid'] = False
        results['issues'].append(f"Missing required columns: {missing_cols}")
    
    # Check index format
    try:
        pids = df.index.str.split('_').str[0].astype(int)
        hours = df.index.str.split('_').str[1].astype(int)
        results['stats']['extracted_ids'] = True
    except:
        results['valid'] = False
        results['issues'].append("Index format incorrect (should be 'patient_id_hour')")
    
    # Check patient IDs
    if results['stats'].get('extracted_ids'):
        invalid_pids = set(pids.unique()) - set(VALID_PATIENT_IDS)
        if invalid_pids:
            results['valid'] = False
            results['issues'].append(f"Invalid patient IDs: {invalid_pids}")
    
    # Check target variable
    if 'target' in df.columns:
        unique_targets = df['target'].unique()
        if not set(unique_targets).issubset({0, 1}):
            results['valid'] = False
            results['issues'].append(f"Target should be binary (0,1), found: {unique_targets}")
    
    # Check vital sign ranges (basic sanity checks)
    if 'heart_rate' in df.columns:
        hr_range = (df['heart_rate'].min(), df['heart_rate'].max())
        if hr_range[0] < 30 or hr_range[1] > 200:
            results['issues'].append(f"Heart rate out of realistic range: {hr_range}")
    
    if 'temperature' in df.columns:
        temp_range = (df['temperature'].min(), df['temperature'].max())
        if temp_range[0] < 30 or temp_range[1] > 45:
            results['issues'].append(f"Temperature out of realistic range: {temp_range}")
    
    # Calculate statistics
    results['stats'].update({
        'n_measurements': len(df),
        'n_patients': df.index.str.split('_').str[0].nunique(),
        'sepsis_prevalence': (df['target'] == 1).mean() if 'target' in df.columns else 0,
        'avg_icu_stay_hours': df.groupby(df.index.str.split('_').str[0]).size().mean(),
        'patients_with_sepsis': df[df['target']==1].index.str.split('_').str[0].nunique() if 'target' in df.columns else 0,
        'has_missing_values': df.isnull().any().any(),
        'missing_percentage': (df.isnull().sum().sum() / df.size * 100) if df.isnull().any().any() else 0
    })
    
    return results

def main():
    """Main function to generate sepsis test datasets."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic sepsis detection data for ICU patients"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./sepsis_data/',
        help='Output directory for generated datasets'
    )
    parser.add_argument(
        '--n_patients', 
        type=int, 
        default=None,
        help='Number of ICU patients (overrides scenario configs)'
    )
    parser.add_argument(
        '--random_seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--single_dataset', 
        type=int, 
        default=None,
        help='Generate single dataset with specified index'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SepsisDataGenerator(random_state=args.random_seed)
    
    print("🏥 Sepsis Detection Data Generator for ICU Patients")
    print("=" * 55)
    
    if args.single_dataset:
        # Generate single dataset
        n_patients = args.n_patients or 50
        df = generator.generate_dataset(
            n_patients=n_patients,
            dataset_idx=args.single_dataset
        )
        
        # Save dataset
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save in multiple formats
        pickle_file = output_path / f"filtered_df_{args.single_dataset}GBC.pkl"
        df.to_pickle(pickle_file)
        print(f"✅ Saved: {pickle_file}")
        
        csv_file = output_path / f"sepsis_dataset_{args.single_dataset}.csv"
        df.to_csv(csv_file)
        print(f"📊 Saved CSV: {csv_file}")
        
        # Validate
        validation = validate_sepsis_data(df)
        print(f"\n📊 Validation: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")
        if validation['issues']:
            for issue in validation['issues']:
                print(f"  ⚠️  {issue}")
        
        print(f"\n📈 Clinical Statistics:")
        for key, value in validation['stats'].items():
            if isinstance(value, float):
                if 'percentage' in key or 'prevalence' in key:
                    print(f"  {key}: {value:.1%}")
                else:
                    print(f"  {key}: {value:.1f}")
            else:
                print(f"  {key}: {value}")
    
    else:
        # Generate multiple scenarios
        scenarios = create_sepsis_scenarios()
        
        # Override patient count if specified
        if args.n_patients:
            for scenario in scenarios:
                scenario['n_patients'] = args.n_patients
        
        print(f"Generating {len(scenarios)} sepsis detection scenarios...")
        
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for scenario in scenarios:
            df = generator.generate_dataset(**scenario)
            
            idx = scenario['dataset_idx']
            
            # Save pickle files
            pickle_file1 = output_path / f"filtered_df_{idx}.pkl"
            df.to_pickle(pickle_file1)
            
            pickle_file2 = output_path / f"filtered_df_{idx}GBC.pkl"
            df.to_pickle(pickle_file2)
            
            # Save CSV for inspection
            csv_file = output_path / f"sepsis_dataset_{idx}.csv"
            df.to_csv(csv_file)
            
            print(f"✅ Generated scenario {idx}: {len(df)} measurements from {scenario['n_patients']} patients")
        
        print(f"\n📊 Sepsis Dataset Summary:")
        print(f"{'Index':<8} {'Patients':<10} {'Measurements':<15} {'Sepsis %':<10} {'Avg Stay (h)':<12}")
        print("-" * 65)
        
        for scenario in scenarios:
            # Quick stats from saved file
            pickle_file = output_path / f"filtered_df_{scenario['dataset_idx']}GBC.pkl"
            df_temp = pd.read_pickle(pickle_file)
            sepsis_pct = (df_temp['target'] == 1).mean()
            avg_stay = df_temp.groupby(df_temp.index.str.split('_').str[0]).size().mean()
            
            print(f"{scenario['dataset_idx']:<8} {scenario['n_patients']:<10} "
                  f"{len(df_temp):<15} {sepsis_pct:.1%}{'':>6} {avg_stay:.1f}")
    
    print(f"\n🎯 Next Steps for Sepsis Detection Pipeline:")
    print(f"1. Update config.yaml:")
    print(f"   paths:")
    print(f"     data_dir: {Path(args.output_dir).absolute()}")
    print(f"   file_indices: [10, 20, 30, 40, 50]")
    print(f"2. Use sepsis-optimized models: XGBoost, Random Forest, Logistic Regression")
    print(f"3. Run pipeline: python main.py")
    print(f"4. Monitor for data leakage: Compare LOPOCV vs 10-Fold CV results")
    print(f"5. Clinical interpretation: Focus on early detection metrics (sensitivity)")

if __name__ == "__main__":
    main()