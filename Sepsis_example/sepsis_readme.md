# Sepsis Detection in ICU Patients: Subject-Aware Validation Example

This example demonstrates how to apply the Subject-Aware Model Validation Pipeline to **sepsis detection in ICU patients** using repeated measures data. This is a critical clinical application where proper validation methodology can mean the difference between a helpful tool and a dangerous one.

## 🏥 Clinical Context

### What is Sepsis?
**Sepsis** is a life-threatening condition that arises when the body's response to infection causes injury to its own tissues and organs. In ICU settings:

- **Prevalence**: 10-20% of ICU patients develop sepsis
- **Mortality**: 15-30% mortality rate, higher with septic shock
- **Time-Critical**: Early detection (within 6 hours) reduces mortality by 7.6%
- **Cost**: $24 billion annually in the US healthcare system

### Why Repeated Measures?
ICU patients have continuous monitoring generating:
- **Hourly measurements**: Vital signs, lab values, clinical assessments
- **Multiple time points**: 12-168 hours of ICU stay per patient
- **Temporal patterns**: Sepsis onset and progression over time
- **Patient-specific baselines**: Individual normal ranges vary significantly

### The Data Leakage Problem
Traditional cross-validation in sepsis detection can lead to:
- **Inflated performance**: Models learn patient-specific patterns
- **Poor generalization**: High validation scores don't translate to new patients
- **Clinical risk**: Overconfident models may miss sepsis in real deployment

## 🎯 Why Subject-Aware Validation Matters

### The Challenge
```
Patient A: [Hour 1] [Hour 2] [Hour 3] ... [Hour 24] [Hour 25 - SEPSIS]
Patient B: [Hour 1] [Hour 2] [Hour 3] ... [Hour 36] [Hour 37 - SEPSIS]
```

**Standard 10-Fold CV**: May put Patient A's Hour 24 in training and Hour 25 in test
- ❌ Model learns Patient A's specific patterns
- ❌ Unrealistic: knows patient's baseline before predicting sepsis

**Subject-Aware LOPOCV**: Trains on Patients B, C, D... Tests on all of Patient A
- ✅ Model must generalize to completely new patients  
- ✅ Realistic: no prior knowledge of test patient's patterns

## 🚀 Quick Start

### Option 1: Complete Automated Example
```bash
# Run the complete example (recommended)
python run_sepsis_example.py --n_patients 50 --full_analysis

# Or with custom settings
python run_sepsis_example.py --n_patients 100 --output_dir ./my_sepsis_study/
```

### Option 2: Step-by-Step Execution
```bash
# 1. Generate synthetic sepsis data
python generate_sepsis_data.py --n_patients 50 --output_dir ./sepsis_data/

# 2. Start MLflow server
mlflow ui --port 5000 &

# 3. Run validation pipeline
python main.py --config config_sepsis.yaml

# 4. View results at http://localhost:5000
```

### Option 3: Jupyter Notebook
```bash
# Interactive analysis
jupyter notebook Sepsis_Detection_Example.ipynb
```

## 📊 Generated Synthetic Data

### Patient Characteristics
The synthetic data generator creates realistic ICU patients with:

- **Demographics**: Age (18-95), gender, comorbidities
- **Severity scores**: APACHE II (0-40), Charlson Index (0-10)
- **ICU stay**: 12-168 hours with hourly measurements
- **Sepsis onset**: 10-20% prevalence, typically 6-48 hours into stay

### Clinical Features (13 total)
```python
Vital Signs:
├── heart_rate              # 60-180 bpm
├── systolic_bp             # 60-200 mmHg  
├── diastolic_bp            # 40-120 mmHg
├── mean_arterial_pressure  # Calculated: (SBP + 2*DBP)/3
├── respiratory_rate        # 8-45 breaths/min
├── temperature             # 32-42°C
├── oxygen_saturation       # 70-100%
└── shock_index            # HR/SBP ratio

Laboratory Values:
├── white_blood_cells       # 0.5-50 x10³/µL
├── lactate                # 0.3-15 mmol/L
├── procalcitonin          # 0-50 ng/mL
├── c_reactive_protein     # 0-300 mg/L
└── platelets              # 10-600 x10³/µL
```

### Realistic Sepsis Patterns
- **Early signs**: Tachycardia, fever, elevated WBC
- **Progression**: Hypotension, organ dysfunction markers
- **Severe sepsis**: Shock index elevation, lactate increase
- **Patient variation**: Individual baseline patterns that could cause leakage

## 🔍 Expected Results

### Performance Comparison
```
Validation Strategy    |  Accuracy  |  F1-Score  |  Interpretation
--------------------- | ---------- | ---------- | ---------------
10-Fold CV            |   0.89     |   0.85     |  Overoptimistic
Group 3-Fold CV       |   0.82     |   0.78     |  More realistic  
LOPOCV               |   0.78     |   0.73     |  True patient-level performance
```

### Data Leakage Assessment
- **High leakage** (>10% gap): Standard CV learning patient patterns
- **Moderate leakage** (5-10% gap): Some patient-specific learning
- **Low leakage** (<5% gap): Good generalization

### Feature Importance (Expected)
1. **Lactate** - Tissue hypoxia marker, elevated in septic shock
2. **Procalcitonin** - Bacterial infection biomarker  
3. **C-Reactive Protein** - Inflammatory response indicator
4. **White Blood Cells** - Immune system response
5. **Temperature** - Fever or hypothermia in sepsis

## 🏥 Clinical Interpretation

### Model Performance Metrics
- **Sensitivity (Recall)**: Most critical - must catch sepsis cases
- **Specificity**: Important - reduce false alarms and alert fatigue  
- **PPV/NPV**: Depends on sepsis prevalence in your ICU
- **AUC-ROC**: Overall discrimination ability

### Validation Strategy Recommendations

| Use Case | Recommended CV | Rationale |
|----------|---------------|-----------|
| **Research Publication** | LOPOCV + 10-Fold | Report both; discuss leakage |
| **Clinical Deployment** | LOPOCV | Must generalize to new patients |
| **Algorithm Development** | Group 3-Fold | Balance of realism and efficiency |
| **Regulatory Submission** | LOPOCV + Temporal | Most conservative validation |

### Clinical Deployment Considerations
- **Real-time feasibility**: All models <1ms inference time
- **Integration**: Hourly automated predictions from EMR data
- **Alert fatigue**: Balance sensitivity with specificity
- **Human factors**: Interpretable features for clinical acceptance

## ⚠️ Important Limitations

### Synthetic Data Limitations
- **Simplified pathophysiology**: Real sepsis is more complex
- **Missing confounders**: Medications, interventions, comorbidities
- **Idealized patterns**: Real ICU data has more noise and artifacts
- **Population bias**: May not represent your specific patient population

### Validation Considerations
- **Temporal drift**: Patient populations change over time
- **Site specificity**: Each ICU has different patient characteristics
- **Definition variability**: Sepsis criteria vary between institutions
- **Missing data**: Real ICU data has more missingness patterns

## 🎯 Next Steps for Real Implementation

### 1. Data Preparation
```python
# Adapt your real ICU data to the expected format
real_icu_df = pd.DataFrame({
    # Use your actual feature names and patient IDs
    'patient_id_hour': ['ICU001_1', 'ICU001_2', ...],
    'heart_rate': [...],
    'temperature': [...],
    # ... other clinical features
    'target': [...]  # 0=no sepsis, 1=sepsis
})
```

### 2. Validation Protocol
```python
# Recommended validation approach
protocols = [
    "LOPOCV",           # Primary validation
    "Temporal split",   # Train: 2020-2022, Test: 2023
    "Site validation",  # Train: Site A, Test: Site B  
    "Prospective"       # Deploy and monitor performance
]
```

### 3. Clinical Integration
- **EMR integration**: Automated feature extraction
- **Decision support**: Integrate with clinical workflows
- **Alert system**: Configurable thresholds and notifications
- **Performance monitoring**: Track model drift and calibration

### 4. Regulatory Considerations
- **FDA guidance**: Software as Medical Device (SaMD) pathway
- **Clinical validation**: Prospective clinical trial
- **Documentation**: Comprehensive validation and risk analysis
- **Quality management**: ISO 13485 compliance for medical devices

## 📚 References and Further Reading

### Key Publications
1. **Sepsis-3 Definitions**: Singer M, et al. JAMA. 2016
2. **SOFA Score**: Vincent JL, et al. Intensive Care Med. 1996  
3. **ML in Sepsis**: Fleuren LM, et al. Intensive Care Med. 2020
4. **Subject-Aware Validation**: Your paper reference here

### Clinical Guidelines
- **Surviving Sepsis Campaign**: International Guidelines 2021
- **NICE Guidelines**: Sepsis Recognition and Early Management
- **CMS SEP-1**: Core Measure for Sepsis Management

### Technical Resources
- **MIMIC-III Database**: Real ICU data for research
- **PhysioNet**: Physiological signal databases
- **FDA AI/ML Guidance**: Medical device software guidance

## 🤝 Contributing

### Clinical Input Needed
- Sepsis definition validation
- Feature engineering suggestions  
- Clinical workflow integration ideas
- Real-world deployment experiences

### Technical Improvements
- Additional clinical features
- More sophisticated sepsis progression models
- Integration with clinical decision support systems
- Performance optimization for real-time deployment

## 📞 Support

For questions about this sepsis detection example:

- **Clinical questions**: [Your clinical contact]
- **Technical issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Implementation help**: [Your support email]
- **Collaboration opportunities**: [Research contact]

---

**⚠️ Important Disclaimer**: This is a research tool using synthetic data. Not intended for clinical use without proper validation on real patient data and regulatory approval.