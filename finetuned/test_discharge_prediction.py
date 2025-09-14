#!/usr/bin/env python3
"""
Test discharge prediction using both base and fine-tuned models.
"""
import modal
import subprocess
import json

def get_patient_data(row_index=1):
    """Get patient data from CSV for testing."""
    import pandas as pd
    df = pd.read_csv('train-test.csv')
    patient = df.iloc[row_index]
    
    return {
        'age_group': patient['Age Group'],
        'gender': patient['Gender'],
        'diagnosis': patient['CCS Diagnosis Description'],
        'severity': patient['APR Severity of Illness Description'],
        'admission_type': patient['Type of Admission'],
        'actual_los': patient['Length of Stay'],
        'emergency_dept': patient['Emergency Department Indicator']
    }

def create_discharge_prompt(patient_data):
    """Create a structured prompt for discharge prediction."""
    return f"""Based on the following patient information, predict the length of stay in days:

Patient Profile:
- Age: {patient_data['age_group']}
- Gender: {patient_data['gender']}
- Admission: {patient_data['admission_type']}
- Diagnosis: {patient_data['diagnosis']}
- Severity: {patient_data['severity']}
- Emergency Dept: {patient_data['emergency_dept']}

Question: How many days will this patient stay in the hospital?
Answer with just the number of days followed by brief reasoning."""

def test_model_predictions():
    """Test both base and fine-tuned models on discharge prediction."""
    
    # Test with patient row 1
    patient = get_patient_data(1)
    prompt = create_discharge_prompt(patient)
    
    print("=== DISCHARGE PREDICTION TEST ===")
    print(f"Patient: {patient['age_group']} {patient['gender']}")
    print(f"Diagnosis: {patient['diagnosis']}")
    print(f"Severity: {patient['severity']}")
    print(f"Actual Length of Stay: {patient['actual_los']} days")
    print("=" * 50)
    
    # Test base model (using openai/gpt-oss-120b directly)
    print("\n🔍 Testing BASE MODEL...")
    base_cmd = [
        'modal', 'run', 'gpt_oss_120b_finetune.py::serve_gpt_oss_120b',
        '--base-model-name', 'openai/gpt-oss-120b',
        '--model-path', '/tmp/dummy',  # Won't be used for base model
        '--prompt', prompt,
        '--max-new-tokens', '150'
    ]
    
    try:
        result = subprocess.run(base_cmd, capture_output=True, text=True, cwd='/Users/nessatong/Desktop/CS Projects/Hackmit25')
        base_output = result.stdout
        print("Base Model Response:")
        print(base_output)
    except Exception as e:
        print(f"Base model test failed: {e}")
    
    # Test fine-tuned model
    print("\n🎯 Testing FINE-TUNED MODEL...")
    ft_cmd = [
        'modal', 'run', 'gpt_oss_120b_finetune.py::serve_gpt_oss_120b',
        '--model-path', '/models/gpt-oss-120b-finetune',
        '--prompt', prompt,
        '--max-new-tokens', '150'
    ]
    
    try:
        result = subprocess.run(ft_cmd, capture_output=True, text=True, cwd='/Users/nessatong/Desktop/CS Projects/Hackmit25')
        ft_output = result.stdout
        print("Fine-tuned Model Response:")
        print(ft_output)
    except Exception as e:
        print(f"Fine-tuned model test failed: {e}")

if __name__ == "__main__":
    test_model_predictions()
