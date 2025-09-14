"""
Modal-based test script to compare hospital stay length predictions between 
base GPT-OSS-20B and fine-tuned model on random discharge records.
"""

import modal

# Same image as the fine-tuning script
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "pandas",
        "numpy",
        "huggingface_hub>=0.16.0",
    ])
    .apt_install(["git"])
)

app = modal.App("model-comparison-test", image=image)

# Use the same volumes as the training script
dataset_volume = modal.Volume.from_name("gpt-oss-datasets", create_if_missing=True)
cache_volume = modal.Volume.from_name("gpt-oss-cache", create_if_missing=True)

@app.function(
    gpu="H100",  # Fastest GPU for 20B model inference
    volumes={"/datasets": dataset_volume, "/cache": cache_volume},
    timeout=3600,
    memory=16384,
)
def compare_models(n_samples: int = 3):
    import pandas as pd
    import numpy as np
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    print("🏥 Hospital Stay Length Prediction Comparison")
    print("=" * 60)
    
    # Load random samples
    print(f"Loading {n_samples} random samples from discharge-data.csv...")
    df = pd.read_csv("/datasets/discharge-data.csv")
    
    # Basic cleaning
    df = df.dropna(subset=['CCS Diagnosis Description'])
    df = df[df['CCS Diagnosis Description'].str.strip() != ""]
    
    # Sample random rows
    samples = df.sample(n=n_samples, random_state=42)
    print(f"Selected {len(samples)} random samples")
    
    def create_medical_summary(row):
        """Create medical summary from discharge record."""
        return {
            'Age': row.get('Age', 'Unknown'),
            'Gender': row.get('Gender', 'Unknown'), 
            'Admission Type': row.get('Admission Type', 'Unknown'),
            'Diagnosis': row.get('CCS Diagnosis Description', 'Unknown'),
            'Severity of Illness': row.get('Severity of Illness', 'Unknown')
        }
    
    def extract_length_of_stay(text):
        """Extract predicted length of stay from model output."""
        patterns = [
            r'(\d+)\s*days?(?:\s+stay)?',
            r'(\d+)-day(?:\s+stay)?',
            r'stay(?:\s+of)?\s+(\d+)\s*days?',
            r'length(?:\s+of)?\s+stay[:\s]*(\d+)\s*days?',
            r'hospitalization[:\s]*(\d+)\s*days?',
            r'(\d+)\s*day(?:\s+length)?(?:\s+of)?(?:\s+stay)?'
        ]
        
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    return int(matches[0])
                except ValueError:
                    continue
        
        return None
    
    def predict_length_of_stay(model, tokenizer, patient_info, model_name):
        """Generate prediction for length of stay"""
        prompt = f"""Based on the following patient information, predict the length of hospital stay in days:

Patient Information:
- Age: {patient_info.get('Age', 'Unknown')}
- Gender: {patient_info.get('Gender', 'Unknown')}
- Admission Type: {patient_info.get('Admission Type', 'Unknown')}
- Diagnosis: {patient_info.get('Diagnosis', 'Unknown')}
- Severity of Illness: {patient_info.get('Severity of Illness', 'Unknown')}

Predicted length of stay: """

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract predicted days
        predicted_days = extract_length_of_stay(generated_text)
        
        print(f"\n{model_name} Response:")
        print(f"Generated: {generated_text[:150]}...")
        print(f"Extracted prediction: {predicted_days} days")
        
        return predicted_days, generated_text
    
    print("\n📊 Loading models...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with caching
    print("Loading base model...")
    import os
    cached_base_path = "/cache/openai_gpt-oss-20b"
    cached_ft_path = "/cache/gpt-oss-20b-finetune-mini"
    
    if os.path.exists(cached_base_path):
        print("✅ Using cached base model")
        base_model = AutoModelForCausalLM.from_pretrained(
            cached_base_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        print("📥 Downloading and caching base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="/cache",
            low_cpu_mem_usage=True
        )
        # Save to cache for future use
        base_model.save_pretrained(cached_base_path)
        print("✅ Base model cached successfully")
    
    # Load fine-tuned model with caching
    print("Loading fine-tuned model...")
    if os.path.exists(cached_ft_path):
        print("✅ Using cached fine-tuned model")
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            cached_ft_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        print("📥 Loading fine-tuned model from base + adapter...")
        # Load base model first, then apply LoRA adapter
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            cached_base_path if os.path.exists(cached_base_path) else "openai/gpt-oss-20b",
            torch_dtype=torch.bfloat16,
            device_map="auto", 
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        # Load and apply LoRA adapter
        from peft import PeftModel
        finetuned_model = PeftModel.from_pretrained(
            finetuned_model,
            "nessa272/gpt-oss-20b-finetune-mini",
            torch_dtype=torch.bfloat16
        )
        # Save merged model to cache
        finetuned_model = finetuned_model.merge_and_unload()
        finetuned_model.save_pretrained(cached_ft_path)
        print("✅ Fine-tuned model cached successfully")
    
    print("\n🧪 Running predictions...")
    
    results = []
    
    for i, (idx, row) in enumerate(samples.iterrows()):
        print(f"\n" + "="*60)
        print(f"TEST CASE {i+1}/{n_samples}")
        print("="*60)
        
        # Get actual length of stay
        actual_los = row.get('Length of Stay', 'Unknown')
        try:
            actual_los = float(actual_los) if actual_los != 'Unknown' else None
        except:
            actual_los = None
        
        # Create medical summary
        medical_summary = create_medical_summary(row)
        summary_str = f"Age: {medical_summary['Age']}, Gender: {medical_summary['Gender']}, Admission: {medical_summary['Admission Type']}, Diagnosis: {medical_summary['Diagnosis']}"
        print(f"Medical Summary: {summary_str[:200]}...")
        print(f"Actual Length of Stay: {actual_los} days")
        
        # Base model prediction
        base_pred, base_response = predict_length_of_stay(
            base_model, tokenizer, medical_summary, "Base Model"
        )
        
        # Fine-tuned model prediction
        ft_pred, ft_response = predict_length_of_stay(
            finetuned_model, tokenizer, medical_summary, "Fine-tuned Model"
        )
        
        # Calculate errors if actual is available
        base_error = abs(base_pred - actual_los) if base_pred and actual_los else None
        ft_error = abs(ft_pred - actual_los) if ft_pred and actual_los else None
        
        result = {
            'case': i+1,
            'actual_los': actual_los,
            'base_prediction': base_pred,
            'finetuned_prediction': ft_pred,
            'base_error': base_error,
            'finetuned_error': ft_error,
        }
        results.append(result)
        
        print(f"\n📊 COMPARISON:")
        print(f"Actual: {actual_los} days")
        print(f"Base Model: {base_pred} days (Error: {base_error})")
        print(f"Fine-tuned: {ft_pred} days (Error: {ft_error})")
        
        if base_error and ft_error:
            if ft_error < base_error:
                print("✅ Fine-tuned model is more accurate!")
            elif base_error < ft_error:
                print("❌ Base model is more accurate")
            else:
                print("🤝 Both models equally accurate")
    
    # Summary statistics
    print(f"\n" + "="*60)
    print("📈 SUMMARY STATISTICS")
    print("="*60)
    
    valid_results = [r for r in results if r['base_error'] is not None and r['finetuned_error'] is not None]
    
    if valid_results:
        base_errors = [r['base_error'] for r in valid_results]
        ft_errors = [r['finetuned_error'] for r in valid_results]
        
        print(f"Cases with valid comparisons: {len(valid_results)}")
        print(f"Base Model - Mean Error: {np.mean(base_errors):.2f} days")
        print(f"Fine-tuned Model - Mean Error: {np.mean(ft_errors):.2f} days")
        print(f"Base Model - Median Error: {np.median(base_errors):.2f} days")
        print(f"Fine-tuned Model - Median Error: {np.median(ft_errors):.2f} days")
        
        ft_better = sum(1 for r in valid_results if r['finetuned_error'] < r['base_error'])
        base_better = sum(1 for r in valid_results if r['base_error'] < r['finetuned_error'])
        tied = len(valid_results) - ft_better - base_better
        
        print(f"\nWin/Loss Record:")
        print(f"Fine-tuned Better: {ft_better}/{len(valid_results)}")
        print(f"Base Better: {base_better}/{len(valid_results)}")
        print(f"Tied: {tied}/{len(valid_results)}")
        
        if np.mean(base_errors) > 0:
            improvement = ((np.mean(base_errors) - np.mean(ft_errors)) / np.mean(base_errors)) * 100
            print(f"\nOverall Improvement: {improvement:.1f}%")
    
    return results

@app.local_entrypoint()
def main(n_samples: int = 10):
    """Run the model comparison test."""
    print(f"Starting model comparison with {n_samples} samples...")
    results = compare_models.remote(n_samples)
    print("Comparison completed!")
    return results

if __name__ == "__main__":
    main()
