"""
Modal web API for healthcare length of stay predictions using fine-tuned GPT-OSS-20B.
Takes patient information and returns conservative LOS prediction with medical reasoning.
"""

import modal
import re
from typing import Dict, Any

# Modal image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "peft>=0.4.0",
        "fastapi",
        "pydantic",
        "huggingface_hub>=0.16.0",
    ])
    .apt_install(["git"])
)

app = modal.App("patient-los-api", image=image)
cache_volume = modal.Volume.from_name("gpt-oss-cache", create_if_missing=True)

# Define data structures (will import BaseModel inside functions)
# PatientInfo fields: age, gender, diagnosis, severity, mortality_risk, admission_type, expected_disposition, additional_info
# LOSPrediction fields: predicted_los_days, reasoning, confidence, patient_summary

@app.function(
    gpu="H100:1",
    volumes={"/cache": cache_volume},
    timeout=300,
    container_idle_timeout=120,
)
@modal.asgi_app(label="predict-los")
def predict_length_of_stay():
    from fastapi import FastAPI
    from pydantic import BaseModel
    from typing import Optional
    
    class PatientInfo(BaseModel):
        age: int
        gender: str
        diagnosis: str
        severity: str = ""
        mortality_risk: str = ""
        admission_type: str = "emergency"
        expected_disposition: str = "home"
        additional_info: str = ""
    
    web_app = FastAPI()
    
    @web_app.post("/predict")
    def predict_los_endpoint(patient_data: PatientInfo) -> Dict[str, Any]:
        """
        API endpoint to predict length of stay for a patient using the fine-tuned model.
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import signal
        import time
        
        import os
        
        # Use the same cached path as single_case_test.py
        cached_ft_path = "/cache/gpt-oss-20b-finetune-mini"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "openai/gpt-oss-20b",
            cache_dir="/cache",
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Check for cached merged model first
        if os.path.exists(cached_ft_path) and os.listdir(cached_ft_path):
            print("🚀 Using cached fine-tuned merged model")
            model = AutoModelForCausalLM.from_pretrained(
                cached_ft_path,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            model.eval()
        else:
            print("⚠️  No cached merged model found, loading base + adapter...")
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                "openai/gpt-oss-20b",
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir="/cache",
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            
            # Load LoRA adapter and merge
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                base_model,
                "nessa272/gpt-oss-20b-finetune-mini",
                dtype=torch.bfloat16
            )
            model = model.merge_and_unload()
            model.eval()
            
            # Save merged model to cache for future use
            print("💾 Saving merged model to cache...")
            model.save_pretrained(cached_ft_path)
            print("✅ Fine-tuned model cached successfully")
        
        # Extract patient information from Pydantic model
        age = patient_data.age
        gender = patient_data.gender
        diagnosis = patient_data.diagnosis
        severity = patient_data.severity
        mortality_risk = patient_data.mortality_risk
        admission_type = patient_data.admission_type
        expected_disposition = patient_data.expected_disposition
        additional_info = patient_data.additional_info
        
        # Format patient information into prompt
        patient_summary = f"{age} year old {gender}"
        
        # Create the medical prompt
        messages = [
            {
                "role": "user",
                "content": f"""You are a healthcare AI assistant. Predict the length of stay (LOS) for this patient admission.

IMPORTANT: Be conservative in your predictions and lean toward longer length of stay estimates. It is better to overestimate than underestimate hospital stay duration for patient safety and resource planning.

Patient: {patient_summary}
Diagnosis: {diagnosis}
Severity: {severity}
Mortality Risk: {mortality_risk}
Admission Type: {admission_type}
Expected Disposition: {expected_disposition}
{f"Additional Info: {additional_info}" if additional_info else ""}

Please provide:
1. A conservative length of stay prediction in days (err on the side of longer stays)
2. Detailed medical reasoning for your prediction

When making your prediction, consider factors that may extend the stay:
- Patient age and comorbidities
- Severity of condition and potential complications
- Recovery time and monitoring needs
- Discharge planning requirements

Format your response as:
**Predicted length of stay:** X days

**Medical reasoning:**
[Provide a detailed paragraph explaining your reasoning, considering the patient's age, diagnosis, severity, and other factors that influence length of stay. Emphasize why a conservative approach is warranted.]"""
            }
        ]
        
        # Generate response with timeout handling
        try:
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Set a timeout for model inference (4 minutes)
            inference_timeout = 240
            start_time = time.time()
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            
            inference_time = time.time() - start_time
            print(f"⏱️ Model inference completed in {inference_time:.2f} seconds")
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(input_text):].strip()
            
            # Extract LOS prediction
            los_prediction = extract_los_prediction(generated_text)
            
            # Extract reasoning
            reasoning = extract_reasoning(generated_text)
            
            # Determine confidence based on factors
            confidence = determine_confidence(patient_data.dict(), los_prediction)
            
            return {
                "predicted_los_days": los_prediction,
                "reasoning": reasoning,
                "confidence": confidence,
                "patient_summary": patient_summary
            }
            
        except torch.cuda.OutOfMemoryError:
            print("❌ CUDA out of memory during inference")
            return {
                "error": "GPU_OUT_OF_MEMORY",
                "message": "Model inference failed due to insufficient GPU memory. Please try again later.",
                "predicted_los_days": 5,  # Conservative fallback
                "reasoning": "Unable to generate detailed reasoning due to GPU memory constraints. This is a conservative estimate based on typical NSTEMI cases.",
                "confidence": "Low",
                "patient_summary": patient_summary
            }
            
        except Exception as e:
            print(f"❌ Model inference error: {str(e)}")
            # Check if it's a timeout-related error
            if "timeout" in str(e).lower() or "time" in str(e).lower():
                return {
                    "error": "INFERENCE_TIMEOUT",
                    "message": "Model inference timed out. This may occur when the model falls back to CPU processing.",
                    "predicted_los_days": 5,  # Conservative fallback
                    "reasoning": "Unable to generate detailed reasoning due to inference timeout. This is a conservative estimate based on typical cases for this patient profile.",
                    "confidence": "Low",
                    "patient_summary": patient_summary
                }
            else:
                return {
                    "error": "INFERENCE_ERROR",
                    "message": f"Model inference failed: {str(e)}",
                    "predicted_los_days": 5,  # Conservative fallback
                    "reasoning": "Unable to generate reasoning due to model inference error. This is a conservative estimate.",
                    "confidence": "Low",
                    "patient_summary": patient_summary
                }
    
    return web_app

def extract_los_prediction(text: str) -> int:
    """Extract length of stay prediction from generated text."""
    # Look for patterns like "X days" or "Predicted length of stay: X days"
    patterns = [
        r"(?:predicted length of stay|prediction).*?(\d+)\s*days?",
        r"(\d+)\s*days?",
        r"stay.*?(\d+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    
    # Default conservative prediction if no clear number found
    return 5

def extract_reasoning(text: str) -> str:
    """Extract medical reasoning from generated text."""
    # Look for reasoning section
    reasoning_patterns = [
        r"medical reasoning[:\s]*(.+?)(?:\n\n|\Z)",
        r"reasoning[:\s]*(.+?)(?:\n\n|\Z)",
        r"\*\*medical reasoning\*\*[:\s]*(.+?)(?:\n\n|\Z)",
    ]
    
    for pattern in reasoning_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            if len(reasoning) > 50:  # Ensure we have substantial reasoning
                return reasoning
    
    # If no specific reasoning section found, return the whole response
    # but clean it up
    cleaned_text = re.sub(r'\*\*[^*]+\*\*', '', text)  # Remove markdown headers
    cleaned_text = cleaned_text.strip()
    
    if len(cleaned_text) > 100:
        return cleaned_text
    
    return "Based on the patient's clinical presentation, age, diagnosis, and severity level, this length of stay prediction considers standard care protocols and expected recovery timeline."

def determine_confidence(patient_data: Dict[str, Any], los_prediction: int) -> str:
    """Determine confidence level based on patient factors."""
    confidence_score = 0
    
    # Age factor - ensure age is converted to int
    age = patient_data.get("age", 0)
    try:
        age = int(age) if age else 0
        if 18 <= age <= 80:
            confidence_score += 1
    except (ValueError, TypeError):
        pass  # Skip age factor if conversion fails
    
    # Clear diagnosis
    diagnosis = patient_data.get("diagnosis", "")
    if len(diagnosis) > 10:
        confidence_score += 1
    
    # Severity specified
    severity = patient_data.get("severity", "")
    if severity.lower() in ['mild', 'moderate', 'severe']:
        confidence_score += 1
    
    # Reasonable LOS prediction
    if 1 <= los_prediction <= 30:
        confidence_score += 1
    
    if confidence_score >= 3:
        return "High"
    elif confidence_score >= 2:
        return "Medium"
    else:
        return "Low"

@app.function()
@modal.asgi_app(label="health-check")
def health_check():
    """Health check endpoint."""
    from fastapi import FastAPI
    
    web_app = FastAPI()
    
    @web_app.get("/health")
    def health():
        return {"status": "healthy", "service": "patient-los-api"}
    
    return web_app

# Additional utility endpoints
@app.function()
@modal.asgi_app(label="models")
def get_model_info():
    from fastapi import FastAPI
    
    web_app = FastAPI()
    
    @web_app.get("/info")
    def model_info():
        return {
            "base_model": "openai/gpt-oss-20b",
            "fine_tuned_model": "nessa272/gpt-oss-20b-finetune-mini",
            "method": "LoRA fine-tuning",
            "domain": "healthcare"
        }
    
    return web_app

if __name__ == "__main__":
    # For local testing
    print("Patient LOS API - use 'modal serve patient_api.py' to run locally")
