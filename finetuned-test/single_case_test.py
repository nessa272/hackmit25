"""
Modal-based test script to test a specific NSTEMI medical case on both
base GPT-OSS-20B and fine-tuned model, optimized for speed (no FlashAttention).
"""

import modal

# Same image as the fine-tuning script
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.38.0",   # newer release
        "peft>=0.4.0",
        "pandas",
        "numpy",
        "huggingface_hub>=0.16.0",
    ])
    .apt_install(["git"])
)

app = modal.App("single-case-test", image=image)

# Use the same volumes as the training script
cache_volume = modal.Volume.from_name("gpt-oss-cache", create_if_missing=True)

@app.function(
    gpu="H100",
    volumes={"/cache": cache_volume},
    timeout=1800,
    memory=16384,
)
def test_nstemi_case():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import os
    import re

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    print("🏥 NSTEMI Medical Case Analysis Test")
    print("=" * 60)

    # Prompt
    prompt = """Based on the following patient information, predict the length of hospital stay in days and provide detailed medical reasoning:

Patient Information:
- Age: 67 year old Male
- Race: Unknown, Ethnicity: Unknown  
- Admission Type: Emergency admission
- Diagnosis: NSTEMI (I21.4), status post PCI with drug-eluting stent to mid-LAD
- Severity of Illness: Moderate severity
- Risk of Mortality: Moderate risk
- Expected Disposition: Home or self care

Please provide:
1. Your predicted length of stay in days
2. Detailed medical reasoning for your prediction
3. Key factors that influenced your decision

Predicted length of stay: """

    messages = [
        {"role": "system", "content": "You are a medical AI assistant specializing in hospital length of stay predictions. Provide evidence-based predictions with clear reasoning."},
        {"role": "user", "content": prompt}
    ]

    # Tokenizer
    print("📊 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def render_chat(messages, add_generation_prompt=False):
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt
                )
            except Exception:
                pass
        text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return text + ("\nASSISTANT:" if add_generation_prompt else "")

    def extract_length_of_stay(text: str):
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

    def _finish_model_setup(model):
        model.config.use_cache = True
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.use_cache = True
        model.eval()
        return model

    def load_base_model():
        src = "openai/gpt-oss-20b"
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            src,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="/cache",
            low_cpu_mem_usage=True,
        )
        return _finish_model_setup(model)

    def load_ft_model_from_adapter(base_path_or_repo):
        print("Loading fine-tuned model (base + adapter)...")
        base = AutoModelForCausalLM.from_pretrained(
            base_path_or_repo,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="/cache",
            low_cpu_mem_usage=True,
        )
        base = _finish_model_setup(base)
        ft = PeftModel.from_pretrained(
            base,
            "nessa272/gpt-oss-20b-finetune-mini",
            torch_dtype=torch.bfloat16
        )
        ft = ft.merge_and_unload()
        return _finish_model_setup(ft)

    def generate_response(model, tokenizer, messages, model_name):
        formatted_prompt = render_chat(messages, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print(f"\n🤖 {model_name} Prediction:")
        print("-" * 40)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask', None),
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        predicted_days = extract_length_of_stay(generated_text)

        print(f"Generated Response: {generated_text}")
        print(f"Extracted Prediction: {predicted_days} days")
        print("-" * 40)
        return generated_text, predicted_days

    cached_base_path = "/cache/openai_gpt-oss-20b"
    cached_ft_path = "/cache/gpt-oss-20b-finetune-mini"

    print("\n🧪 Running length of stay predictions...")

    # --- Run base model first ---
    base_model = load_base_model()
    base_response, base_prediction = generate_response(base_model, tokenizer, messages, "Base GPT-OSS-20B")
    del base_model
    torch.cuda.empty_cache()

    # --- Run fine-tuned model ---
    if os.path.exists(cached_ft_path) and os.listdir(cached_ft_path):
        print("✅ Using cached fine-tuned merged model")
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            cached_ft_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        finetuned_model = _finish_model_setup(finetuned_model)
    else:
        finetuned_model = load_ft_model_from_adapter("openai/gpt-oss-20b")
        finetuned_model.save_pretrained(cached_ft_path)
        print("✅ Fine-tuned model cached successfully")

    ft_response, ft_prediction = generate_response(finetuned_model, tokenizer, messages, "Fine-tuned GPT-OSS-20B")

    # --- Compare ---
    actual_los = 6
    base_error = abs(base_prediction - actual_los) if base_prediction is not None else None
    ft_error = abs(ft_prediction - actual_los) if ft_prediction is not None else None

    print("\n" + "="*60)
    print("📊 LENGTH OF STAY PREDICTION COMPARISON")
    print("="*60)
    print(f"Actual LOS: {actual_los} days\n")

    print("📈 PREDICTION RESULTS:")
    print(f"  Base Model: {base_prediction} days (Error: {base_error})" if base_prediction else "  Base Model: Failed to extract")
    print(f"  Fine-tuned: {ft_prediction} days (Error: {ft_error})" if ft_prediction else "  Fine-tuned: Failed to extract")

    return {
        "base_response": base_response,
        "finetuned_response": ft_response,
        "base_prediction": base_prediction,
        "ft_prediction": ft_prediction,
        "base_error": base_error,
        "ft_error": ft_error,
        "actual_los": actual_los
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        with app.run():
            result = test_nstemi_case.remote()
            print("Test completed successfully!")
