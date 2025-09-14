"""
Script to merge LoRA adapter with base model and save the full merged model.
This avoids having to merge the adapter every time we load the model.
"""

import modal

# Same image as other scripts
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "peft>=0.4.0",
        "huggingface_hub>=0.16.0",
    ])
    .apt_install(["git"])
)

app = modal.App("save-merged-model", image=image)
cache_volume = modal.Volume.from_name("gpt-oss-cache", create_if_missing=True)

@app.function(
    gpu="H100:1",
    volumes={"/cache": cache_volume},
    timeout=600,
)
def merge_and_save_model():
    """
    Load base model, merge with LoRA adapter, and save the full merged model.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    print("🔄 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/cache",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    
    print("🔄 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        "nessa272/gpt-oss-20b-finetune-mini",
        torch_dtype=torch.bfloat16
    )
    
    print("🔄 Merging LoRA adapter with base model...")
    merged_model = model.merge_and_unload()
    
    # Save the merged model to cache volume
    merged_model_path = "/cache/gpt-oss-20b-merged"
    print(f"💾 Saving merged model to {merged_model_path}...")
    
    merged_model.save_pretrained(
        merged_model_path,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # Also save the tokenizer
    print("💾 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "openai/gpt-oss-20b",
        cache_dir="/cache",
        trust_remote_code=True
    )
    tokenizer.save_pretrained(merged_model_path)
    
    print("✅ Merged model saved successfully!")
    print(f"📍 Model location: {merged_model_path}")
    
    return {
        "status": "success",
        "merged_model_path": merged_model_path,
        "message": "Model merged and saved successfully"
    }

if __name__ == "__main__":
    print("Use 'modal run save_merged_model.py::merge_and_save_model' to merge and save the model")
