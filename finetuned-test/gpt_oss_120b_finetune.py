"""
GPT-OSS-20B Fine-tuning on Modal (H100)
CSV training data -> LoRA -> saved adapter.
Minimal, no W&B, safe chat fallback
"""

import modal

# --- Image with just what we use ---
# If you will NOT use 4/8-bit quant at serve time, you can remove "bitsandbytes".
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "peft>=0.4.0",
        "pandas",
        "numpy",
        "bitsandbytes>=0.39.0",   # optional; remove if you won't quantize for serving
        "huggingface_hub>=0.16.0",
    ])
    .apt_install(["git"])
)

app = modal.App("gpt-oss-20b-finetune", image=image)

# Persistent volumes
dataset_volume = modal.Volume.from_name("gpt-oss-datasets", create_if_missing=True)
model_volume   = modal.Volume.from_name("gpt-oss-20b-models", create_if_missing=True)
cache_volume   = modal.Volume.from_name("gpt-oss-cache", create_if_missing=True)

@app.function(
    gpu="H100:4",  # 4 H100 GPUs for faster training
    volumes={"/datasets": dataset_volume, "/models": model_volume, "/cache": cache_volume},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=3600,
    memory=16384,
    scaledown_window=300,
)
def finetune_gpt_oss_20b(
    csv_file_path: str = "/datasets/discharge-data.csv",
    model_name: str = "openai/gpt-oss-20b",
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 128,
    use_lora: bool = True,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    # data -> text controls
    create_medical_summaries: bool = True,
    include_diagnosis: bool = True,
    include_demographics: bool = True,
    include_costs: bool = True,
    reasoning_level: str = "medium",
    experiment_name: str = "gpt-oss-20b-finetune",
    save_steps: int = 200,
    eval_steps: int = 200,
    # HuggingFace Hub settings
    push_to_hub: bool = True,
    hf_username: str = "nessa272",
    hf_repo_name: str = "gpt-oss-20b-finetune-mini",
    hf_token: str = None,
):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import json, torch, pandas as pd
    import gc  # For garbage collection to manage memory
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    print("Loading CSV with streaming:", csv_file_path)
    # Stream CSV in chunks to avoid memory issues
    chunk_size = 10000
    chunks = []
    total_rows = 0
    
    for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
        total_rows += len(chunk)
        chunks.append(chunk)
        if total_rows >= 50000:  # Stop early if we have enough data
            break
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df)} rows from {total_rows} total | Columns: {list(df.columns)}")
    
    # Shuffle and limit to 50,000 rows
    df = df.sample(n=min(50000, len(df)), random_state=42).reset_index(drop=True)
    print(f"After shuffling and limiting: {len(df)} rows")

    # Minimal required columns for summary generation
    required_cols = ['CCS Diagnosis Description', 'Age Group', 'Gender']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Basic cleaning
    df = df.dropna(subset=['CCS Diagnosis Description'])
    df = df[df['CCS Diagnosis Description'].str.strip() != ""]
    if "Facility Name" in df.columns:
        df = df[~df['Facility Name'].str.contains('Redacted', na=False)]
    
    # Fix data type issues for PyArrow conversion
    # Convert problematic columns to strings to avoid mixed type errors
    numeric_columns = ['Length of Stay', 'Total Charges', 'Total Costs']
    for col in numeric_columns:
        if col in df.columns:
            # Convert to string first, then handle NaN values
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(['nan', 'None', ''], '0')
            # Try to convert back to numeric, fallback to string
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except:
                df[col] = df[col].astype(str)
    
    # Ensure all object columns are strings to avoid Arrow conversion issues
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).replace('nan', 'Unknown')
    
    print("After cleaning:", len(df))

    print("Loading tokenizer:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Check for cached model first
    cache_path = f"/cache/{model_name.replace('/', '_')}"
    
    if os.path.exists(cache_path):
        print(f"Loading cached model from {cache_path}")
        model = AutoModelForCausalLM.from_pretrained(
            cache_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True,
        )
    else:
        print(f"Loading model from HuggingFace: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,    # <- use bf16 explicitly
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True,
        )
        # Cache the model for future use
        print(f"Caching model to {cache_path}")
        model.save_pretrained(cache_path)
        tokenizer.save_pretrained(cache_path)
        cache_volume.commit()
        print("Model cached successfully")

    # LoRA
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Enable training mode and ensure gradients
        model.train()
        
        # Verify trainable parameters
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        
        print(f"Found {len(trainable_params)} trainable parameters")
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found! LoRA may not be applied correctly.")
        
        # Print first few trainable params for verification
        for i, name in enumerate(trainable_params[:5]):
            print(f"  {name}")
        if len(trainable_params) > 5:
            print(f"  ... and {len(trainable_params) - 5} more")

    # Enable gradient checkpointing once (no duplicate flag elsewhere)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # Chat formatting fallback
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

    def create_medical_summary(row):
        parts = []
        if include_demographics:
            age = row.get('Age Group', 'Unknown')
            gender = row.get('Gender', 'Unknown')
            race = row.get('Race', 'Unknown')
            ethnicity = row.get('Ethnicity', 'Unknown')
            parts.append(f"Patient: {age} year old {gender}, {race}, {ethnicity}")

        admission_type = row.get('Type of Admission', 'Unknown')
        los = row.get('Length of Stay', 'Unknown')
        parts.append(f"Admission: {admission_type} admission with {los} day length of stay")

        if include_diagnosis:
            dx = row.get('CCS Diagnosis Description', 'Unknown diagnosis')
            proc = row.get('CCS Procedure Description', 'NO PROC')
            severity = row.get('APR Severity of Illness Description', 'Unknown')
            mort = row.get('APR Risk of Mortality', 'Unknown')
            parts.append(f"Diagnosis: {dx}")
            if isinstance(proc, str) and proc != 'NO PROC':
                parts.append(f"Procedure: {proc}")
            parts.append(f"Severity: {severity} severity, {mort} mortality risk")

        disp = row.get('Patient Disposition', 'Unknown')
        parts.append(f"Disposition: {disp}")

        if include_costs:
            try:
                charges = float(row.get('Total Charges', 0) or 0)
                costs   = float(row.get('Total Costs', 0) or 0)
                pay     = row.get('Payment Typology 1', 'Unknown')
                parts.append(f"Billing: ${charges:,.2f} charges, ${costs:,.2f} costs, {pay} insurance")
            except Exception:
                pass

        return ". ".join(parts) + "."

    def preprocess_function(examples):
        import pandas as pd
        texts = []
        df_batch = pd.DataFrame(examples)
        for _, r in df_batch.iterrows():
            medical_text = create_medical_summary(r) if create_medical_summaries \
                           else r.get('CCS Diagnosis Description', 'Unknown medical condition')
            messages = [
                {"role": "system", "content": f"You are a medical AI assistant. Analyze with {reasoning_level} reasoning."},
                {"role": "user", "content": f"Analyze this medical case and provide insights: {medical_text}"},
                {"role": "assistant", "content": f"Medical Case Analysis: {medical_text}"},
            ]
            texts.append(render_chat(messages, add_generation_prompt=False))

        toks = tokenizer(texts, truncation=True, padding=False, max_length=max_length, return_tensors=None)
        toks["labels"] = toks["input_ids"].copy()
        return toks

    # Use streaming dataset processing to avoid memory issues
    print("Creating streaming dataset...")
    dataset = Dataset.from_pandas(df)
    
    if len(dataset) > 100:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        train_dataset, eval_dataset = dataset, None

    print("Tokenizing with streaming...")
    # Process in smaller batches to reduce memory usage
    batch_size_tokenize = 1000
    train_dataset = train_dataset.map(
        preprocess_function, 
        batched=True, 
        batch_size=batch_size_tokenize,
        remove_columns=train_dataset.column_names, 
        desc="Tokenize train"
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            preprocess_function, 
            batched=True, 
            batch_size=batch_size_tokenize,
            remove_columns=eval_dataset.column_names, 
            desc="Tokenize eval"
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    output_dir = f"/models/{experiment_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_steps=eval_steps if eval_dataset else None,
        save_steps=save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        report_to=None,
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
        fp16=False,
        bf16=True,   # H100 supports bf16 and it matches the model's dtype
        tf32=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Training…")
    trainer.train()

    print("Saving adapter/tokenizer…")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Push to HuggingFace Hub if requested
    if push_to_hub:
        try:
            from huggingface_hub import HfApi, login
            import os
            
            print(f"Pushing model to HuggingFace Hub: {hf_username}/{hf_repo_name}")
            
            # Get token from Modal secret
            if hf_token is None:
                # Try different possible environment variable names
                hf_token = (os.environ.get("HUGGINGFACE_TOKEN") or 
                           os.environ.get("HF_TOKEN") or 
                           os.environ.get("HUGGING_FACE_HUB_TOKEN"))
                
                if not hf_token:
                    # Debug: print available environment variables
                    print("Available environment variables:")
                    for key in sorted(os.environ.keys()):
                        if any(term in key.lower() for term in ['hf', 'hugging', 'token']):
                            print(f"  {key}")
                    
                    error_msg = """
HuggingFace token not found. Please check your Modal secret configuration.
The secret 'hf-token' exists but the environment variable name might be different.
"""
                    raise ValueError(error_msg)
            
            # Login with token
            login(token=hf_token)
            
            # Create repository if it doesn't exist
            api = HfApi()
            repo_id = f"{hf_username}/{hf_repo_name}"
            
            try:
                api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
                print(f"Repository {repo_id} created/verified")
            except Exception as e:
                print(f"Repository creation info: {e}")
            
            # Push the model files
            api.upload_folder(
                folder_path=output_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Fine-tuned GPT-OSS-20B on healthcare data - {experiment_name}"
            )
            
            # Create a model card
            model_card_content = f"""---
license: apache-2.0
base_model: {model_name}
tags:
- healthcare
- medical
- fine-tuned
- lora
- gpt-oss-20b
language:
- en
---

# {hf_repo_name}

This is a fine-tuned version of {model_name} using LoRA (Low-Rank Adaptation) on healthcare discharge data.

## Model Details

- **Base Model**: {model_name}
- **Fine-tuning Method**: LoRA (rank={lora_rank}, alpha={lora_alpha})
- **Training Data**: Healthcare discharge records
- **Training Samples**: {len(df)} records
- **Reasoning Level**: {reasoning_level}

## Training Configuration

- **Learning Rate**: {learning_rate}
- **Batch Size**: {batch_size}
- **Epochs**: {num_epochs}
- **Max Length**: {max_length}
- **Gradient Accumulation Steps**: {gradient_accumulation_steps}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")

# Use for inference
inputs = tokenizer("Your medical query here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Results

- **Final Loss**: {trainer.state.log_history[-1].get("train_loss", 0) if trainer.state.log_history else "N/A"}
- **Total Steps**: {trainer.state.global_step}
"""
            
            # Save and upload model card
            with open(f"{output_dir}/README.md", "w") as f:
                f.write(model_card_content)
            
            api.upload_file(
                path_or_fileobj=f"{output_dir}/README.md",
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model"
            )
            
            print(f"✅ Model successfully pushed to: https://huggingface.co/{repo_id}")
            
        except Exception as e:
            print(f"❌ Error pushing to HuggingFace Hub: {e}")
            print("Model saved locally but not uploaded to Hub")

    training_info = {
        "base_model_name": model_name,
        "model_name": model_name,
        "experiment_name": experiment_name,
        "final_loss": (trainer.state.log_history[-1].get("train_loss", 0) if trainer.state.log_history else 0),
        "total_steps": trainer.state.global_step,
        "reasoning_level": reasoning_level,
        "lora_config": {"rank": lora_rank, "alpha": lora_alpha, "enabled": use_lora},
        "training_args": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_epochs": num_epochs
        },
        "huggingface_repo": f"{hf_username}/{hf_repo_name}" if push_to_hub else None
    }
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    model_volume.commit()
    print("Done. Saved to:", output_dir)
    return training_info


@app.function(
    gpu="H100",  # pick a GPU with enough VRAM; for 4/8-bit you can try smaller GPUs
    volumes={"/models": model_volume, "/cache": cache_volume},
    memory=32*1024,
    scaledown_window=300
)
def serve_gpt_oss_20b(
    model_path: str,
    prompt: str,
    base_model_name: str = "openai/gpt-oss-20b",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    reasoning_level: str = "medium",
    load_in_8bit: bool = False,    # set True to try smaller GPUs (requires bitsandbytes in image)
    load_in_4bit: bool = False
):
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    # Check for cached base model first
    cache_path = f"/cache/{base_model_name.replace('/', '_')}"
    
    if os.path.exists(cache_path):
        print(f"Loading cached base model from {cache_path}")
        tokenizer = AutoTokenizer.from_pretrained(cache_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    quant_cfg = None
    if load_in_4bit or load_in_8bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "dtype": torch.bfloat16 if not (load_in_4bit or load_in_8bit) else None,
    }
    if load_in_4bit or load_in_8bit:
        model_kwargs["quantization_config"] = quant_cfg
    
    if os.path.exists(cache_path) and not (load_in_4bit or load_in_8bit):
        base_model = AutoModelForCausalLM.from_pretrained(cache_path, **model_kwargs)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, model_path)

    # chat-format fallback
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

    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Reasoning: {reasoning_level}"},
        {"role": "user", "content": prompt},
    ]
    formatted = render_chat(messages, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    # Slice generated tokens more robustly
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    result = {"prompt": prompt, "response": generated, "reasoning_level": reasoning_level, "model_path": model_path}
    print(f"\n=== INFERENCE RESULT ===")
    print(f"Prompt: {prompt}")
    print(f"Response: {generated}")
    print(f"========================\n")
    
    return result


@app.function(volumes={"/datasets": dataset_volume})
def upload_csv_data(csv_content: str, filename: str = "discharge-data.csv"):
    path = f"/datasets/{filename}"
    with open(path, "w") as f:
        f.write(csv_content)
    dataset_volume.commit()
    print(f"CSV uploaded to {path}")
    return path


@app.function(
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=1800,
)
def push_model_to_hub(
    model_path: str = "/models/gpt-oss-20b-finetune",
    hf_username: str = "nessa272",
    hf_repo_name: str = "gpt-oss-20b-finetune-mini",
    base_model_name: str = "openai/gpt-oss-20b",
):
    import os
    import json
    from huggingface_hub import HfApi, login
    
    print(f"Pushing model to HuggingFace Hub: {hf_username}/{hf_repo_name}")
    
    # Fix all config files to use correct base model name
    config_files = ['adapter_config.json', 'config.json', 'generation_config.json']
    
    for config_file in config_files:
        config_path = f"{model_path}/{config_file}"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update any references to cached paths
            updated = False
            for key in config:
                if isinstance(config[key], str) and '/cache/' in config[key]:
                    config[key] = base_model_name
                    updated = True
                elif key in ['base_model_name_or_path', '_name_or_path']:
                    config[key] = base_model_name
                    updated = True
            
            if updated:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"✅ Fixed {config_file} base model reference to: {base_model_name}")
    
    # Also check and fix any existing README.md
    readme_path = f"{model_path}/README.md"
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            readme_content = f.read()
        
        # Replace any cached paths in existing README
        if '/cache/' in readme_content:
            readme_content = readme_content.replace('/cache/openai_gpt-oss-20b', base_model_name)
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            print("✅ Fixed existing README.md base model references")
    
    # Get token from Modal secret - try multiple variable names
    hf_token = (os.environ.get("HUGGINGFACE_TOKEN") or 
               os.environ.get("HF_TOKEN") or 
               os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    
    if not hf_token:
        # Debug: print available environment variables
        print("Available environment variables:")
        for key in sorted(os.environ.keys()):
            if any(term in key.lower() for term in ['hf', 'hugging', 'token']):
                print(f"  {key}")
        
        raise ValueError("HuggingFace token not found in Modal secret environment variables")
    
    print("✅ Found HuggingFace token")
    
    # Login with token
    login(token=hf_token)
    
    # Create repository if it doesn't exist
    api = HfApi()
    repo_id = f"{hf_username}/{hf_repo_name}"
    
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        print(f"Repository {repo_id} created/verified")
    except Exception as e:
        print(f"Repository creation info: {e}")
    
    # Push the model files
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Fine-tuned GPT-OSS-20B on healthcare data"
    )
    
    # Create a model card
    model_card_content = f"""---
license: apache-2.0
base_model: {base_model_name}
tags:
- healthcare
- medical
- fine-tuned
- lora
- gpt-oss-20b
language:
- en
---

# {hf_repo_name}

This is a fine-tuned version of {base_model_name} using LoRA (Low-Rank Adaptation) on healthcare discharge data.

## Model Details

- **Base Model**: {base_model_name}
- **Fine-tuning Method**: LoRA (rank=32, alpha=64)
- **Training Data**: Healthcare discharge records (50,000 samples)
- **Training Loss**: 0.653
- **Total Steps**: 297

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("{base_model_name}")
tokenizer = AutoTokenizer.from_pretrained("{base_model_name}")

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")

# Use for inference
inputs = tokenizer("Your medical query here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Configuration

- **Learning Rate**: 1e-5
- **Batch Size**: 4
- **Epochs**: 3
- **Max Length**: 128
- **Gradient Accumulation Steps**: 4
- **GPUs**: 4x H100
"""
    
    # Save and upload model card
    with open(f"{model_path}/README.md", "w") as f:
        f.write(model_card_content)
    
    api.upload_file(
        path_or_fileobj=f"{model_path}/README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
    
    print(f"✅ Model successfully pushed to: https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


# --- Local CLI ---
@app.local_entrypoint()
def main(
    command: str = "train",
    csv_file: str = "discharge-data.csv",
    model_path: str = "/models/gpt-oss-20b-finetune",
    prompt: str = "Explain quantum computing in simple terms.",
    reasoning_level: str = "medium"
):
    if command == "train":
        print(f"Fine-tuning on {csv_file}")
        result = finetune_gpt_oss_20b.remote(
            csv_file_path=f"/datasets/{csv_file}",
            reasoning_level=reasoning_level,
            push_to_hub=True,
            hf_username="nessa272",
            hf_repo_name="gpt-oss-20b-finetune-mini",
            hf_token=None
        )
        print("Training info:", result)
    elif command == "serve":
        print(f"Serving… prompt: {prompt!r}")
        result = serve_gpt_oss_20b.remote(
            model_path=model_path,
            prompt=prompt,
            reasoning_level=reasoning_level
        )
        print("Response:", result["response"])
    elif command == "upload":
        print("Use upload_csv_data.remote(csv_content, 'discharge-data.csv') to upload.")
    elif command == "push":
        print("Pushing model to HuggingFace Hub...")
        result = push_model_to_hub.remote()
        print("Push result:", result)
    else:
        print("Unknown command. Use 'train', 'serve', 'upload', or 'push'.")
