#!/usr/bin/env python3
"""
GPT-OSS-120B fine-tuning on Modal (multi-GPU H100) + Push to Hugging Face + Token Check

- Base: lmsys/gpt-oss-120b-bf16 (NON-MXFP4) to avoid MXFP4/BnB clashes
- QLoRA: BitsAndBytes NF4
- CSV streaming with hard cap at 100k rows (shuffle buffer)
- HF cache persisted to volume for faster re-runs
- Optional: creates a new Hugging Face repo and uploads LoRA adapter + tokenizer
- Includes: check_hf_token function to validate your HF token inside Modal
"""

import modal

# ---------- Container Image ----------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.3.0",
        "transformers>=4.42.0",
        "datasets>=2.19.0",
        "peft>=0.11.0",
        "pandas",
        "numpy",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.33.0",
        "huggingface_hub>=0.23.0",
        "requests>=2.31.0",  # for token check
    ])
    .apt_install(["git"])
)

app = modal.App("gpt-oss-120b-finetune-bnb", image=image)

# ---------- Volumes ----------
dataset_volume = modal.Volume.from_name("gpt-oss-datasets", create_if_missing=True)
model_volume   = modal.Volume.from_name("gpt-oss-120b-models", create_if_missing=True)


# ============ TRAINING (BnB NF4; NON-MXFP4 base) ============
@app.function(
    gpu="H100:4",  # 4x H100
    volumes={"/datasets": dataset_volume, "/models": model_volume},
    timeout=12 * 3600,
    memory=24 * 1024,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("hf-token")],  # expects HF_TOKEN in secret
)
def finetune_gpt_oss_120b(
    # Data / model
    csv_file_path: str = "/datasets/discharge-data.csv",
    model_name: str = "lmsys/gpt-oss-120b-bf16",  # NON-MXFP4 base
    model_revision: str | None = None,            # e.g. "main" or commit hash

    # Practical knobs (tuned for speed with short rows)
    learning_rate: float = 1e-5,
    max_steps: int = 150,
    logging_steps: int = 10,
    save_steps: int = 1000,  # final save still occurs at end

    # Throughput / memory (short text -> use bigger per-device batch)
    per_device_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    max_length: int = 64,

    # LoRA
    lora_rank: int = 16,
    lora_alpha: int = 32,

    # Streaming params
    shuffle_buffer: int = 10000,
    profile_steps: int = 0,   # set >0 to run warm-up profiling

    # Output
    experiment_name: str = "gpt-oss-120b-finetune",

    # Push to Hugging Face
    push_to_hub: bool = False,
    hf_repo: str | None = None,   # e.g. "your-username/gpt-oss-120b-finetune"
    hf_private: bool = True,
):
    import os, json, time, torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoConfig,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
    )
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, get_peft_model, TaskType

    # Persist HF cache (big speedup on subsequent runs)
    os.environ["HF_HOME"] = "/models/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/models/hf-cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision("high")  # TF32 on Hopper

    # --- Tokenizer ---
    print("Loading tokenizer:", model_name, f"(revision={model_revision})")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=model_revision, trust_remote_code=True, use_fast=True, cache_dir="/models/hf-cache"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Ensure non-MXFP4 base ---
    try:
        cfg = AutoConfig.from_pretrained(
            model_name, revision=model_revision, trust_remote_code=True, cache_dir="/models/hf-cache"
        )
        qconf = getattr(cfg, "quantization_config", None)
        if qconf is not None and "mxfp4" in type(qconf).__name__.lower():
            raise RuntimeError(
                "This checkpoint is MXFP4-quantized; please select a NON-MXFP4 (bf16/fp16) base."
            )
    except Exception as e:
        print(f"[WARN] Could not fully inspect model config: {e}")

    # --- Load with BnB NF4 (QLoRA) ---
    print("Loading base with BitsAndBytes NF4 (QLoRA)…")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=None,
        quantization_config=quant_cfg,
        cache_dir="/models/hf-cache",
        use_cache=False,
    )

    # --- LoRA ---
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

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # --- Minimal preprocess (short rows) ---
    def preprocess_example(ex):
        text = (
            f"Patient: {ex.get('Age Group','Unknown')} {ex.get('Gender','Unknown')}. "
            f"Diagnosis: {ex.get('CCS Diagnosis Description','Unknown')}."
        )
        toks = tokenizer(text, truncation=True, padding=False, max_length=max_length)
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "labels": toks["input_ids"].copy()
        }

    # --- Streaming dataset with hard cap = 100k rows ---
    print("Loading CSV (streaming):", csv_file_path)
    stream = load_dataset("csv", data_files=csv_file_path, split="train", streaming=True)

    # Peek + record columns to remove
    first = next(iter(stream.take(1)), None)
    if not first:
        raise ValueError("CSV appears empty.")
    for col in ["CCS Diagnosis Description", "Age Group", "Gender"]:
        if col not in first:
            raise ValueError(f"Missing required column: {col}")
    cols_to_remove = list(first.keys())

    # Recreate stream, shuffle, cap
    stream = load_dataset("csv", data_files=csv_file_path, split="train", streaming=True)
    stream = stream.shuffle(buffer_size=shuffle_buffer, seed=42).take(100_000)

    # Map -> ONLY token tensors fed to Trainer
    train_dataset = stream.map(
        preprocess_example, batched=False, remove_columns=cols_to_remove
    )

    # (Optional) quick debug on a few rows
    print("=== DEBUG: Sample tokenized rows (first 3) ===")
    for i, ex in enumerate(train_dataset.take(3)):
        print(f"Row {i}: input_ids[:20]={ex['input_ids'][:20]}  labels[:20]={ex['labels'][:20]}")
    print("=== END DEBUG ===")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
    output_dir = f"/models/{experiment_name}"

    # --- Training args ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="no",
        report_to=None,
        lr_scheduler_type="constant",  # keep LR fixed for short runs
        max_grad_norm=1.0,             # gradient clipping to prevent NaNs
        bf16=False, fp16=False, tf32=True,  # BnB manages compute dtype
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Optional warm-up profiling
    def profile(trainer, steps=0):
        if steps <= 0:
            return None
        orig = (trainer.args.max_steps, trainer.args.save_steps, trainer.args.logging_steps)
        trainer.args.max_steps, trainer.args.save_steps, trainer.args.logging_steps = steps, 10_000_000, max(1, steps // 5)
        t0 = time.time()
        trainer.train()
        dt = max(time.time() - t0, 1e-6)
        eff_bs = trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps
        tps = steps * eff_bs * max_length / dt
        print(f"[PROFILE] ~{tps:,.0f} tokens/sec over {steps} steps")
        trainer.args.max_steps, trainer.args.save_steps, trainer.args.logging_steps = orig
        return tps

    _ = profile(trainer, profile_steps)

    # Safe resume (if running again into same dir)
    last_ckpt = None
    try:
        last_ckpt = get_last_checkpoint(output_dir)
    except Exception:
        pass
    print(f"Resuming from: {last_ckpt if last_ckpt else 'scratch'}")

    print("Training…")
    trainer.train(resume_from_checkpoint=last_ckpt)

    print("Saving adapter + tokenizer…")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # --- Push adapter to Hugging Face (optional) ---
    if push_to_hub:
        from huggingface_hub import HfApi, whoami
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN not found. Create with: modal secret create hf-token --from-dotenv hf.env")

        api = HfApi(token=token)
        if hf_repo is None:
            user = whoami(token=token)["name"]
            hf_repo = f"{user}/{experiment_name}"

        print(f"Creating (or reusing) HF repo: {hf_repo} (private={hf_private})")
        api.create_repo(repo_id=hf_repo, private=hf_private, exist_ok=True)
        print(f"Uploading folder: {output_dir} → {hf_repo}")
        api.upload_folder(
            repo_id=hf_repo,
            folder_path=output_dir,
            commit_message="Upload LoRA adapter + tokenizer",
        )
        print(f"Pushed to https://huggingface.co/{hf_repo}")

    # Save metadata
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump({
            "global_step": trainer.state.global_step,
            "experiment_name": experiment_name,
            "base_model": model_name,
        }, f, indent=2)

    model_volume.commit()
    print("Done. Saved to:", output_dir)
    return {"model_dir": output_dir}


# ============ SERVING (BnB by default) ============
@app.function(
    gpu="H100",
    volumes={"/models": model_volume},
    memory=32 * 1024,
    scaledown_window=300
)
def serve_gpt_oss_120b(
    model_path: str,
    prompt: str = "Write a concise clinical summary.",
    base_model_name: str = "lmsys/gpt-oss-120b-bf16",
    base_model_revision: str | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    import os, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    os.environ["HF_HOME"] = "/models/hf-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/models/hf-cache"

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, revision=base_model_revision, trust_remote_code=True, cache_dir="/models/hf-cache"
    )

    quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        revision=base_model_revision,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=None,
        quantization_config=quant_cfg,
        cache_dir="/models/hf-cache",
    )

    model = PeftModel.from_pretrained(base_model, model_path)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"\n=== INFERENCE RESULT ===\nPrompt: {prompt}\nResponse: {generated}\n========================\n")
    return generated


# ============ TOKEN CHECK (run this to validate HF_TOKEN) ============
@app.function(secrets=[modal.Secret.from_name("hf-token")])
def check_hf_token():
    import os, requests
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ No HF_TOKEN found in environment.")
        return {"ok": False, "error": "missing_token"}

    print("✅ HF_TOKEN loaded, first 10 chars:", token[:10] + "...")
    r = requests.get("https://huggingface.co/api/whoami-v2",
                     headers={"Authorization": f"Bearer {token}"})
    if r.status_code == 200:
        print("✅ Hugging Face identity:", r.json())
        return {"ok": True, "identity": r.json()}
    else:
        print("❌ Failed auth, status:", r.status_code, r.text)
        return {"ok": False, "status": r.status_code, "body": r.text}


# ============ LOCAL CLI (simple helper) ============
@app.local_entrypoint()
def main(
    command: str = "train",
    csv_file: str = "discharge-data.csv",
    model_path: str = "/models/gpt-oss-120b-finetune",
    prompt: str = "Write a concise clinical summary.",
    push_to_hub: bool = False,
    hf_repo: str | None = None,
    hf_private: bool = True,
):
    if command == "train":
        finetune_gpt_oss_120b.remote(
            csv_file_path=f"/datasets/{csv_file}",
            push_to_hub=push_to_hub,
            hf_repo=hf_repo,
            hf_private=hf_private,
        )
    elif command == "serve":
        print(serve_gpt_oss_120b.remote(model_path=model_path, prompt=prompt))
    elif command == "check_hf":
        print(check_hf_token.remote())
    else:
        print("Unknown command. Use 'train', 'serve', or 'check_hf'.")
