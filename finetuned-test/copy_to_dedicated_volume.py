"""
Script to copy the merged model from cache to a dedicated volume.
"""

import modal
import shutil
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.38.0",
    ])
)

app = modal.App("copy-merged-model", image=image)

# Existing cache volume
cache_volume = modal.Volume.from_name("gpt-oss-cache", create_if_missing=True)

# New dedicated volume for merged models
merged_models_volume = modal.Volume.from_name("merged-models", create_if_missing=True)

@app.function(
    volumes={
        "/cache": cache_volume,
        "/merged_models": merged_models_volume
    },
    timeout=300,
)
def copy_merged_model():
    """
    Copy the merged model from cache to dedicated volume.
    """
    source_path = "/cache/gpt-oss-20b-finetune-mini"
    dest_path = "/merged_models/gpt-oss-20b-finetune-mini"
    
    if os.path.exists(source_path) and os.listdir(source_path):
        print(f"📂 Copying merged model from {source_path} to {dest_path}")
        
        # Create destination directory
        os.makedirs(dest_path, exist_ok=True)
        
        # Copy all files
        for item in os.listdir(source_path):
            src_item = os.path.join(source_path, item)
            dst_item = os.path.join(dest_path, item)
            
            if os.path.isdir(src_item):
                shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
            else:
                shutil.copy2(src_item, dst_item)
        
        print("✅ Merged model copied successfully!")
        return {"status": "success", "message": "Model copied to dedicated volume"}
    else:
        print("❌ Source merged model not found")
        return {"status": "error", "message": "Source model not found"}

if __name__ == "__main__":
    print("Use 'modal run copy_to_dedicated_volume.py::copy_merged_model' to copy the model")
