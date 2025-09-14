#!/usr/bin/env python3
"""
Simple script to upload train-test.csv to Modal volume
"""
import modal

# Create Modal app for upload
app = modal.App("data-upload")
dataset_volume = modal.Volume.from_name("gpt-oss-datasets", create_if_missing=True)

@app.function(
    volumes={"/datasets": dataset_volume},
    image=modal.Image.debian_slim().pip_install("pandas")
)
def upload_csv(csv_content: str):
    """Upload CSV content to Modal volume"""
    
    # Save CSV content to Modal volume
    modal_path = "/datasets/train-test.csv"
    with open(modal_path, "w") as f:
        f.write(csv_content)
    
    # Commit the volume
    dataset_volume.commit()
    
    # Count rows for confirmation
    lines = csv_content.strip().split('\n')
    row_count = len(lines) - 1  # Subtract header row
    
    print(f"Successfully uploaded CSV with {row_count} rows to {modal_path}")
    return row_count

@app.local_entrypoint()
def main():
    """Upload the CSV file"""
    # Read local CSV file as text (avoiding pandas locally)
    local_path = "/Users/nessatong/Desktop/CS Projects/Hackmit25/train-test.csv"
    
    try:
        with open(local_path, "r") as f:
            csv_content = f.read()
        
        print(f"Read local CSV file: {len(csv_content)} characters")
        
        # Upload to Modal
        result = upload_csv.remote(csv_content)
        print(f"Upload completed: {result} rows uploaded")
        
    except FileNotFoundError:
        print(f"Error: Could not find {local_path}")
        print("Make sure the train-test.csv file exists in the correct location")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    main()
