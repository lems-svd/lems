import os
import shutil
import glob
import lm_eval
from huggingface_hub import snapshot_download

def sync_custom_tasks(src_folder, task_name, dataset_dir, hf_repo_id="regisss/math_qa"):  # script in allenai/math_qa not supported anymore.
    """
    Downloads the required dataset files if missing, then moves the custom 
    task folder to the lm-eval package directory.
    """
    # 1. Auto-Download Dataset Files (Parquet only)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Check if any parquet files already exist in the directory
    parquet_files = glob.glob(os.path.join(dataset_dir, "**/*.parquet"), recursive=True)
    
    if not parquet_files:
        print(f"[*] Dataset not found in '{dataset_dir}'. Attempting to download...")
        try:
            snapshot_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                local_dir=dataset_dir,
                allow_patterns="**/*.parquet" # We only want the raw data files
            )
            print(f"[+] Successfully downloaded dataset to {dataset_dir}")
        except Exception as e:
            print(f"[!] Download failed (likely blocked by firewall): {e}")
            print(f"[i] Please manually place the .parquet files into {dataset_dir}")
            return # Exit early since the task won't run without data
    else:
        print(f"[i] Dataset files already found in '{dataset_dir}'. Skipping download.")

    # 2. Sync Custom Task Config to lm-eval Package
    package_path = os.path.dirname(lm_eval.__file__)
    dest_folder = os.path.join(package_path, "tasks", task_name)

    if not os.path.exists(dest_folder):
        print(f"[*] Task '{task_name}' not found in package. Syncing from {src_folder}...")
        os.makedirs(os.path.dirname(dest_folder), exist_ok=True)
        
        shutil.copytree(src_folder, dest_folder)
        print(f"[+] Successfully synced config to: {dest_folder}")
    else:
        print(f"[i] Task '{task_name}' already exists in package directory. Skipping sync.")

# --- Example Usage ---
MY_REPO_TASK_PATH = "/workspace/KFAC-SVD/local_datasets/math_qa"
DATASET_SAVE_PATH = "/workspace/KFAC-SVD/local_datasets/math_qa/MathQA"

sync_custom_tasks(
    src_folder=MY_REPO_TASK_PATH, 
    task_name="math_qa_custom",
    dataset_dir=DATASET_SAVE_PATH,
    hf_repo_id="regisss/math_qa"
)