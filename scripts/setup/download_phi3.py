import os
from huggingface_hub import hf_hub_download

def download_phi3():
    model_name = "microsoft/Phi-3-mini-4k-instruct-gguf"
    filename = "Phi-3-mini-4k-instruct-q4.gguf"
    save_dir = "models"
    
    print(f"Downloading {filename} from {model_name}...")
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = hf_hub_download(
        repo_id=model_name,
        filename=filename,
        local_dir=save_dir,
        local_dir_use_symlinks=False
    )
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    download_phi3()
