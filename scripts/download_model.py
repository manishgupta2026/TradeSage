from huggingface_hub import hf_hub_download
import os

MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODELS_DIR = "z:/Trade AI/models"

def download_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    destination = os.path.join(MODELS_DIR, MODEL_FILENAME)
    
    if os.path.exists(destination):
        print(f"Model already exists at {destination}")
        return

    print(f"Downloading {MODEL_FILENAME} from {MODEL_REPO}...")
    try:
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download model: {e}")

if __name__ == "__main__":
    download_model()
