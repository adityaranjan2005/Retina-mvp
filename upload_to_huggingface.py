"""
Upload trained model to Hugging Face Hub for deployment.
Run this once to upload your model.
"""
from huggingface_hub import HfApi, create_repo
import os

# Configuration
HF_TOKEN = input("Enter your Hugging Face token (from https://huggingface.co/settings/tokens): ").strip()
REPO_NAME = "retina-vessel-segmentation"  # Change if needed
MODEL_PATH = "outputs/mvp_model.pt"

def upload_model():
    """Upload model to Hugging Face Hub."""
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at {MODEL_PATH}")
        print("Please train the model first: python -m src.train")
        return
    
    print(f"📦 Model size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
    print(f"🚀 Uploading to Hugging Face Hub...")
    
    api = HfApi()
    
    # Create repository
    try:
        repo_url = create_repo(
            repo_id=REPO_NAME,
            token=HF_TOKEN,
            private=False,
            exist_ok=True
        )
        print(f"✅ Repository created/found: {repo_url}")
    except Exception as e:
        print(f"⚠️ Repository creation: {e}")
    
    # Upload model file
    try:
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo="mvp_model.pt",
            repo_id=REPO_NAME,
            token=HF_TOKEN,
        )
        print(f"✅ Model uploaded successfully!")
        print(f"\n📝 Your model URL:")
        print(f"https://huggingface.co/{api.whoami(token=HF_TOKEN)['name']}/{REPO_NAME}/resolve/main/mvp_model.pt")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    upload_model()
