"""
Download model checkpoint from cloud storage.
Run this before starting the Flask app in production.
"""
import os
import requests
from pathlib import Path

def download_model():
    """Download model from cloud storage URL."""
    
    # Get model URL from environment variable
    model_url = os.environ.get('MODEL_URL', '')
    
    if not model_url:
        print("⚠️  MODEL_URL not set. Skipping model download.")
        print("   Set MODEL_URL environment variable to download model automatically.")
        return False
    
    # Create outputs directory if it doesn't exist
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "mvp_model.pt"
    
    # Skip if model already exists
    if model_path.exists():
        print(f"✅ Model already exists at {model_path}")
        return True
    
    print(f"📥 Downloading model from: {model_url}")
    
    try:
        # Download with progress
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"   Progress: {percent:.1f}%", end='\r')
        
        print(f"\n✅ Model downloaded successfully to {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        if model_path.exists():
            model_path.unlink()  # Remove partial download
        return False

if __name__ == "__main__":
    download_model()
