"""
Download model checkpoint from cloud storage.
Run this before starting the Flask app in production.
"""
import os
import requests
from pathlib import Path

def download_file_from_google_drive(file_id, destination):
    """Download large file from Google Drive with proper handling."""
    # Use multiple approaches for Google Drive
    urls_to_try = [
        f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
        f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
        f"https://docs.google.com/uc?export=download&id={file_id}"
    ]
    
    session = requests.Session()
    
    for idx, url in enumerate(urls_to_try, 1):
        try:
            print(f"   Trying URL format {idx}/{len(urls_to_try)}...")
            response = session.get(url, stream=True, timeout=30)
            
            # Check if we got actual file content (not an HTML error page)
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                print(f"   Got HTML response (likely error page), trying next URL...")
                continue
                
            if response.status_code == 200:
                print(f"   ✅ Successfully connected, downloading...")
                save_response_content(response, destination)
                return True
        except Exception as e:
            print(f"   ❌ Attempt {idx} failed: {e}")
            continue
    
    raise Exception("All Google Drive download methods failed")

def get_confirm_token(response):
    """Extract confirmation token from Google Drive response."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Save streaming response to file with progress."""
    CHUNK_SIZE = 32768
    total_size = 0
    
    try:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
                    print(f"   Downloaded: {total_size / 1024 / 1024:.1f} MB", end='\r')
        print()
        print(f"   Total downloaded: {total_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"\n   ❌ Error while saving: {e}")
        raise

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
    print(f"✅ Created outputs directory: {output_dir.absolute()}")
    
    model_path = output_dir / "mvp_model.pt"
    
    # Skip if model already exists
    if model_path.exists():
        file_size = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model already exists at {model_path} ({file_size:.1f} MB)")
        return True
    
    print(f"📥 Downloading model from: {model_url}")
    print(f"📥 Target location: {model_path.absolute()}")
    
    try:
        # Check if it's a Google Drive URL
        if 'drive.google.com' in model_url:
            # Extract file ID from URL
            if '/d/' in model_url:
                file_id = model_url.split('/d/')[1].split('/')[0]
            elif 'id=' in model_url:
                file_id = model_url.split('id=')[1].split('&')[0]
            else:
                raise ValueError("Could not extract file ID from Google Drive URL")
            
            print(f"   Google Drive file ID: {file_id}")
            download_file_from_google_drive(file_id, model_path)
        else:
            # Regular URL download
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
            print()
        
        file_size = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model downloaded successfully to {model_path} ({file_size:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        if model_path.exists():
            model_path.unlink()  # Remove partial download
        return False

if __name__ == "__main__":
    success = download_model()
    exit(0 if success else 1)
