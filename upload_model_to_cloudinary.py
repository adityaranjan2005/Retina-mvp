"""
Upload model checkpoint to Cloudinary.
Run this once to upload your model to cloud storage.
"""
import cloudinary
import cloudinary.uploader

# Configure Cloudinary
cloudinary.config(
    cloud_name="dhonhnyuq",
    api_key="843784586562915",
    api_secret="U0aPHRuc_dNnXG-v_68od6ra1PE",
    timeout=600  # 10 minutes timeout for large files
)

def upload_model():
    """Upload model to Cloudinary as raw file."""
    
    model_path = "outputs/mvp_model.pt"
    
    print(f"📤 Uploading {model_path} to Cloudinary...")
    print("⏳ This may take 5-10 minutes for large files...")
    
    try:
        # Upload as raw resource type (not image) with large file settings
        result = cloudinary.uploader.upload_large(
            model_path,
            resource_type="raw",
            public_id="retina-mvp/mvp_model",
            chunk_size=6000000,  # 6MB chunks
            overwrite=True,
            invalidate=True
        )
        
        model_url = result['secure_url']
        
        print(f"\n✅ Model uploaded successfully!")
        print(f"\n📋 Copy this URL for your MODEL_URL environment variable:\n")
        print(f"   {model_url}")
        print(f"\n💡 Add this to Render environment variables:")
        print(f"   MODEL_URL={model_url}")
        
        return model_url
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(f"\n💡 Alternative: Use the Cloudinary web dashboard")
        print(f"   1. Go to https://console.cloudinary.com/")
        print(f"   2. Upload → Raw → Choose outputs/mvp_model.pt")
        print(f"   3. After upload, copy the URL")
        return None

if __name__ == "__main__":
    upload_model()
