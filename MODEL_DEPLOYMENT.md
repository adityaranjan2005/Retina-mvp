# Model Deployment Guide

## Problem
The trained model file (`outputs/mvp_model.pt`) is **839.79 MB**, which exceeds GitHub's 100 MB file size limit.

## Solutions

### Option 1: Upload to Cloud Storage (Recommended)

Upload your model to a cloud storage service and set the download URL:

#### A. Google Drive
1. Upload `outputs/mvp_model.pt` to Google Drive
2. Right-click → Share → Anyone with the link can view
3. Get the file ID from the sharing link: `https://drive.google.com/file/d/FILE_ID/view`
4. Use this URL format: `https://drive.google.com/uc?export=download&id=FILE_ID`

#### B. Dropbox
1. Upload `outputs/mvp_model.pt` to Dropbox
2. Get sharing link
3. Replace `www.dropbox.com` with `dl.dropboxusercontent.com` in the URL
4. Remove `?dl=0` from the end

#### C. Cloudinary (Your Account)
```bash
# Upload using Cloudinary CLI
cloudinary upload outputs/mvp_model.pt --resource_type raw
```
Then get the secure URL from Cloudinary dashboard.

#### D. AWS S3 / Azure Blob Storage
Upload the file and generate a public URL or use pre-signed URLs.

### Option 2: Git Large File Storage (LFS)

If you want to keep the model in Git:

```bash
# Install Git LFS
# Windows: Download from https://git-lfs.github.com/
# Mac: brew install git-lfs
# Linux: apt-get install git-lfs

# Initialize Git LFS
git lfs install

# Track model files
git lfs track "*.pt"
git lfs track "*.pth"

# Add .gitattributes
git add .gitattributes

# Re-add the model
git add outputs/mvp_model.pt
git commit -m "Add model with Git LFS"
git push origin master
```

**Note:** GitHub provides 1 GB free LFS storage, then $5/month per 50 GB.

## Current Setup

The repository is configured to:
1. Ignore model files (see `.gitignore`)
2. Download model from `MODEL_URL` environment variable during deployment
3. Use `download_model.py` script in the build process

## Deployment Steps

### 1. Remove Model from Git History

```bash
# Remove the model from git cache (keep local file)
git rm --cached outputs/mvp_model.pt
git rm --cached outputs/*.png outputs/*.json

# Commit the removal
git add .gitignore
git commit -m "Remove large model file from git"

# Push to GitHub
git push origin master
```

### 2. Upload Model to Cloud Storage

Choose one of the cloud storage options above and upload your model file.

### 3. Configure Render Deployment

In Render dashboard, set the `MODEL_URL` environment variable:

```
MODEL_URL=https://your-storage-url/mvp_model.pt
```

Or update `render.yaml`:

```yaml
- key: MODEL_URL
  value: https://your-storage-url/mvp_model.pt
```

### 4. Deploy

Push to GitHub and Render will:
1. Install dependencies
2. Run `download_model.py` to fetch the model
3. Start the Flask app

## Verification

After deployment, check Render logs for:
- ✅ Model downloaded successfully
- ✅ Model loaded at startup

## Alternative: Retrain on Deployment

If you don't want to manage cloud storage, you can retrain the model on first deployment:

1. Keep training data small or download it during build
2. Add training step to build command:
   ```yaml
   buildCommand: pip install -r requirements.txt && python src/train.py
   ```

**Caution:** This increases deployment time significantly (~5-10 minutes) and requires sufficient build resources.

## Current Model Details

- **File:** `outputs/mvp_model.pt`
- **Size:** 839.79 MB
- **Architecture:** Multi-head U-Net (3 separate models)
- **Input Size:** 256×256
- **Device:** CPU (change to 'cuda' for GPU inference)
