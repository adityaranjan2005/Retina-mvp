# Model Deployment Guide

## Problem
The trained model file (`outputs/mvp_model.pt`) is **839.79 MB**, which exceeds GitHub's 100 MB file size limit.

## Solution: Use Cloudinary (Recommended)

You already have Cloudinary configured! Just upload your model:

### Quick Upload

Run the upload script:

```bash
python upload_model_to_cloudinary.py
```

This will:
1. Upload `outputs/mvp_model.pt` to your Cloudinary account
2. Display the secure URL
3. Show you the exact MODEL_URL to use in Render

The URL will look like:
```
https://res.cloudinary.com/dhonhnyuq/raw/upload/v1234567890/retina-mvp/mvp_model.pt
```

### Alternative Cloud Storage Options

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

### 1. Upload Model to Cloudinary ✅ (Easiest)

```bash
python upload_model_to_cloudinary.py
```

Copy the displayed URL (starts with `https://res.cloudinary.com/...`)

### 2. Push to GitHub ✅ (Already Done)

Your code is already on GitHub without the large model file.

### 3. Deploy on Render

1. Go to [render.com](https://render.com) and sign in with GitHub
2. Click "New +" → "Web Service"
3. Connect your repository: `adityaranjan2005/Retina-mvp`
4. Render will auto-detect settings from `render.yaml`
5. **Important:** Add environment variable:
   - Key: `MODEL_URL`
   - Value: (paste the Cloudinary URL from step 1)
6. Click "Create Web Service"

### 4. Verify Deployment

Check Render logs for:
- ✅ `Model downloaded successfully`
- ✅ `Model loaded at startup`

Your app will be live at: `https://retinal-vessel-analysis.onrender.com`

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
