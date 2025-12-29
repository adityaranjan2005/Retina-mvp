# Deployment Guide

## Option 1: Deploy to Render (Recommended - FREE)

Render provides free tier with model stored on Hugging Face Hub.

### Step 1: Upload Model to Hugging Face Hub

1. **Get Hugging Face Token**:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "Write" access
   
2. **Install Hugging Face Hub**:
   ```bash
   pip install huggingface-hub
   ```

3. **Upload your model**:
   ```bash
   python upload_to_huggingface.py
   ```
   - Enter your HF token when prompted
   - Copy the model URL shown (you'll need it)

### Step 2: Deploy to Render

1. **Push code to GitHub**:
   ```bash
   git add -A
   git commit -m "Add deployment files"
   git push origin master
   ```

2. **Create Render account**: https://render.com (free)

3. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: retina-vessel-analysis
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `python app_deploy.py`
     - **Instance Type**: Free

4. **Add Environment Variable**:
   - Go to "Environment" tab
   - Add: `HF_REPO` = `YOUR_USERNAME/retina-vessel-segmentation`
   - Replace with your actual Hugging Face username and repo name

5. **Deploy**: Click "Create Web Service"
   - First deploy takes 10-15 minutes
   - Model downloads from HF Hub (happens once, then cached)

6. **Your app will be live at**: `https://retina-vessel-analysis.onrender.com`

---

## Option 2: Deploy to Hugging Face Spaces (Alternative - FREE)

Hugging Face Spaces is designed for ML demos.

### Step 1: Create Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - **Space name**: retina-vessel-analysis
   - **Space SDK**: Gradio or Docker (Docker for Flask)
   - **Private**: No

### Step 2: Upload Files

1. **Clone the space**:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/retina-vessel-analysis
   cd retina-vessel-analysis
   ```

2. **Copy files**:
   ```bash
   # Copy these files to the space directory:
   - app_deploy.py (rename to app.py)
   - requirements.txt
   - src/ (entire folder)
   - templates/ (entire folder)
   ```

3. **Add model to space** (if small enough) or use HF Hub download

4. **Push to HF Space**:
   ```bash
   git add -A
   git commit -m "Initial commit"
   git push
   ```

5. **Space will be live at**: `https://huggingface.co/spaces/YOUR_USERNAME/retina-vessel-analysis`

---

## Option 3: Deploy to Railway (Alternative - FREE with limits)

1. **Create account**: https://railway.app
2. **Deploy from GitHub**: Connect repo
3. **Add environment variable**: `HF_REPO`
4. **Deploy**: Railway auto-detects Python

---

## Requirements for Deployment

All platforms need:
- `requirements.txt` with all dependencies
- `huggingface-hub` added to requirements
- Model uploaded to Hugging Face Hub (free, unlimited size for public repos)

## Model Storage Solutions

### Hugging Face Hub (Recommended - FREE)
- ✅ Designed for ML models
- ✅ Unlimited size for public repos
- ✅ Fast CDN delivery
- ✅ Version control
- Use: Upload with `upload_to_huggingface.py`

### Alternative: Google Drive
If you prefer Google Drive:
1. Upload model to Google Drive
2. Make it publicly accessible
3. Get shareable link
4. Use in app_deploy.py

---

## Testing Deployment

After deployment, test these endpoints:
1. **Main app**: `https://your-app-url.com/`
2. **Health check**: `https://your-app-url.com/health`

---

## Troubleshooting

### Model Download Fails
- Check `HF_REPO` environment variable
- Ensure model is public on Hugging Face
- Check logs for specific errors

### Out of Memory
- Reduce `img_size` in model checkpoint
- Use CPU instead of GPU (already configured)

### Slow Cold Starts
- First request after inactivity downloads model (~2-3 min)
- Subsequent requests are fast
- Consider paid tier for always-on instance

---

## Cost Estimate

All these options are **FREE** for your use case:
- **Render Free Tier**: 750 hours/month
- **Hugging Face Spaces**: Unlimited (with CPU)
- **Railway**: $5 credit/month (enough for light usage)

For production (paid tiers start at $7-20/month):
- More compute
- Always-on instance
- Custom domain
