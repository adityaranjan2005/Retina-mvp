# ✅ Deployment Checklist

Follow these steps to deploy your retinal vessel analysis system globally.

## Current Status

✅ Model trained (800MB+)
✅ Model uploaded to Hugging Face Hub
✅ Deployment app tested locally (running successfully)
✅ All deployment files created
⏳ **Next: Push to GitHub and deploy to Render**

---

## Step 1: Prepare Git

### 1.1 Check if Git is initialized
Open a new terminal and run:
```powershell
cd C:\Users\YOGA\OneDrive\Desktop\Retina-mvp
git status
```

**If you see "fatal: not a git repository"**, initialize:
```powershell
git init
```

### 1.2 Check Git configuration
```powershell
git config user.name
git config user.email
```

**If empty**, configure:
```powershell
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

---

## Step 2: Commit Your Code

### 2.1 Add all files
```powershell
git add -A
```

### 2.2 Check what will be committed
```powershell
git status
```

You should see files like:
- src/dataset.py
- src/model.py
- app_deploy.py
- requirements.txt
- render.yaml
- etc.

**You should NOT see**:
- outputs/mvp_model.pt (ignored)
- upload_simple.py (ignored - contains token)
- data/images/*.ppm (ignored - too large)

### 2.3 Commit
```powershell
git commit -m "Add Retinal Vessel Analysis MVP with deployment config"
```

---

## Step 3: Create GitHub Repository

### 3.1 Go to GitHub
Open browser: https://github.com/new

### 3.2 Fill in details
- **Repository name**: `retina-vessel-analysis`
- **Description**: "AI-powered retinal vessel segmentation with A/V classification"
- **Visibility**: ✅ **PUBLIC** (required for Render free tier)
- **Initialize repository**: ❌ Leave all checkboxes UNCHECKED
- Click **"Create repository"**

### 3.3 Copy the commands shown
GitHub will show commands like:
```bash
git remote add origin https://github.com/YOUR_USERNAME/retina-vessel-analysis.git
git branch -M main
git push -u origin main
```

### 3.4 Run those commands
In your terminal:
```powershell
# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/retina-vessel-analysis.git
git branch -M main
git push -u origin main
```

**You may be asked for GitHub credentials**:
- Username: your GitHub username
- Password: use a **Personal Access Token** (not your GitHub password)
  - Get token at: https://github.com/settings/tokens
  - Click "Generate new token (classic)"
  - Select: `repo` (Full control of private repositories)
  - Copy the token and paste it when asked for password

### 3.5 Verify
Go to: https://github.com/YOUR_USERNAME/retina-vessel-analysis

You should see all your files there (except those in .gitignore).

---

## Step 4: Deploy to Render

### 4.1 Create Render account
1. Go to: https://render.com
2. Click "Get Started for Free"
3. Sign up with GitHub (easiest - auto-connects repos)
4. Authorize Render to access your repositories

### 4.2 Create Web Service
1. In Render Dashboard, click **"New +"** (top right)
2. Select **"Web Service"**
3. Find your repository: `retina-vessel-analysis`
4. Click **"Connect"**

### 4.3 Configure service

**Fill in the form**:

```
Name: retina-vessel-analysis
(This becomes your URL: retina-vessel-analysis.onrender.com)

Region: Oregon (US West)
(Or choose closest to you)

Branch: main

Runtime: Python 3

Build Command: pip install -r requirements.txt

Start Command: python app_deploy.py

Instance Type: Free
```

### 4.4 Add environment variable

Scroll down to **"Environment Variables"** or click **"Advanced"**:

```
Key: HF_REPO
Value: adityaranjan2005/retina-vessel-segmentation
```

Click **"Add"**

### 4.5 Deploy!

1. Review all settings
2. Scroll to bottom
3. Click **"Create Web Service"**
4. Deployment starts automatically

---

## Step 5: Monitor Deployment

### 5.1 Watch logs

Render will show real-time build logs. You'll see:

```
==> Installing dependencies
Collecting torch==2.0.0
...
==> Starting application
🚀 Starting Retinal Vessel Analysis System...
📥 Downloading model from Hugging Face Hub
mvp_model.pt: 100%|██████████| 881M/881M [03:45<00:00]
✅ Model loaded successfully
⚕️ Retinal Vessel Analysis System Ready
```

### 5.2 Wait for completion

**Timeline**:
- 0-3 min: Installing Python packages
- 3-12 min: Downloading 800MB model
- 12-15 min: Starting Flask app

### 5.3 Success!

When you see: **"Your service is live 🎉"**

Your URL will appear at the top: `https://retina-vessel-analysis.onrender.com`

---

## Step 6: Test Your Deployment

### 6.1 Open in browser

Go to: https://retina-vessel-analysis.onrender.com

You should see the professional black/white UI.

### 6.2 Test health endpoint

In terminal or browser:
```
https://retina-vessel-analysis.onrender.com/health
```

Should return:
```json
{
  "status": "healthy",
  "message": "Retinal Vessel Analysis System is running"
}
```

### 6.3 Upload test image

1. Click the upload area
2. Select any retinal image (from data/images/)
3. Click "Analyze Image"
4. Wait 10-30 seconds
5. View results:
   - Vessel segmentation
   - Centerline extraction
   - A/V classification
   - Metrics

---

## 🎉 You're Done!

Your application is now live and accessible worldwide!

**Your URL**: https://retina-vessel-analysis.onrender.com

**Share it with**:
- Researchers
- Medical professionals
- Students
- Anyone interested in retinal analysis

---

## 🔧 Troubleshooting

### Git push fails
**Error**: "remote: Permission denied"

**Fix**: Use Personal Access Token instead of password
- Get token: https://github.com/settings/tokens
- Generate new token (classic) with `repo` scope
- Use token as password when pushing

### Render build fails
**Check**: Build logs for specific error

**Common fixes**:
1. Verify requirements.txt is in root directory
2. Ensure all imports are installed
3. Check Python version in runtime.txt matches your code

### Model download fails
**Check**: HF_REPO environment variable is correct

**Fix**: 
1. Go to Render Dashboard → Your Service → Environment
2. Verify: `HF_REPO=adityaranjan2005/retina-vessel-segmentation`
3. Re-deploy if changed

### App doesn't start
**Check**: Start command in Render settings

**Should be**: `python app_deploy.py` (NOT app.py)

---

## 📞 Need Help?

1. Check [DEPLOYMENT_STEPS.md](DEPLOYMENT_STEPS.md) for detailed guide
2. Review Render logs for specific errors
3. Test app_deploy.py locally first
4. Verify all environment variables are set

---

## ⏱️ Current Step

You are here: **Step 1 - Prepare Git**

**Next action**: Open a NEW terminal (stop the running app_deploy.py first with Ctrl+C) and run:

```powershell
cd C:\Users\YOGA\OneDrive\Desktop\Retina-mvp
git status
```

Then follow the steps above!

---

**Good luck! You're almost there! 🚀**
