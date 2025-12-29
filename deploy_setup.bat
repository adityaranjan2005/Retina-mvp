@echo off
echo ========================================
echo   RETINAL VESSEL ANALYSIS DEPLOYMENT
echo ========================================
echo.

echo Step 1: Installing dependencies...
pip install huggingface-hub

echo.
echo Step 2: Uploading model to Hugging Face Hub...
echo You'll need a Hugging Face token from: https://huggingface.co/settings/tokens
echo.
python upload_to_huggingface.py

echo.
echo ========================================
echo   NEXT STEPS:
echo ========================================
echo 1. Copy the model URL shown above
echo 2. Update HF_REPO in app_deploy.py with your username/repo
echo 3. Push to GitHub: git add -A; git commit -m "Deploy"; git push
echo 4. Go to https://render.com and create a new Web Service
echo 5. Connect your GitHub repo and add HF_REPO environment variable
echo.
echo Full instructions in DEPLOYMENT.md
pause
