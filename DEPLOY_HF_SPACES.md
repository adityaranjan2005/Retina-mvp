# 🚀 Deploy to Hugging Face Spaces (FREE)

Hugging Face Spaces is perfect for ML models - free GPU/CPU, built for large models, no memory limits!

## Why HF Spaces?

✅ **FREE** forever
✅ Built for ML models (handles 800MB+ easily)
✅ GPU available (faster inference)
✅ No memory limits on free tier
✅ Automatic model caching
✅ Beautiful Gradio interface included

## Quick Deploy (5 minutes)

### Step 1: Create Hugging Face Space

1. Go to: https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   ```
   Space name: retinal-vessel-analysis
   License: MIT
   SDK: Gradio
   Space hardware: CPU basic (free)
   Visibility: Public
   ```
4. Click **"Create Space"**

### Step 2: Upload Files

You'll see an empty Space. Upload these files:

**Required files** (in order):
1. **app_gradio.py** → Rename to **app.py** (required name for HF Spaces)
2. **requirements_spaces.txt** → Rename to **requirements.txt**
3. **README_SPACES.md** → Rename to **README.md**
4. **src/** folder (entire folder with all Python files)

**How to upload**:
- Click "Files" tab in your Space
- Click "Add file" → "Upload files"
- Drag and drop OR click to browse
- Click "Commit changes to main"

### Step 3: Wait for Build

HF Spaces will automatically:
1. Install dependencies (3-5 minutes)
2. Download model from your HF Hub repo
3. Start Gradio app
4. Show "Running" status

Watch the build logs by clicking "App" → "View logs"

### Step 4: Test Your App

Once "Running", your app is live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/retinal-vessel-analysis
```

1. Upload a retinal image
2. Click "Analyze Image"
3. View results!

---

## Alternative: Git Push (Advanced)

If you prefer command line:

```bash
# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/retinal-vessel-analysis
cd retinal-vessel-analysis

# Copy files
cp ../Retina-mvp/app_gradio.py app.py
cp ../Retina-mvp/requirements_spaces.txt requirements.txt
cp ../Retina-mvp/README_SPACES.md README.md
cp -r ../Retina-mvp/src .

# Commit and push
git add .
git commit -m "Add retinal vessel analysis app"
git push
```

---

## File Checklist

In your HF Space, you should have:

```
retinal-vessel-analysis/
├── app.py                    # Main Gradio app (renamed from app_gradio.py)
├── requirements.txt          # Dependencies (from requirements_spaces.txt)
├── README.md                 # Space description (from README_SPACES.md)
└── src/
    ├── __init__.py          # Empty file (create if missing)
    ├── model.py             # Model architecture
    ├── metrics.py           # Metrics computation
    └── dataset.py           # (optional, not used in inference)
```

---

## Troubleshooting

### Build fails with "Module not found"

**Fix**: Add missing package to requirements.txt

Check build logs for exact package name.

### Model download fails

**Check**:
1. Your model repo is public: https://huggingface.co/adityaranjan2005/retina-vessel-segmentation
2. Model file exists: mvp_model.pt

### App won't start

**Check logs** in HF Space:
- Click "App" → "View logs"
- Look for Python errors
- Common issue: Missing `src/__init__.py` file

**Fix**: Create empty `src/__init__.py`:
```bash
# In your Space, add file: src/__init__.py (empty file)
```

### Inference is slow

**Upgrade to GPU** (still free):
1. Go to Space settings
2. Change "Space hardware" to "CPU basic" → "T4 small" (free)
3. Wait for rebuild
4. Much faster inference!

---

## Comparison: Render vs HF Spaces

| Feature | Render Free | HF Spaces Free |
|---------|-------------|----------------|
| RAM | 512 MB ❌ | 16 GB ✅ |
| Model size | Too small | ✅ No limit |
| GPU | ❌ No | ✅ Optional (T4) |
| Sleep | 15 min idle | ❌ No sleep |
| Build time | 15 min | 5 min |
| ML optimized | ❌ No | ✅ Yes |

**Winner**: Hugging Face Spaces for ML models! 🏆

---

## What You Get

Your live app at: `https://huggingface.co/spaces/YOUR_USERNAME/retinal-vessel-analysis`

Features:
- ✅ Upload retinal images
- ✅ Real-time analysis
- ✅ Vessel segmentation display
- ✅ Centerline extraction
- ✅ A/V classification (red=artery, blue=vein)
- ✅ Quantitative metrics
- ✅ Beautiful Gradio UI
- ✅ Shareable link
- ✅ Embed in websites

---

## Customization

### Add Example Images

Edit `app_gradio.py`:

```python
gr.Examples(
    examples=[
        ["examples/retina1.jpg"],
        ["examples/retina2.jpg"],
    ],
    inputs=input_image,
    label="Example Images"
)
```

Then upload example images to `examples/` folder in your Space.

### Enable GPU

1. Go to Space settings
2. Select "T4 small" GPU (free tier)
3. Save
4. Space will rebuild with GPU

Your app will automatically use CUDA if available.

### Custom Domain

HF Spaces Pro ($9/month) allows custom domains:
- `retina.yourdomain.com`
- Better branding
- White-label option

---

## Share Your App

Once deployed, share:
- Direct link: `https://huggingface.co/spaces/YOUR_USERNAME/retinal-vessel-analysis`
- Embed in website:
  ```html
  <iframe src="https://huggingface.co/spaces/YOUR_USERNAME/retinal-vessel-analysis" 
          width="100%" height="800"></iframe>
  ```
- Social media: HF auto-generates preview cards

---

## Cost

**FREE FOREVER** ✅

No credit card required.

Optional upgrades:
- **Persistent storage**: $5/month (saves user uploads)
- **Private Space**: $9/month (hide from public)
- **Better GPU (A10G)**: $9/month (faster inference)

But the free tier is perfect for most use cases!

---

## Next Steps

1. ✅ Deploy to HF Spaces (5 minutes)
2. Test with your retinal images
3. Share the link
4. (Optional) Add to your GitHub README:
   ```markdown
   **Live Demo**: [Try it on HF Spaces](https://huggingface.co/spaces/YOUR_USERNAME/retinal-vessel-analysis)
   ```

---

## Support

- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Gradio Docs**: https://gradio.app/docs/
- **Community**: https://discuss.huggingface.co/

---

**Ready to deploy?** Follow Step 1 above! 🚀
