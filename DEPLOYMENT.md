# Deployment Guide - Retinal Vessel Analysis System

## 🚀 Deployment Options

### 1. **Heroku** (Recommended for Quick Deployment)
```bash
# Install Heroku CLI, then:
heroku login
heroku create your-app-name
git push heroku main
```

### 2. **AWS EC2 / Azure VM**
- Launch Ubuntu instance
- Install Python, dependencies
- Use Gunicorn + Nginx
- Set up SSL certificate

### 3. **Google Cloud Run / Azure App Service**
- Containerized deployment
- Auto-scaling
- Managed infrastructure

### 4. **Railway / Render** (Easy alternatives to Heroku)
- Connect GitHub repo
- Auto-deploy on push
- Free tier available

## 📋 Pre-Deployment Checklist

### 1. Create Production Requirements
The current `requirements.txt` is ready, but add production server:
```
gunicorn>=21.0.0
```

### 2. Environment Variables
Create `.env` file (DO NOT commit this):
```
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
CLOUDINARY_CLOUD_NAME=dhonhnyuq
CLOUDINARY_API_KEY=843784586562915
CLOUDINARY_API_SECRET=U0aPHRuc_dNnXG-v_68od6ra1PE
```

### 3. Update app.py for Production
Replace the last line:
```python
# Development
if __name__ == '__main__':
    load_model_once()
    app.run(debug=True, host='0.0.0.0', port=5000)

# Production
if __name__ == '__main__':
    load_model_once()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
```

### 4. Create Procfile (for Heroku/Railway)
```
web: gunicorn app:app --timeout 120 --workers 2
```

### 5. Security Considerations
- Set `debug=False` in production
- Use environment variables for secrets
- Enable HTTPS/SSL
- Add rate limiting
- Implement user authentication (if needed)

## 🔧 Deployment Steps

### For Heroku:
```bash
# 1. Initialize git (if not already)
git init
git add .
git commit -m "Initial commit"

# 2. Create Heroku app
heroku create retinal-vessel-analysis

# 3. Set environment variables
heroku config:set FLASK_ENV=production
heroku config:set CLOUDINARY_CLOUD_NAME=dhonhnyuq
heroku config:set CLOUDINARY_API_KEY=843784586562915
heroku config:set CLOUDINARY_API_SECRET=U0aPHRuc_dNnXG-v_68od6ra1PE

# 4. Deploy
git push heroku main

# 5. Open app
heroku open
```

### For Railway:
1. Go to railway.app
2. Connect GitHub repository
3. Add environment variables in dashboard
4. Deploy automatically

### For Custom Domain:
1. Deploy to any platform above
2. Get your app URL (e.g., your-app.herokuapp.com)
3. In domain registrar (GoDaddy, Namecheap, etc.):
   - Add CNAME record: www → your-app.herokuapp.com
   - Add A record or ALIAS: @ → platform IP
4. Configure SSL (Let's Encrypt - free)

## 📦 Model File Considerations

**Important**: The model file `outputs/mvp_model.pt` is ~260MB. 

Options:
1. **Include in repo** (not recommended if >100MB)
2. **Use Git LFS** for large files
3. **Upload to cloud storage** (S3, Azure Blob, Google Cloud Storage)
4. **Use Cloudinary** (already configured)

## 🔐 Security Updates Needed

1. Add CORS restrictions (currently allows all)
2. Add rate limiting to prevent abuse
3. Add file size limits for uploads
4. Sanitize file uploads
5. Add API authentication if needed

## 🌐 Custom Domain Setup

After deployment, to use your own domain:
1. Get your platform URL (e.g., your-app.railway.app)
2. In your domain DNS settings:
   ```
   Type: CNAME
   Name: www
   Value: your-app.railway.app
   
   Type: A or ALIAS
   Name: @
   Value: [platform IP or use CNAME flattening]
   ```
3. Wait for DNS propagation (5-30 minutes)

## 📊 Monitoring & Logs

- Heroku: `heroku logs --tail`
- Railway: Check dashboard logs
- AWS: CloudWatch
- Add application monitoring (e.g., Sentry)

## 💰 Cost Estimates

- **Heroku**: $7-25/month (Eco/Basic dynos)
- **Railway**: $5-20/month (pay-as-you-go)
- **AWS EC2**: $10-50/month (t2.micro-medium)
- **Google Cloud Run**: Pay per request (very cheap for low traffic)
- **Azure App Service**: $13-50/month

## 🚨 Known Issues for Deployment

1. **Large model file**: Consider model compression or cloud storage
2. **CPU inference**: Slow on basic tiers (upgrade for GPU if needed)
3. **Memory**: Requires at least 1GB RAM for model loading
4. **Timeout**: Increase timeout for analysis (default 30s → 120s)

## ✅ Post-Deployment Testing

1. Test image upload functionality
2. Verify all three outputs display correctly
3. Check metrics computation
4. Test on mobile devices
5. Verify SSL certificate
6. Test under load

## 📱 Mobile Optimization

The current design is responsive, but test on:
- iPhone (Safari)
- Android (Chrome)
- Tablets

Would you like me to create these deployment files or help you set up on a specific platform?
