# Running the Web Application

## Quick Start

1. **Install additional dependencies**:
```bash
pip install flask flask-cors cloudinary scipy
```

2. **Make sure you have a trained model**:
```bash
# If you haven't trained yet:
python -m src.train --epochs 2 --img_size 256 --batch_size 2
```

3. **Start the web server**:
```bash
python app.py
```

4. **Open your browser**:
```
http://localhost:5000
```

## Features

- 📤 **Drag & Drop Upload**: Easy image upload interface
- 🔬 **Real-time Analysis**: Instant vessel segmentation
- 📊 **Detailed Metrics**: Centerline length, branch points, tortuosity
- 🎨 **Visual Results**: 
  - Vessel segmentation
  - Centerline/skeleton
  - Artery/Vein classification
- ☁️ **Cloudinary Support**: Optional cloud storage for results

## API Endpoints

### POST /api/analyze
Upload and analyze a fundus image.

**Request**:
- Content-Type: multipart/form-data
- Body: 
  - image: Image file
  - use_cloudinary: "true" or "false" (optional)

**Response**:
```json
{
  "success": true,
  "vessel_image": "base64_or_url",
  "centerline_image": "base64_or_url",
  "av_image": "base64_or_url",
  "metrics": {
    "centerline_length_px": 2778.0,
    "branch_points": 189,
    "endpoints": 176,
    "tortuosity_proxy_mean": 1.666,
    "tortuosity_proxy_max": 3.162
  }
}
```

### GET /health
Check if the server is running and model is loaded.

## Configuration

Cloudinary credentials are configured in `app.py`:
- cloud_name: "dhonhnyuq"
- api_key: "843784586562915"
- api_secret: "U0aPHRuc_dNnXG-v_68od6ra1PE"

## Troubleshooting

**Model not found**:
- Make sure `outputs/mvp_model.pt` exists
- Train the model first using `python -m src.train`

**Port already in use**:
- Change the port in `app.py`: `app.run(port=5001)`

**Out of memory**:
- The model was trained with img_size=256, works well on CPU
- For larger images, train with bigger img_size on GPU
