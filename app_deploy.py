"""
Deployment version of app.py that downloads model from Hugging Face Hub.
This version is optimized for cloud deployment.
"""
import os
import json
import base64
from io import BytesIO
from pathlib import Path
import requests
import gc
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
from huggingface_hub import hf_hub_download

# Memory optimization for free tier
torch.set_num_threads(1)
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

from src.model import MultiHeadRetinaModel
from src.metrics import compute_skeleton_metrics, postprocess_vessel_mask, extract_centerline_from_vessel
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
HF_REPO = os.environ.get("HF_REPO", "adityaranjan2005/retina-vessel-segmentation")
MODEL_FILENAME = "mvp_model.pt"
LOCAL_MODEL_PATH = "mvp_model.pt"

# Global model variable
model = None
img_size = 512
device = 'cpu'

def download_model_from_hf():
    """Download model from Hugging Face Hub."""
    print(f"📥 Downloading model from Hugging Face Hub: {HF_REPO}")
    
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=MODEL_FILENAME,
            cache_dir="."
        )
        print(f"✅ Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        # Try local file as fallback
        if os.path.exists(LOCAL_MODEL_PATH):
            print(f"📁 Using local model: {LOCAL_MODEL_PATH}")
            return LOCAL_MODEL_PATH
        raise

def load_model_once():
    """Load model once at startup."""
    global model, img_size, device
    
    # Download or locate model
    checkpoint_path = download_model_from_hf()
    
    device = 'cpu'  # Force CPU for free tier memory constraints
    print(f"🔧 Loading model on device: {device}")
    
    # Load with memory optimization
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = MultiHeadRetinaModel(
        encoder_name="resnet34",
        encoder_weights=None
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Free checkpoint memory
    del checkpoint
    gc.collect()
    
    img_size = 512  # Fixed size to avoid checkpoint dependency
    print(f"✅ Model loaded successfully (img_size={img_size}, device={device})")

def get_inference_transform(img_size: int) -> A.Compose:
    """Get inference transform."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def numpy_to_pil(arr: np.ndarray, mode: str = 'L') -> Image.Image:
    """Convert numpy array to PIL image."""
    if mode == 'L':
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr, mode=mode)

def predict_image(image: Image.Image) -> dict:
    """Run inference on an image."""
    image_rgb = image.convert('RGB')
    image_np = np.array(image_rgb)
    original_size = image_np.shape[:2]
    
    transform = get_inference_transform(img_size)
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    vessel_pred = torch.sigmoid(outputs['vessel']).cpu().numpy()[0, 0]
    centerline_pred = torch.sigmoid(outputs['centerline']).cpu().numpy()[0, 0]
    av_pred = torch.softmax(outputs['av'], dim=1).cpu().numpy()[0]
    av_pred = np.argmax(av_pred, axis=0)
    
    from scipy.ndimage import zoom
    
    h, w = original_size
    vessel_pred = zoom(vessel_pred, (h / vessel_pred.shape[0], w / vessel_pred.shape[1]), order=1)
    centerline_pred = zoom(centerline_pred, (h / centerline_pred.shape[0], w / centerline_pred.shape[1]), order=1)
    av_pred = zoom(av_pred, (h / av_pred.shape[0], w / av_pred.shape[1]), order=0)
    
    vessel_binary = (vessel_pred > 0.5).astype(np.uint8)
    centerline_binary = (centerline_pred > 0.5).astype(np.uint8)
    
    vessel_binary = postprocess_vessel_mask(vessel_binary, kernel_size=3) // 255
    centerline_binary = extract_centerline_from_vessel(vessel_binary)
    
    metrics = compute_skeleton_metrics(centerline_binary)
    
    return {
        'vessel': vessel_binary,
        'centerline': centerline_binary,
        'av': av_pred.astype(np.uint8),
        'metrics': metrics
    }

def create_av_visualization(av_mask: np.ndarray) -> np.ndarray:
    """Create a colorized A/V visualization."""
    h, w = av_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[av_mask == 1] = [255, 0, 0]
    colored[av_mask == 2] = [0, 0, 255]
    return colored

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        image = Image.open(file.stream)
        results = predict_image(image)
        
        vessel_img = numpy_to_pil(results['vessel'])
        centerline_img = numpy_to_pil(results['centerline'])
        av_colored = create_av_visualization(results['av'])
        av_img = Image.fromarray(av_colored, mode='RGB')
        
        response = {
            'success': True,
            'vessel_image': pil_to_base64(vessel_img),
            'centerline_image': pil_to_base64(centerline_img),
            'av_image': pil_to_base64(av_img),
            'metrics': results['metrics']
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': device
    })

if __name__ == '__main__':
    print("🚀 Starting Retinal Vessel Analysis System...")
    load_model_once()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
