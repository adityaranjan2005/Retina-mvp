import os
import json
import base64
from io import BytesIO
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
import cloudinary
import cloudinary.uploader

from src.model import MultiHeadRetinaModel
from src.metrics import compute_skeleton_metrics, postprocess_vessel_mask, extract_centerline_from_vessel
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Cloudinary
cloudinary.config(
    cloud_name="dhonhnyuq",
    api_key="843784586562915",
    api_secret="U0aPHRuc_dNnXG-v_68od6ra1PE"
)

# Global model variable
model = None
img_size = 256
device = 'cpu'

def load_model_once():
    """Load model once at startup. Download if not exists."""
    global model, img_size, device
    
    checkpoint_path = "outputs/mvp_model.pt"
    
    # Download model if it doesn't exist
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Model not found at {checkpoint_path}")
        print("📥 Downloading model from cloud storage...")
        try:
            # Import and run download script
            import sys
            sys.path.insert(0, os.path.dirname(__file__))
            from download_model import download_model
            
            if not download_model():
                raise FileNotFoundError("Model download failed")
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model file still not found after download at {checkpoint_path}")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            raise
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from {checkpoint_path} on {device}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MultiHeadRetinaModel(
        encoder_name="resnet34",
        encoder_weights=None
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    img_size = checkpoint.get('img_size', 256)
    print(f"✅ Model loaded successfully (img_size={img_size}, device={device})")

# Load model when gunicorn starts (not just in __main__)
print("=" * 60)
print("Initializing application...")
try:
    load_model_once()
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
print("=" * 60)

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
        # For grayscale (vessel, centerline)
        arr = (arr * 255).astype(np.uint8)
    else:
        # For A/V mask, map 0/1/2 to colors
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr, mode=mode)

def predict_image(image: Image.Image) -> dict:
    """Run inference on an image."""
    # Convert to RGB and numpy
    image_rgb = image.convert('RGB')
    image_np = np.array(image_rgb)
    original_size = image_np.shape[:2]
    
    # Transform
    transform = get_inference_transform(img_size)
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Post-process predictions
    vessel_pred = torch.sigmoid(outputs['vessel']).cpu().numpy()[0, 0]
    centerline_pred = torch.sigmoid(outputs['centerline']).cpu().numpy()[0, 0]
    av_pred = torch.softmax(outputs['av'], dim=1).cpu().numpy()[0]
    av_pred = np.argmax(av_pred, axis=0)
    
    # Resize to original size
    from scipy.ndimage import zoom
    
    h, w = original_size
    vessel_pred = zoom(vessel_pred, (h / vessel_pred.shape[0], w / vessel_pred.shape[1]), order=1)
    centerline_pred = zoom(centerline_pred, (h / centerline_pred.shape[0], w / centerline_pred.shape[1]), order=1)
    av_pred = zoom(av_pred, (h / av_pred.shape[0], w / av_pred.shape[1]), order=0)
    
    # Threshold
    vessel_binary = (vessel_pred > 0.5).astype(np.uint8)
    centerline_binary = (centerline_pred > 0.5).astype(np.uint8)
    
    # Postprocess vessel
    vessel_binary = postprocess_vessel_mask(vessel_binary, kernel_size=3) // 255
    
    # Re-extract centerline from postprocessed vessel
    centerline_binary = extract_centerline_from_vessel(vessel_binary)
    
    # Compute metrics
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
    
    # Background = black (0,0,0)
    # Artery = red (255,0,0)
    # Vein = blue (0,0,255)
    colored[av_mask == 1] = [255, 0, 0]  # Arteries in red
    colored[av_mask == 2] = [0, 0, 255]  # Veins in blue
    
    return colored

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image."""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
        
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Load image
        image = Image.open(file.stream)
        
        # Run prediction
        results = predict_image(image)
        
        # Convert images to base64
        vessel_img = numpy_to_pil(results['vessel'])
        centerline_img = numpy_to_pil(results['centerline'])
        av_colored = create_av_visualization(results['av'])
        av_img = Image.fromarray(av_colored, mode='RGB')
        
        # Optionally upload to Cloudinary (for persistence)
        use_cloudinary = request.form.get('use_cloudinary', 'false') == 'true'
        
        if use_cloudinary:
            # Upload results to Cloudinary
            vessel_url = cloudinary.uploader.upload(
                BytesIO(vessel_img.tobytes()),
                folder="retina/vessel"
            )['secure_url']
            
            centerline_url = cloudinary.uploader.upload(
                BytesIO(centerline_img.tobytes()),
                folder="retina/centerline"
            )['secure_url']
            
            av_url = cloudinary.uploader.upload(
                BytesIO(av_img.tobytes()),
                folder="retina/av"
            )['secure_url']
            
            response = {
                'success': True,
                'vessel_image': vessel_url,
                'centerline_image': centerline_url,
                'av_image': av_url,
                'metrics': results['metrics']
            }
        else:
            # Return as base64
            response = {
                'success': True,
                'vessel_image': pil_to_base64(vessel_img),
                'centerline_image': pil_to_base64(centerline_img),
                'av_image': pil_to_base64(av_img),
                'metrics': results['metrics']
            }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error in /api/analyze: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': device
    })

if __name__ == '__main__':
    # Load model at startup
    print("Loading model...")
    try:
        load_model_once()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
    print("Starting Flask server...")
    
    # Get port from environment (for cloud platforms) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'development') != 'production'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
