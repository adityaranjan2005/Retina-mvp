"""
Gradio app for Hugging Face Spaces deployment.
Optimized for large model inference.
"""
import gradio as gr
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
import cv2

from src.model import MultiHeadRetinaModel
from src.metrics import compute_skeleton_metrics, postprocess_vessel_mask, extract_centerline_from_vessel
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
HF_REPO = "adityaranjan2005/retina-vessel-segmentation"
MODEL_FILENAME = "mvp_model.pt"
IMG_SIZE = 512

# Global model
model = None
device = None

def load_model():
    """Load model from Hugging Face Hub."""
    global model, device
    
    print("📥 Downloading model from Hugging Face Hub...")
    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=MODEL_FILENAME,
        cache_dir="."
    )
    print(f"✅ Model downloaded: {model_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Loading model on device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = MultiHeadRetinaModel(
        encoder_name="resnet34",
        encoder_weights=None
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    del checkpoint
    print("✅ Model loaded successfully!")
    return model

def get_inference_transform(img_size: int) -> A.Compose:
    """Get inference transform."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def create_av_visualization(av_mask: np.ndarray) -> np.ndarray:
    """Create colorized A/V visualization."""
    h, w = av_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[av_mask == 1] = [255, 0, 0]  # Red for arteries
    colored[av_mask == 2] = [0, 0, 255]  # Blue for veins
    return colored

def analyze_retinal_image(image: np.ndarray):
    """Analyze retinal fundus image."""
    if image is None:
        return None, None, None, "Please upload an image"
    
    # Convert to PIL
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image
    
    image_rgb = image_pil.convert('RGB')
    image_np = np.array(image_rgb)
    original_size = image_np.shape[:2]
    
    # Transform
    transform = get_inference_transform(IMG_SIZE)
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    vessel_pred = torch.sigmoid(outputs['vessel']).cpu().numpy()[0, 0]
    centerline_pred = torch.sigmoid(outputs['centerline']).cpu().numpy()[0, 0]
    av_pred = torch.softmax(outputs['av'], dim=1).cpu().numpy()[0]
    av_pred = np.argmax(av_pred, axis=0)
    
    # Resize to original
    from scipy.ndimage import zoom
    h, w = original_size
    vessel_pred = zoom(vessel_pred, (h / vessel_pred.shape[0], w / vessel_pred.shape[1]), order=1)
    centerline_pred = zoom(centerline_pred, (h / centerline_pred.shape[0], w / centerline_pred.shape[1]), order=1)
    av_pred = zoom(av_pred, (h / av_pred.shape[0], w / av_pred.shape[1]), order=0)
    
    # Binarize
    vessel_binary = (vessel_pred > 0.5).astype(np.uint8)
    centerline_binary = (centerline_pred > 0.5).astype(np.uint8)
    
    # Post-process
    vessel_binary = postprocess_vessel_mask(vessel_binary, kernel_size=3) // 255
    centerline_binary = extract_centerline_from_vessel(vessel_binary)
    
    # Compute metrics
    metrics = compute_skeleton_metrics(centerline_binary)
    
    # Create visualizations
    vessel_vis = (vessel_binary * 255).astype(np.uint8)
    vessel_vis = cv2.cvtColor(vessel_vis, cv2.COLOR_GRAY2RGB)
    
    centerline_vis = (centerline_binary * 255).astype(np.uint8)
    centerline_vis = cv2.cvtColor(centerline_vis, cv2.COLOR_GRAY2RGB)
    
    av_vis = create_av_visualization(av_pred.astype(np.uint8))
    
    # Format metrics
    metrics_text = f"""
    📏 **Centerline Length**: {metrics['centerline_length_px']:.2f} pixels
    
    🔀 **Branch Points**: {metrics['branch_points']}
    
    🔚 **Endpoints**: {metrics['endpoints']}
    
    📐 **Tortuosity** (mean): {metrics['tortuosity_proxy_mean']:.4f}
    📐 **Tortuosity** (max): {metrics['tortuosity_proxy_max']:.4f}
    
    ---
    *Higher tortuosity values indicate more curved/tortuous vessels*
    """
    
    return vessel_vis, centerline_vis, av_vis, metrics_text

# Load model at startup
print("🚀 Initializing Retinal Vessel Analysis System...")
load_model()

# Create Gradio interface
with gr.Blocks(title="Retinal Vessel Analysis", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("""
    # 🩺 Retinal Vessel Analysis System
    
    AI-powered segmentation of retinal blood vessels with artery/vein classification.
    
    **Upload a retinal fundus image** to get:
    - Vessel segmentation
    - Centerline extraction
    - Artery/Vein classification
    - Quantitative metrics
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Retinal Image", type="numpy")
            analyze_btn = gr.Button("🔬 Analyze Image", variant="primary", size="lg")
        
        with gr.Column():
            metrics_output = gr.Markdown(label="Metrics")
    
    with gr.Row():
        vessel_output = gr.Image(label="Vessel Segmentation")
        centerline_output = gr.Image(label="Centerline Extraction")
        av_output = gr.Image(label="Artery/Vein Classification")
    
    gr.Markdown("""
    ---
    ### About
    
    This system uses a multi-head U-Net architecture with ResNet34 encoder to perform:
    - **Vessel Segmentation**: Binary mask of all blood vessels
    - **Centerline Extraction**: Skeletonized vessel structure for analysis
    - **A/V Classification**: Distinguishes arteries (red) from veins (blue)
    
    **Metrics Explained**:
    - **Centerline Length**: Total length of vessel network in pixels
    - **Branch Points**: Junctions where vessels split (≥3 neighbors)
    - **Endpoints**: Terminal points of vessels (1 neighbor)
    - **Tortuosity**: Vessel curvature measure (higher = more tortuous)
    
    Clinical relevance: Vessel tortuosity changes are associated with diabetic retinopathy, hypertension, and other vascular diseases.
    
    ---
    **Model**: Hosted on [Hugging Face Hub](https://huggingface.co/adityaranjan2005/retina-vessel-segmentation)
    
    **Source Code**: [GitHub Repository](https://github.com/YOUR_USERNAME/retina-vessel-analysis)
    """)
    
    # Connect button
    analyze_btn.click(
        fn=analyze_retinal_image,
        inputs=[input_image],
        outputs=[vessel_output, centerline_output, av_output, metrics_output]
    )
    
    # Example images (optional)
    gr.Examples(
        examples=[
            # Add paths to example images if available
        ],
        inputs=input_image,
        label="Example Images"
    )

if __name__ == "__main__":
    demo.launch()
