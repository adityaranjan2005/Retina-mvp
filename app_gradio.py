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
    ### AI-Powered Retinal Image Segmentation and Classification
    
    ---
    
    ## 🎯 Aim
    
    This system aims to provide automated, accurate, and accessible retinal vessel analysis for:
    - **Early disease detection**: Diabetic retinopathy, hypertension, cardiovascular disease
    - **Clinical decision support**: Assisting ophthalmologists in diagnosis
    - **Research applications**: Quantitative analysis of retinal vasculature
    - **Accessibility**: Making advanced AI analysis available to everyone, free of cost
    
    ---
    
    ## 📸 Image Requirements
    
    **Supported Image Types:**
    - **Retinal fundus photographs** (color images of the back of the eye)
    - **Formats**: JPG, PNG, PPM, TIFF
    - **Resolution**: Any size (automatically resized to 512x512 for processing)
    - **Quality**: Clear, well-lit images with visible blood vessels
    
    **Best Results With:**
    - High-contrast retinal images
    - Centered optic disc (optional)
    - Minimal glare or artifacts
    - Standard fundus camera images
    
    **Not Suitable For:**
    - OCT (Optical Coherence Tomography) scans
    - Fluorescein angiography images
    - Extremely low-quality or blurry images
    
    ---
    
    ## 📋 Procedure
    
    **Step 1**: Upload your retinal fundus image using the upload box below
    
    **Step 2**: Click the "🔬 Analyze Image" button
    
    **Step 3**: Wait 10-30 seconds while the AI processes your image
    
    **Step 4**: Review the results:
    - **Vessel Segmentation**: Binary mask showing all detected blood vessels
    - **Centerline Extraction**: Skeletonized vessel structure for precise measurement
    - **Artery/Vein Classification**: Color-coded map (Red = Arteries, Blue = Veins)
    - **Quantitative Metrics**: Numerical measurements of vessel characteristics
    
    **Step 5**: Analyze the metrics for clinical insights
    
    ---
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
    
    ## 🔬 How It Works
    
    ### Technical Architecture
    
    **1. Deep Learning Model:**
    - **Architecture**: Multi-head U-Net with ResNet34 encoder
    - **Pre-training**: ImageNet weights for transfer learning
    - **Training**: Supervised learning on annotated retinal fundus images
    - **Model Size**: 800MB+ with 21.3M parameters
    
    **2. Three Specialized Heads:**
    - **Vessel Head**: Binary segmentation (vessel vs background)
    - **Centerline Head**: Skeletonization for structural analysis
    - **A/V Head**: 3-class classification (background/artery/vein)
    
    **3. Post-Processing:**
    - Morphological closing to connect vessel fragments
    - Skeletonization for centerline extraction
    - Connected component analysis
    - Topological feature extraction
    
    **4. Metrics Computation:**
    - **Centerline Length**: Pixel-by-pixel measurement of vessel paths
    - **Branch Points**: Detecting junctions where vessels split (≥3 neighbors)
    - **Endpoints**: Identifying vessel terminals (1 neighbor)
    - **Tortuosity**: Calculating path length / euclidean distance ratio
    
    ### Processing Pipeline
    
    ```
    Input Image → Preprocessing → Multi-head U-Net → Post-processing → Metrics Extraction → Results
    ```
    
    ---
    
    ## 📊 Results and Interpretation
    
    ### Vessel Segmentation
    - **White regions**: Detected blood vessels
    - **Black regions**: Background (non-vessel tissue)
    - **Use case**: Overall vessel density, coverage area
    
    ### Centerline Extraction
    - **White lines**: Vessel centerlines (skeleton)
    - **Use case**: Precise length measurement, topological analysis
    - **Clinical value**: Vessel network connectivity, structural changes
    
    ### Artery/Vein Classification
    - **Red regions**: Arteries (oxygenated blood from heart)
    - **Blue regions**: Veins (deoxygenated blood to heart)
    - **Black regions**: Background
    - **Clinical value**: A/V ratio (abnormal in hypertension, diabetic retinopathy)
    
    ### Quantitative Metrics
    
    **Centerline Length (pixels):**
    - Total length of vessel network
    - Higher values = more extensive vasculature
    
    **Branch Points:**
    - Number of vessel bifurcations
    - Increased in neovascularization (diabetic retinopathy)
    
    **Endpoints:**
    - Terminal points of vessel segments
    - May indicate vessel occlusion or incomplete segmentation
    
    **Tortuosity (mean & max):**
    - Ratio of actual path length to straight-line distance
    - Values > 1.1: Mild tortuosity
    - Values > 1.3: Moderate tortuosity
    - Values > 1.5: Severe tortuosity
    - **Clinical significance**: Associated with diabetic retinopathy, hypertension, aging
    
    ---
    
    ## 🎓 Clinical Applications
    
    1. **Diabetic Retinopathy Screening**: Detecting early vascular changes
    2. **Hypertension Assessment**: Measuring vessel narrowing and tortuosity
    3. **Cardiovascular Risk**: Retinal vessels reflect systemic vascular health
    4. **Stroke Prediction**: Abnormal vessel patterns correlate with stroke risk
    5. **Research Studies**: Quantitative data for longitudinal studies
    
    ---
    
    ## 🙏 Credits and Acknowledgments
    
    ### Datasets
    - **Kaggle**: Open-source retinal image datasets for training and validation
    - **Crimson (University)**: Annotated retinal vessel datasets with A/V labels
    - **DRIVE Dataset**: Digital Retinal Images for Vessel Extraction
    
    ### Infrastructure
    - **Hugging Face**: Free GPU/CPU resources and model hosting (881MB model storage)
    - **Hugging Face Spaces**: Free deployment platform for ML applications
    - **PyTorch & Open Source**: Deep learning framework and libraries
    
    ### Model Repository
    - **Hugging Face Hub**: [adityaranjan2005/retina-vessel-segmentation](https://huggingface.co/adityaranjan2005/retina-vessel-segmentation)
    - **Source Code**: [GitHub Repository](https://github.com/adityaranjan2005/Retina-mvp)
    
    ---
    
    ## 👨‍💻 About the Developer
    
    **Aditya Ranjan**
    - Entrepreneur & Social Worker
    - Medical AI Researcher (free time dedication)
    - *"Spending free time in medical research to make life better for all."*
    
    ### Connect with Aditya
    - **X (Twitter)**: [@em_Adii](https://x.com/em_Adii)
    - **LinkedIn**: [linkedin.com/in/adityaranjan2005](https://linkedin.com/in/adityaranjan2005)
    - **GitHub**: [@adityaranjan2005](https://github.com/adityaranjan2005)
    - **Hugging Face**: [@adityaranjan2005](https://huggingface.co/adityaranjan2005)
    
    ---
    
    ## ⚠️ Disclaimer
    
    This system is designed for **research and educational purposes only**. It should **NOT** be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice.
    
    ### Limitations
    - Performance varies with image quality
    - Trained on specific datasets (may not generalize to all populations)
    - A/V classification accuracy depends on image contrast
    - Not FDA-approved or clinically validated
    
    ---
    
    ## 📜 License
    
    **MIT License** - Free for research and educational use
    
    This project is open-source and available for:
    - Academic research
    - Educational purposes
    - Non-commercial applications
    - Derivative works (with attribution)
    
    ---
    
    ## 🌟 Future Enhancements
    
    - [ ] Vessel width measurement
    - [ ] Optic disc and cup segmentation
    - [ ] Diabetic retinopathy grading (mild/moderate/severe)
    - [ ] Batch processing for multiple images
    - [ ] PDF report generation
    - [ ] Integration with DICOM medical imaging standards
    
    ---
    
    ### 💡 Support This Project
    
    If you find this tool useful:
    - ⭐ Star the [GitHub repository](https://github.com/adityaranjan2005/Retina-mvp)
    - 🤝 Share with researchers and medical professionals
    - 💬 Provide feedback for improvements
    - 🔬 Contribute to the codebase
    
    **Together, we can make medical AI accessible to everyone!**
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
    demo.launch(ssr_mode=False, server_name="0.0.0.0")
