---
title: Retinal Vessel Analysis
emoji: 🩺
colorFrom: black
colorTo: gray
sdk: gradio
sdk_version: 5.12.0
app_file: app_gradio.py
pinned: false
license: mit
models:
  - adityaranjan2005/retina-vessel-segmentation
tags:
  - medical
  - computer-vision
  - retinal-imaging
  - vessel-segmentation
  - biomedical
---

# Retinal Vessel Analysis System

AI-powered retinal vessel segmentation with artery/vein classification and quantitative metrics.

## Features

- **Vessel Segmentation**: Binary mask of retinal blood vessels
- **Centerline Extraction**: Skeletonized vessel structure
- **Artery/Vein Classification**: 3-class segmentation (background/artery/vein)
- **Quantitative Metrics**: Length, branch points, endpoints, tortuosity

## Model

Multi-head U-Net architecture with ResNet34 encoder, trained on retinal fundus images.

Model weights: [adityaranjan2005/retina-vessel-segmentation](https://huggingface.co/adityaranjan2005/retina-vessel-segmentation)

## Usage

1. Upload a retinal fundus image
2. Click "Analyze Image"
3. View segmentation results and quantitative metrics

## Clinical Applications

- Diabetic retinopathy screening
- Hypertension assessment
- Vascular disease monitoring
- Research tool for retinal analysis

## Technical Details

- **Input**: RGB retinal fundus images
- **Output**: Vessel masks, centerline skeletons, A/V classification
- **Metrics**: Centerline length, branch point detection, tortuosity analysis
- **Post-processing**: Morphological closing, skeletonization

## License

MIT License - Free for research and commercial use.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{retina_vessel_analysis_2024,
  author = {Your Name},
  title = {Retinal Vessel Analysis System},
  year = {2024},
  url = {https://huggingface.co/spaces/YOUR_USERNAME/retinal-vessel-analysis}
}
```
