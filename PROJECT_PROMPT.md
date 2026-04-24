# Aqua Detection System - Project Prompt & System Instructions

## Project Overview

**Project Name:** Real-Time Aquatic Debris Detection System  
**Domain:** Computer Vision, Environmental Monitoring, Marine Science  
**Primary Technology Stack:** Python, YOLOv8, PyTorch, Streamlit  
**Purpose:** Automated detection and classification of 24 distinct aquatic object classes for real-time marine environment surveillance and environmental monitoring

---

## Executive Summary

The Aqua Detection System is a production-ready deep learning application that leverages YOLOv8 for real-time detection of aquatic debris, diving equipment, and anthropogenic waste in underwater environments. The system is designed for environmental scientists, marine researchers, and infrastructure maintenance teams to automate debris tracking, environmental assessment, and underwater infrastructure monitoring.

**Core Objectives:**
1. Detect and classify 24 aquatic object categories with high accuracy
2. Provide real-time inference capabilities (≤13 FPS target for video processing)
3. Offer an intuitive user interface for non-ML specialists
4. Enable reproducible model training and evaluation
5. Support batch processing for large-scale environmental surveys

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                  │
│                  (app.py - Main Application)                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Input Layer (images/video/webcam)           │   │
│  └────────────────┬─────────────────────────────────────┘   │
│                   │                                           │
│  ┌────────────────▼─────────────────────────────────────┐   │
│  │      Inference Engine (PyTorch + YOLOv8m)           │   │
│  │  • Preprocessing (normalization, resizing)          │   │
│  │  • Forward pass through YOLOv8m backbone            │   │
│  │  • Post-processing (NMS, confidence filtering)      │   │
│  └────────────────┬─────────────────────────────────────┘   │
│                   │                                           │
│  ┌────────────────▼─────────────────────────────────────┐   │
│  │     Analytics & Visualization Layer                 │   │
│  │  • Bounding box rendering                           │   │
│  │  • Confidence distribution analysis                 │   │
│  │  • Detection count statistics                       │   │
│  │  • Performance metrics (FPS, inference time)        │   │
│  └────────────────┬─────────────────────────────────────┘   │
│                   │                                           │
│  ┌────────────────▼─────────────────────────────────────┐   │
│  │        Output Layer (web UI, downloads)             │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘

Backend Support (Training & Model Management):
├── train.py (Training orchestration)
├── inference.py (Low-level inference utilities)
└── Model Zoo (Multiple YOLOv8 variants for A/B testing)
```

### Key Files & Their Responsibilities

| File | Purpose | Key Functions |
|------|---------|---------------|
| `app.py` | Main Streamlit web application | Interactive UI, real-time inference, visualization |
| `train.py` | Model training pipeline | Dataset loading, training loops, checkpoint management |
| `inference.py` | Inference utilities | Model loading, batch processing, result formatting |
| `YOLO/dataset.yaml` | Dataset configuration | Class definitions, dataset paths for training |
| `YOLO/obj.names` | Class name mapping | 24 object class identifiers |

---

## Detected Object Classes (24 Categories)

### Class Taxonomy

**Recreational Diving Equipment (7):**
- Goggles, Flipper, Snorkel, Dive Weight, Lure, ROV

**Dining/Kitchen Items (5):**
- Plastic Cup, Fork, Spoon, Knife, Soda Can

**Tools & Hardware (5):**
- Scissors, Metal Rod, Pipe, Screwdriver, Case

**Containers & Waste (4):**
- Bottle, Plastic Bag, Cup, Aqua

**Miscellaneous (3):**
- Car, Tripod, LoCo

---

## Model Specifications

### YOLOv8m Architecture Details

**Architecture Components:**
- **Backbone:** CSPDarknet with residual connections (C3 modules)
- **Neck:** PANet (Path Aggregation Network) for multi-scale feature fusion
- **Head:** Decoupled objectness and classification branches
- **Total Parameters:** 25.9M
- **Input Resolution:** 640×640 pixels (with letter-box augmentation)

**Performance Characteristics:**
- **GPU Inference:** ~78 ms per image (NVIDIA RTX 3060+)
- **CPU Inference:** ~650 ms per image (Intel i7 equivalent)
- **Model Size:** ~51 MB (FP32), ~26 MB (INT8 quantized)
- **Expected Accuracy:** mAP ≈ 0.60–0.68 on aquatic dataset

**Available Model Variants:**

| Variant | Size | Speed | Accuracy | Use Case |
|---------|------|-------|----------|----------|
| YOLOv8n | 3.2M | ⚡⚡⚡ | ⭐⭐ | Mobile/Edge |
| YOLOv8s | 11.2M | ⚡⚡ | ⭐⭐⭐ | Embedded systems |
| **YOLOv8m** | **25.9M** | **⚡** | **⭐⭐⭐⭐** | **[DEFAULT]** |
| YOLOv8l | 43.7M | ◐ | ⭐⭐⭐⭐⭐ | Maximum accuracy |

---

## Training Pipeline

### Dataset Structure

```
YOLO/
├── dataset.yaml              # Configuration file
├── obj.names                 # Class labels
└── images/
    ├── train/                # Training images (~70% of data)
    ├── val/                  # Validation images (~15% of data)
    └── test/                 # Test images (~15% of data, e.g., Duluth_*.jpg)
└── labels/
    ├── train/                # YOLO-format annotations
    ├── val/
    └── test/                 # e.g., Duluth_1011.txt, Duluth_1014.txt
```

### Training Configuration (Hyperparameters)

```python
# Core Training Parameters
EPOCHS = 100                  # Training iterations
BATCH_SIZE = 16              # Samples per gradient update
IMAGE_SIZE = 640             # Input resolution

# Optimization
OPTIMIZER = 'SGD'            # Stochastic Gradient Descent
MOMENTUM = 0.937             # SGD momentum
LR_INIT = 0.01               # Initial learning rate
SCHEDULER = 'cosine'         # Cosine annealing schedule

# Regularization
EARLY_STOPPING_PATIENCE = 20 # Epochs without improvement before stopping
WEIGHT_DECAY = 0.0005        # L2 regularization coefficient

# Data Augmentation
MOSAIC_PROB = 1.0            # Probability of 4-image mix
FLIP_PROB = 0.5              # Random flip probability
HSV_H_DELTA = 0.015          # Hue augmentation range
HSV_S_DELTA = 0.7            # Saturation augmentation range
HSV_V_DELTA = 0.4            # Value augmentation range
ROTATION_RANGE = 10          # Rotation degrees (±)
TRANSLATE_RANGE = 0.1        # Translation as fraction of image size
SCALE_RANGE = 0.5            # Scaling range (0.5 to 1.5)
```

### Loss Function

YOLOv8 employs a combined loss function:

$$L_{total} = L_{localization} + L_{objectness} + L_{classification}$$

Where:
- $L_{localization}$: CIoU loss for bounding box regression
- $L_{objectness}$: Binary cross-entropy for objectness score
- $L_{classification}$: Cross-entropy for multi-class classification

### Training Workflow

1. **Data Loading:** Load YOLO-format annotations and images
2. **Augmentation:** Apply real-time data augmentation pipeline
3. **Forward Pass:** Image → YOLOv8m backbone → predictions
4. **Loss Computation:** Compare predictions to ground truth
5. **Backpropagation:** Update weights via SGD optimizer
6. **Validation:** Evaluate on validation set every epoch
7. **Checkpoint:** Save best model (based on validation mAP)
8. **Post-Training:** Generate results.csv with metrics and training plots

---

## Inference Engine Specifications

### Preprocessing Pipeline

```
Raw Image Input
    ↓
[1] BGR to RGB conversion (OpenCV default)
    ↓
[2] Resize to 640×640 with letter-box padding
    ↓
[3] Normalize to [0, 1] range
    ↓
[4] Transpose to CHW format (3, 640, 640)
    ↓
[5] Stack to batch dimension if needed
```

### Post-Processing Pipeline

```
Raw Model Predictions (bounding boxes, confidence, class logits)
    ↓
[1] Confidence Filtering: τ_conf = user_defined_threshold (default: 0.5)
    ├─ Remove predictions with confidence < τ_conf
    ↓
[2] Non-Maximum Suppression (NMS)
    ├─ Calculate pairwise IoU between all predictions
    ├─ Remove duplicate detections with IoU > τ_iou
    ├─ τ_iou = user_defined_threshold (default: 0.45)
    ↓
[3] Coordinate Denormalization
    ├─ Convert normalized [0, 1] coordinates back to pixel space
    ├─ Account for letter-box padding offset
    ↓
[4] Format Output
    ├─ Class label assignment
    ├─ Confidence score formatting
    ├─ Bounding box coordinates (x_min, y_min, x_max, y_max) in pixels
```

### Configurable Parameters (User Interface)

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Confidence Threshold** | [0.0, 1.0] | 0.5 | Minimum confidence for detection display |
| **IOU Threshold** | [0.0, 1.0] | 0.45 | NMS threshold for duplicate removal |
| **Model Selection** | {yolov8n, yolov8s, yolov8m, yolov8l} | yolov8m | Model variant for inference |
| **Input Modality** | {Image, Video, Webcam} | Image | Detection source type |

### Confidence Classification Scheme

Detection results are color-coded based on confidence:

```
┌─────────────────────────────────────────────┐
│  🟢 HIGH CONFIDENCE (≥ 0.70)               │
│     Green bounding box                      │
│     → Suitable for automated decisions      │
│     → High probability of correct detection │
├─────────────────────────────────────────────┤
│  🟠 MEDIUM CONFIDENCE (0.40 – 0.70)        │
│     Orange bounding box                     │
│     → Requires human review                 │
│     → Moderate confidence level             │
├─────────────────────────────────────────────┤
│  🔴 LOW CONFIDENCE (< 0.40)                │
│     Red bounding box                        │
│     → Manual verification recommended       │
│     → Low probability of correct detection  │
└─────────────────────────────────────────────┘
```

---

## Application Interface (Streamlit)

### Dashboard Layout

**Sidebar (Configuration Panel):**
- Input type selector (Image / Video / Webcam)
- Confidence threshold slider
- IOU threshold slider
- Model selection dropdown
- Performance monitoring

**Main Canvas:**
- Input display (original image/video frame)
- Detection results overlay
- Bounding boxes with class labels and confidence scores
- Real-time statistics panel

### Detection Modes

#### 1. Image Mode
- Single image upload or batch processing
- Shows: Original + Annotated side-by-side
- Outputs: Detection table with class, confidence, coordinates
- Exports: Annotated images (PNG/JPG)

#### 2. Video Mode
- MP4, AVI, MOV file upload
- Frame-by-frame processing with configurable interval
- Outputs: Annotated video file with overlays
- Metrics: Average detections/frame, total FPS

#### 3. Webcam Mode
- Real-time camera feed capture
- Toggle to start/stop streaming
- Live performance monitoring (current FPS)
- Optional frame snapshot capture

---

## Data Processing & Analytics

### Detection Output Format

```json
{
  "image_id": "duluth_1011",
  "detections": [
    {
      "class_id": 5,
      "class_name": "Bottle",
      "confidence": 0.87,
      "bounding_box": {
        "x_min": 150,
        "y_min": 200,
        "x_max": 300,
        "y_max": 450
      },
      "color": "green"
    },
    {
      "class_id": 8,
      "class_name": "Plastic_Bag",
      "confidence": 0.62,
      "bounding_box": {
        "x_min": 400,
        "y_min": 100,
        "x_max": 550,
        "y_max": 350
      },
      "color": "orange"
    }
  ],
  "total_detections": 2,
  "inference_time_ms": 78,
  "fps": 12.8
}
```

### Analytics Computed

- **Detection Distribution:** Bar chart of object counts by class
- **Confidence Statistics:** Mean, std dev, histogram of confidence scores
- **Class-Specific Metrics:** Per-class precision, recall, F1-score
- **Temporal Analysis:** Detection trends across video frames
- **Performance Metrics:** FPS, inference latency, batch throughput

---

## System Requirements & Deployment

### Minimum Specifications

```
┌──────────────────────────────────────────┐
│        MINIMUM REQUIREMENTS               │
├──────────────────────────────────────────┤
│ Operating System: Windows 10+, macOS 10+  │
│ Python Version: 3.8+                      │
│ RAM: 4 GB                                 │
│ Storage: 2 GB (models + dependencies)     │
│ Processor: Intel i5 / AMD Ryzen 5         │
│ GPU: Optional (CPU mode supported)        │
└──────────────────────────────────────────┘
```

### Recommended Specifications

```
┌──────────────────────────────────────────┐
│      RECOMMENDED FOR PRODUCTION           │
├──────────────────────────────────────────┤
│ Operating System: Windows 11, Ubuntu 22+  │
│ Python Version: 3.10+                     │
│ RAM: 8 GB - 16 GB                         │
│ Storage: 8 GB SSD                         │
│ Processor: Intel i7-12700+ / Ryzen 7 5800 │
│ GPU: NVIDIA RTX 3060+ (8GB VRAM min)     │
│ CUDA: 11.8+ (for GPU acceleration)        │
│ cuDNN: 8.0+                               │
└──────────────────────────────────────────┘
```

### GPU Acceleration

For NVIDIA GPUs, install CUDA-enabled PyTorch:

```bash
# CUDA 11.8 support
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

**GPU vs CPU Performance:**
- YOLOv8m on NVIDIA RTX 3060: ~78 ms/image (12.8 FPS)
- YOLOv8m on intel i7 (CPU): ~650 ms/image (1.5 FPS)
- **Speedup Factor:** ~8–10x faster with GPU

---

## Development Workflow

### Environment Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "from ultralytics import YOLO; import streamlit; print('✓ Ready')"
```

### Running the Application

```bash
# Development mode (auto-reload on code changes)
streamlit run app.py

# With custom port
streamlit run app.py --server.port 8502

# Production mode (no cache, optimized)
streamlit run app.py --logger.level=error
```

### Training a Custom Model

```bash
# Standard training
python train.py

# With custom hyperparameters (edit train.py)
python train.py --epochs 150 --batch_size 32 --device 0
```

---

## File Organization Best Practices

### Naming Conventions

**Image Files:**
- Format: `{location}_{sequence_number}.jpg`
- Example: `duluth_1011.jpg`, `duluth_1014.jpg`

**Annotation Files:**
- Format: `{location}_{sequence_number}.txt`
- Content: YOLO format (normalized coordinates)

**Training Runs:**
- Directory: `runs/detect/train{N}/`
- Checkpoints: `weights/best.pt`, `weights/last.pt`
- Logs: `results.csv`, `training_plots.png`

### Dataset Splits

```
Recommended split ratio: 70% Train / 15% Val / 15% Test

YOLO/
├── images/
│   ├── train/   → 350 images (if 500 total)
│   ├── val/     → 75 images
│   └── test/    → 75 images (Duluth samples)
└── labels/
    ├── train/   → 350 .txt files
    ├── val/     → 75 .txt files
    └── test/    → 75 .txt files
```

---

## Troubleshooting Guide

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| ModuleNotFoundError | Missing dependencies | `pip install -r requirements.txt --upgrade` |
| CUDA out of memory | Batch size too large | Reduce batch size or use smaller model (YOLOv8n) |
| Slow inference | CPU-only mode | Install CUDA-enabled PyTorch or use GPU |
| Webcam not detected | Permission denied | Grant camera permissions in system settings |
| Streamlit not responding | Port already in use | Use `streamlit run app.py --server.port 8502` |
| Low detection accuracy | Poor training data | Increase dataset size or augmentation |

---

## Performance Optimization Tips

1. **GPU Utilization:** Enable CUDA for 8–10x speedup
2. **Batch Processing:** Process multiple images together for higher throughput
3. **Model Selection:** Use YOLOv8n for edge devices, YOLOv8l only if accuracy critical
4. **Input Resolution:** Reduce from 640×640 to 416×416 for 2x speedup (slight accuracy loss)
5. **Precision:** Use INT8 quantization for 4x inference speedup (requires retraining)

---

## Contributing & Extension Points

### Adding New Object Classes

1. Update `YOLO/obj.names` with new class names
2. Retrain model with annotated dataset containing new classes
3. Update class definition table in documentation

### Custom Inference Pipeline

Modify `inference.py` to:
- Add custom preprocessing (color correction, histogram equalization)
- Implement temporal smoothing for video detections
- Add multi-model ensemble for higher accuracy

### UI Enhancements

Extend `app.py` to:
- Add object tracking across frames
- Implement GIS integration for geographic tagging
- Add data export to databases (SQL, MongoDB)
- Build REST API endpoints for headless inference

---

## References & Resources

- **YOLOv8 Documentation:** https://docs.ultralytics.com/
- **Streamlit Documentation:** https://docs.streamlit.io/
- **PyTorch Documentation:** https://pytorch.org/docs/
- **OpenCV Documentation:** https://docs.opencv.org/
- **Dataset Annotation Tools:** Roboflow, CVAT, Label Studio

---

## License & Attribution

- **YOLOv8 License:** AGPL-3.0 (Ultralytics)
- **Project License:** [Specify your project license]
- **Commercial Use:** Requires separate license from Ultralytics for commercial YOLO deployment

---

## Contact & Support

For questions, issues, or contributions:
1. Review this comprehensive project prompt
2. Check troubleshooting section above
3. Consult official YOLOv8 documentation
4. Contact project team or repository maintainers

---

**Last Updated:** March 2026  
**Version:** 1.0  
**Status:** Production Ready
