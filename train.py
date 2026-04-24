from ultralytics import YOLO
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load a model
model = YOLO('yolov8m.pt')  # load a pretrained model

# Train the model
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    device=0,  # Use GPU device 0 (first GPU)
    project='runs/detect',
    name='aqua_detection',
    exist_ok=False,
    pretrained=True,
    optimizer='SGD',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    fl_gamma=0.0,
    label_smoothing=0.0,
    nbs=64,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    seed=0,
    close_mosaic=10,
    fraction=1.0,
)
