from ultralytics import YOLO
import cv2
import os

def run_inference(model_path, source, conf=0.5):
    """
    Run inference on image or video
    
    Args:
        model_path: Path to trained model
        source: Path to image or video file
        conf: Confidence threshold
    """
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        project='runs/detect',
        name='inference'
    )
    
    return results

def detect_on_video(model_path, video_path, conf=0.5, save_output=True):
    """
    Run detection on video file
    
    Args:
        model_path: Path to trained model
        video_path: Path to video file
        conf: Confidence threshold
        save_output: Whether to save output video
    """
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if save_output:
        out = cv2.VideoWriter(
            'output_video.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(source=frame, conf=conf, verbose=False)
        annotated_frame = results[0].plot()
        
        if save_output:
            out.write(annotated_frame)
        
        cv2.imshow('Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    model_path = "models/aqua_detection.pt"
    
    # For image
    # run_inference(model_path, "path/to/image.jpg")
    
    # For video
    # detect_on_video(model_path, "path/to/video.mp4")
