import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import pandas as pd
import time
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Aqua Detection", page_icon="🐟", layout="wide")
st.title("🐟 Aqua Detection System")
st.markdown("Real-time object detection in aquatic environments using YOLOv8")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Configuration")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.1)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.3)
model_choice = st.sidebar.radio(
    "Select Model", ["YOLOv8n","YOLO11n","YOLO26n"]
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(model_name):
    model_map = {
        "YOLOv8n": "runs/detect/train3/weights/best.pt",
        "YOLO11n": "runs/detect/train5/weights/best.pt",
        "YOLO26n": "runs/detect/train6/weights/best.pt"
    }
    return YOLO(model_map[model_name])

try:
    model = load_model(model_choice)
    st.sidebar.success(f"✓ {model_choice} loaded successfully")
    st.sidebar.write("Model Classes:", model.names)
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# ---------------- SESSION STATE ----------------
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# ---------------- TABS ----------------
tabs = st.tabs(["Image", "Video", "Webcam", "Model Comparison", "Optimization", "Analytics"])

# =================================================
# IMAGE TAB
# =================================================
with tabs[0]:
    st.subheader("Upload and Detect Image")
    uploaded_files = st.file_uploader(
        "Choose image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            image = Image.open(file)
            st.session_state.uploaded_image = image
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", width=400)

            results = model.predict(
                source=image, conf=confidence, iou=iou_threshold, verbose=False
            )
            result = results[0]
            annotated = result.plot()

            with col2:
                st.image(annotated, caption="Detected Image", width=400)

            if len(result.boxes) > 0:
                df = pd.DataFrame({
                    "Class": [
                        model.names[int(cls)]
                        for cls in result.boxes.cls.cpu().numpy()
                    ],
                    "Confidence": result.boxes.conf.cpu().numpy(),
                    "Xmin": result.boxes.xyxy[:, 0].cpu().numpy(),
                    "Ymin": result.boxes.xyxy[:, 1].cpu().numpy(),
                    "Xmax": result.boxes.xyxy[:, 2].cpu().numpy(),
                    "Ymax": result.boxes.xyxy[:, 3].cpu().numpy(),
                })
                
                df = df.sort_values("Confidence", ascending=False)
                df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.2%}")

                st.markdown("### 📊 Detection Results")
                col_count, col_avg = st.columns(2)
                with col_count:
                    st.metric("Total Objects Detected", len(df))
                with col_avg:
                    st.metric("Avg Confidence", f"{df['Confidence'].apply(lambda x: float(x.strip('%'))/100).mean():.2%}")
                
                st.dataframe(df, use_container_width=True)
                st.session_state.detection_history.append(df)

                # Object count chart
                class_counts = df["Class"].value_counts()
                col1, col2 = st.columns([1.5, 1])
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    class_counts.plot(kind="barh", ax=ax, color="steelblue")
                    ax.set_xlabel("Count")
                    ax.set_title("Object Count by Class")
                    st.pyplot(fig)
            else:
                st.warning("⚠️ No objects detected in this image")

# =================================================
# VIDEO TAB
# =================================================
with tabs[1]:
    st.subheader("Upload and Detect Video")

    uploaded_video = st.file_uploader(
        "Choose a video", type=["mp4", "avi", "mov", "mkv"], key="video"
    )

    if uploaded_video:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.video(uploaded_video)

        if st.button("▶ Start Video Detection"):
            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()
            progress = st.progress(0)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            detection_counts = []
            video_detected_classes = []
            start_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(
                    source=frame, conf=confidence, iou=iou_threshold, verbose=False
                )

                if len(results[0].boxes) > 0:
                    for cls in results[0].boxes.cls.cpu().numpy():
                        video_detected_classes.append(model.names[int(cls)])

                detection_counts.append(len(results[0].boxes))
                annotated = results[0].plot()

                stframe.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    width=600
                )

                frame_count += 1
                progress.progress(frame_count / total_frames)

            cap.release()
            os.remove(video_path)

            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            st.success("✅ Video processing completed")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Frames", frame_count)
            col2.metric(
                "Avg Detections / Frame",
                round(np.mean(detection_counts), 2)
            )
            col3.metric("FPS", round(fps, 2))

            unique_objects = set(video_detected_classes)
            if unique_objects:
                st.success(
                    f"✅ Object Detected: {', '.join(unique_objects)}"
                )
            else:
                st.warning("⚠️ No trained object detected in this video")

# =================================================
# WEBCAM TAB
# =================================================
with tabs[2]:
    st.subheader("Live Webcam Detection")
    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame, conf=confidence, iou=iou_threshold, verbose=False
            )
            annotated = results[0].plot()

            stframe.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                width=600
            )

        cap.release()

# =================================================
# MODEL COMPARISON TAB
# =================================================
with tabs[3]:
    st.subheader("🔬 Model Comparison for Research")
    st.markdown("Compare YOLOv8n, YOLO11n, and YOLO26n models on same image")
    
    uploaded_compare_file = st.file_uploader(
        "Upload image for comparison", type=["jpg", "jpeg", "png"], key="compare"
    )
    
    # Use uploaded image or let user upload new one
    if uploaded_compare_file:
        image = Image.open(uploaded_compare_file)
        st.session_state.uploaded_image = image
    elif st.session_state.uploaded_image is not None:
        st.info("ℹ️ Using image from Image tab")
        image = st.session_state.uploaded_image
    else:
        image = None
    
    if image is not None:
        st.image(image, caption="Test Image", width=400)
        
        if st.button("🚀 Compare All Models"):
            models_info = {
                "YOLOv8n": "runs/detect/train3/weights/best.pt",
                "YOLO11n": "runs/detect/train5/weights/best.pt",
                "YOLO26n": "runs/detect/train6/weights/best.pt"
            }
            
            comparison_results = []
            col1, col2, col3 = st.columns(3)
            
            for idx, (model_name, model_path) in enumerate(models_info.items()):
                with [col1, col2, col3][idx]:
                    st.markdown(f"### {model_name}")
                    
                    try:
                        start_time = time.time()
                        model_comp = YOLO(model_path)
                        
                        results = model_comp.predict(
                            source=image, conf=confidence, iou=iou_threshold, verbose=False
                        )
                        
                        inference_time = time.time() - start_time
                        result = results[0]
                        annotated = result.plot()
                        
                        st.image(annotated, width=300)
                        
                        st.metric("Objects Detected", len(result.boxes))
                        st.metric("Inference Time", f"{inference_time:.3f}s")
                        
                        if len(result.boxes) > 0:
                            avg_conf = result.boxes.conf.cpu().numpy().mean()
                            st.metric("Avg Confidence", f"{avg_conf:.2%}")
                        
                        # Store results for comparison
                        comparison_results.append({
                            "Model": model_name,
                            "Objects Detected": len(result.boxes),
                            "Inference Time (s)": round(inference_time, 4),
                            "Avg Confidence": round(result.boxes.conf.cpu().numpy().mean() if len(result.boxes) > 0 else 0, 4),
                            "Detections": result.boxes.data.cpu().numpy().tolist() if len(result.boxes) > 0 else []
                        })
                    
                    except Exception as e:
                        st.error(f"Error with {model_name}: {e}")
            
            # Comparison Table
            st.markdown("---")
            st.markdown("### 📊 Comparison Summary")
            
            comp_df = pd.DataFrame([{
                "Model": r["Model"],
                "Objects": r["Objects Detected"],
                "Inference Time (s)": r["Inference Time (s)"],
                "Avg Confidence": r["Avg Confidence"]
            } for r in comparison_results])
            
            st.dataframe(comp_df, use_container_width=True)
            
            # Visualization
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig, ax = plt.subplots(figsize=(5, 3))
                comp_df.set_index("Model")["Objects"].plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
                ax.set_title("Objects Detected Comparison")
                ax.set_ylabel("Count")
                st.pyplot(fig)
            
            with col_chart2:
                fig, ax = plt.subplots(figsize=(5, 3))
                comp_df.set_index("Model")["Inference Time (s)"].plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
                ax.set_title("Inference Speed Comparison")
                ax.set_ylabel("Time (seconds)")
                st.pyplot(fig)
            
            # Download CSV
            csv = comp_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison Results (CSV)",
                data=csv,
                file_name="model_comparison.csv",
                mime="text/csv"
            )

# =================================================
# OPTIMIZATION TAB
# =================================================
with tabs[4]:
    st.subheader("🔧 Model Optimization & Accuracy Tips")
    
    opt_tab1, opt_tab2, opt_tab3, opt_tab4 = st.tabs(
        ["Ensemble", "Hyperparameters", "Training Metrics", "Accuracy Tips"]
    )
    
    # -------- ENSEMBLE TAB --------
    with opt_tab1:
        st.markdown("### 🎯 Ensemble Detection (Vote-based)")
        st.info("Combines predictions from all 3 models for better accuracy")
        
        ensemble_file = st.file_uploader(
            "Upload image for ensemble detection", type=["jpg", "jpeg", "png"], key="ensemble"
        )
        
        # Use uploaded image or let user upload new one
        if ensemble_file:
            image = Image.open(ensemble_file)
            st.session_state.uploaded_image = image
        elif st.session_state.uploaded_image is not None:
            st.info("ℹ️ Using image from Image tab")
            image = st.session_state.uploaded_image
        else:
            image = None
        
        if image is not None:
            st.image(image, caption="Test Image", width=400)
            
            if st.button("🚀 Run Ensemble Detection"):
                models_info = {
                    "YOLOv8n": "runs/detect/train3/weights/best.pt",
                    "YOLO11n": "runs/detect/train5/weights/best.pt",
                    "YOLO26n": "runs/detect/train6/weights/best.pt"
                }
                
                all_detections = []
                
                for model_name, model_path in models_info.items():
                    model_ens = YOLO(model_path)
                    results = model_ens.predict(
                        source=image, conf=confidence, iou=iou_threshold, verbose=False
                    )
                    
                    if len(results[0].boxes) > 0:
                        for i, box in enumerate(results[0].boxes):
                            all_detections.append({
                                "Model": model_name,
                                "Class": model_ens.names[int(box.cls)],
                                "Confidence": float(box.conf),
                                "x1": float(box.xyxy[0][0]),
                                "y1": float(box.xyxy[0][1]),
                                "x2": float(box.xyxy[0][2]),
                                "y2": float(box.xyxy[0][3])
                            })
                
                if all_detections:
                    df_ensemble = pd.DataFrame(all_detections)
                    
                    # Vote-based consensus
                    st.markdown("#### Detection Consensus (Vote System)")
                    consensus = df_ensemble.groupby("Class").agg({
                        "Model": "count",
                        "Confidence": "mean"
                    }).rename(columns={"Model": "Votes", "Confidence": "Avg_Confidence"})
                    
                    consensus = consensus.sort_values("Votes", ascending=False)
                    consensus = consensus.reset_index()
                    
                    st.dataframe(consensus, use_container_width=True)
                    
                    st.markdown("**Interpretation:**")
                    st.write("- **3 Votes**: All models agree ✅ (High confidence)")
                    st.write("- **2 Votes**: Majority agree ⚠️ (Medium confidence)")  
                    st.write("- **1 Vote**: Single model detection ❌ (Low confidence)")
                    
                    # High confidence detections only
                    high_conf_classes = consensus[consensus["Votes"] >= 2]["Class"].tolist()
                    st.metric("High-Confidence Detections (2+ votes)", len(high_conf_classes))
                    
                    if high_conf_classes:
                        st.success(f"✅ Agreed detections: {', '.join(high_conf_classes)}")
                else:
                    st.warning("No objects detected by any model")
    
    # -------- HYPERPARAMETERS TAB --------
    with opt_tab2:
        st.markdown("### ⚙️ Tune Detection Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Current Settings")
            st.write(f"**Confidence Threshold:** {confidence}")
            st.write(f"**IOU Threshold:** {iou_threshold}")
            
            st.markdown("#### Recommendations")
            if confidence < 0.4:
                st.warning("⚠️ Low confidence threshold may cause false positives. Try 0.4-0.6")
            if confidence > 0.8:
                st.warning("⚠️ High confidence threshold may miss real objects. Try 0.4-0.6")
            
            if iou_threshold < 0.3:
                st.warning("⚠️ Low IOU may cause duplicate detections. Try 0.4-0.6")
        
        with col2:
            st.markdown("#### How to Optimize")
            st.write("""
            **Confidence Threshold:**
            - ↑ Higher = Fewer detections, fewer false positives
            - ↓ Lower = More detections, more false positives
            - **Best Range: 0.4 - 0.6**
            
            **IOU Threshold:**
            - ↑ Higher = Fewer overlapping boxes
            - ↓ Lower = More overlapping boxes allowed
            - **Best Range: 0.4 - 0.6**
            """)
    
    # -------- TRAINING METRICS TAB --------
    with opt_tab3:
        st.markdown("### 📈 Model Training Metrics")
        
        model_metrics = {
            "YOLOv8n": {
                "Parameters": "3.2M",
                "Speed": "~2ms",
                "Best For": "Real-time detection, edge devices"
            },
            "YOLO11n": {
                "Parameters": "2.6M",
                "Speed": "~1.5ms",
                "Best For": "Faster inference, mobile"
            },
            "YOLO26n": {
                "Parameters": "6.4M",
                "Speed": "~4ms",
                "Best For": "Higher accuracy, more compute"
            }
        }
        
        for model_name, metrics in model_metrics.items():
            with st.expander(f"📊 {model_name} Details"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Parameters", metrics["Parameters"])
                col2.metric("Inference Speed", metrics["Speed"])
                col3.write(f"**Use Case:** {metrics['Best For']}")
        
        # Load and show results.csv if available
        st.markdown("#### Training History")
        model_paths = {
            "YOLOv8n": "runs/detect/train3/results.csv",
            "YOLO11n": "runs/detect/train5/results.csv",
            "YOLO26n": "runs/detect/train6/results.csv"
        }
        
        selected_model = st.selectbox("Select model to view training history", list(model_paths.keys()))
        
        try:
            results_path = model_paths[selected_model]
            if os.path.exists(results_path):
                df_results = pd.read_csv(results_path)
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
                if "epoch" in df_results.columns:
                    axes[0, 0].plot(df_results["epoch"], df_results.get("metrics/precision(B)", []))
                    axes[0, 0].set_title("Precision over Epochs")
                    axes[0, 0].set_xlabel("Epoch")
                    
                    axes[0, 1].plot(df_results["epoch"], df_results.get("metrics/recall(B)", []), color="orange")
                    axes[0, 1].set_title("Recall over Epochs")
                    axes[0, 1].set_xlabel("Epoch")
                    
                    axes[1, 0].plot(df_results["epoch"], df_results.get("train/box_loss", []), color="red")
                    axes[1, 0].set_title("Training Loss")
                    axes[1, 0].set_xlabel("Epoch")
                    
                    axes[1, 1].plot(df_results["epoch"], df_results.get("val/box_loss", []), color="purple")
                    axes[1, 1].set_title("Validation Loss")
                    axes[1, 1].set_xlabel("Epoch")
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info(f"Training history not found for {selected_model}")
        except Exception as e:
            st.warning(f"Could not load training metrics: {e}")
    
    # -------- ACCURACY TIPS TAB --------
    with opt_tab4:
        st.markdown("### 💡 Tips to Improve Accuracy")
        
        st.markdown("""
        #### 📷 **Data Quality**
        - ✅ Use high-resolution images (≥640x640)
        - ✅ Ensure proper lighting and contrast
        - ✅ Include diverse backgrounds and angles
        - ✅ Balance dataset (equal samples per class)
        
        #### 🎯 **Detection Settings**
        - ✅ Use Confidence: 0.4-0.6 (balance sensitivity)
        - ✅ Use IOU: 0.4-0.6 (avoid duplicate detections)
        - ✅ Try Ensemble voting for critical applications
        - ✅ Increase imgsz parameter if you have GPU memory
        
        #### 🚀 **Model Selection**
        - ✅ **YOLO26n**: Best accuracy (larger model)
        - ✅ **YOLOv8n**: Good balance of speed & accuracy
        - ✅ **YOLO11n**: Fastest inference
        
        #### 📊 **Advanced Improvements**
        - ✅ Retrain model with more epochs
        - ✅ Data augmentation (rotation, flip, brightness)
        - ✅ Collect more challenging samples
        - ✅ Use larger models (YOLOv8m, YOLOv8l)
        - ✅ Ensemble multiple models
        - ✅ Post-processing filters (NMS adjustment)
        
        #### ⚠️ **Common Issues**
        - ❌ Low confidence = many false positives → **increase threshold**
        - ❌ Missing detections = false negatives → **decrease threshold**
        - ❌ Wrong class prediction → **collect more training data**
        - ❌ Overlapping boxes → **increase IOU threshold**
        """)
        
        st.markdown("---")
        st.markdown("### 📝 Quick Accuracy Checklist")
        st.checkbox("✅ Using correct image resolution (≥640x640)")
        st.checkbox("✅ Tuned confidence threshold (0.4-0.6)")
        st.checkbox("✅ Checked all 3 models, selected best performer")
        st.checkbox("✅ Considered ensemble voting for critical cases")
        st.checkbox("✅ Verified training data quality")

# =================================================
# ANALYTICS TAB
# =================================================
with tabs[5]:
    st.subheader("Detection Analytics")

    if st.session_state.detection_history:
        df_all = pd.concat(st.session_state.detection_history)

        fig, ax = plt.subplots(figsize=(5, 3))
        df_all["Class"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Overall Object Count")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.hist(df_all["Confidence"], bins=10)
        ax2.set_title("Confidence Distribution")
        st.pyplot(fig2)
    else:
        st.info("No detection history yet.")
