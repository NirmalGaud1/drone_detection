import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import requests
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO

# Initialize models
ROBOFLOW_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",  # Updated to standard Roboflow inference endpoint
    api_key="dvO9HlZOMA5WCA7NoXtQ"
)
YOLO_MODEL = YOLO('yolov8n.pt')  # Replace with your custom YOLOv8 model trained for drones
MODEL_ID = "drones_new/3"  # Updated Roboflow model ID

def process_detections(image, results, model_type, conf_threshold):
    """Process detection results from YOLOv8 or Roboflow and draw bounding boxes."""
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    if model_type == "YOLOv8":
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0].item()
                if conf < conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                label = f"{result.names[cls_id]} {conf:.2f}"
                
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_np, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    elif model_type == "Roboflow":
        st.write("Raw Roboflow Results:", results)  # Debug output
        predictions = results.get('predictions', [])
        if not predictions:
            st.warning("No drones detected by Roboflow model.")
        for pred in predictions:
            conf = pred['confidence']
            if conf < conf_threshold:
                continue
            x = int(pred['x'])
            y = int(pred['y'])
            w = int(pred['width'])
            h = int(pred['height'])
            
            x1 = x - w//2
            y1 = y - h//2
            x2 = x + w//2
            y2 = y + h//2
            
            label = f"{pred['class']} {conf:.2f}"
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

def main():
    st.title("Drone Detection System")
    st.sidebar.header("Settings")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Detection Model",
        ["YOLOv8", "Roboflow API"],
        index=1  # Default to Roboflow since it’s drone-specific
    )
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        0.0, 1.0, 0.5, 0.01
    )
    
    # Input selection
    input_option = st.selectbox(
        "Select Input Type",
        ["Image Upload", "Image URL", "Webcam", "Video Upload"]
    )

    if input_option == "Image Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Detect Drones"):
                try:
                    if model_type == "YOLOv8":
                        results = YOLO_MODEL(image, conf=conf_threshold)
                    else:
                        results = ROBOFLOW_CLIENT.infer(image, model_id=MODEL_ID)
                    processed_image = process_detections(image, results, model_type, conf_threshold)
                    st.image(processed_image, caption="Detection Result", use_column_width=True)
                    st.json(results[0].boxes if model_type == "YOLOv8" else results)
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")

    elif input_option == "Image URL":
        url = st.text_input("Enter Image URL")
        if url:
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    image = Image.open(response.raw)
                    st.image(image, caption="Input Image", use_column_width=True)
                    
                    if st.button("Detect Drones"):
                        try:
                            if model_type == "YOLOv8":
                                results = YOLO_MODEL(image, conf=conf_threshold)
                            else:
                                results = ROBOFLOW_CLIENT.infer(url, model_id=MODEL_ID)
                            processed_image = process_detections(image, results, model_type, conf_threshold)
                            st.image(processed_image, caption="Detection Result", use_column_width=True)
                            st.json(results[0].boxes if model_type == "YOLOv8" else results)
                        except Exception as e:
                            st.error(f"Error during detection: {str(e)}")
                else:
                    st.error("Failed to fetch image from URL.")
            except Exception as e:
                st.error(f"Error processing URL: {str(e)}")

    elif input_option == "Webcam":
        st.warning("Webcam feature works best when run locally")
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            st.image(image, caption="Webcam Image", use_column_width=True)
            
            if st.button("Detect Drones"):
                try:
                    if model_type == "YOLOv8":
                        results = YOLO_MODEL(image, conf=conf_threshold)
                    else:
                        results = ROBOFLOW_CLIENT.infer(image, model_id=MODEL_ID)
                    processed_image = process_detections(image, results, model_type, conf_threshold)
                    st.image(processed_image, caption="Detection Result", use_column_width=True)
                    st.json(results[0].boxes if model_type == "YOLOv8" else results)
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")

    elif input_option == "Video Upload":
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                
                try:
                    if model_type == "YOLOv8":
                        results = YOLO_MODEL(pil_image, conf=conf_threshold)
                    else:
                        results = ROBOFLOW_CLIENT.infer(pil_image, model_id=MODEL_ID)
                    processed_frame = process_detections(pil_image, results, model_type, conf_threshold)
                    stframe.image(processed_frame, channels="RGB")
                except Exception as e:
                    st.error(f"Error processing frame: {str(e)}")
                    break
            
            cap.release()
            tfile.close()

if __name__ == "__main__":
    main()
