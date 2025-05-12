import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import requests
from inference_sdk import InferenceHTTPClient
import google.generativeai as genai
import os
import logging
from math import sqrt

# Set up logging to a file
logging.basicConfig(filename='debug.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# Initialize Roboflow client
ROBOFLOW_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="dvO9HlZOMA5WCA7NoXtQ"
)
MODEL_ID = "dronedet-9ndje/2"

# Configure Google Generative AI
API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"
def configure_generative_model(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        logging.error(f"Error configuring generative model: {e}")
        st.error(f"Error configuring the generative model: {e}")
        return None

GEMINI_MODEL = configure_generative_model(API_KEY)

def process_detections(image, results, conf_threshold, prev_positions, frame_time, frame_count):
    """Process Roboflow detection results, draw bounding boxes, and calculate coordinates/speed."""
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Log results for debugging
    logging.debug(f"Frame {frame_count} Roboflow Results: {results}")
    
    predictions = results.get('predictions', [])
    drone_count = 0
    drone_info = []
    current_positions = []
    
    for pred in predictions:
        conf = pred['confidence']
        if conf < conf_threshold:
            continue
        x = int(pred['x'])
        y = int(pred['y'])
        w = int(pred['width'])
        h = int(pred['height'])
        
        # Calculate bounding box coordinates
        x1 = x - w // 2
        y1 = y - h // 2
        x2 = x + w // 2
        y2 = y + h // 2
        
        # Draw bounding box and label
        label = f"Drone {drone_count + 1} {conf:.2f}"
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Store coordinates
        center = (x, y)
        box = (x1, y1, x2, y2)
        current_positions.append(center)
        
        # Calculate speed (if previous positions exist)
        speed = 0
        if prev_positions and frame_time > 0:
            # Find closest previous position (simple tracking)
            min_dist = float('inf')
            for prev_pos in prev_positions:
                dist = sqrt((center[0] - prev_pos[0])**2 + (center[1] - prev_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
            if min_dist != float('inf'):
                speed = min_dist / frame_time  # Pixels per second
        
        drone_info.append({
            'id': drone_count + 1,
            'center': center,
            'box': box,
            'confidence': conf,
            'speed': speed
        })
        drone_count += 1
    
    return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), drone_count, drone_info, current_positions

def analyze_frame_with_gemini(frame):
    """Analyze a video frame using Google Generative AI."""
    if GEMINI_MODEL is None:
        return "Gemini model not initialized."
    
    try:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        pil_image.save(temp_file.name)
        
        uploaded_file = genai.upload_file(temp_file.name)
        prompt = "Describe this image in detail. Does it likely contain a drone? If so, describe its approximate location (e.g., center, top-left)."
        response = GEMINI_MODEL.generate_content([prompt, uploaded_file])
        
        os.unlink(temp_file.name)
        genai.delete_file(uploaded_file.name)
        
        return response.text
    except Exception as e:
        logging.error(f"Error analyzing frame with Gemini: {e}")
        return f"Error analyzing frame with Gemini: {str(e)}"

def main():
    st.title("Drone Detection System")
    st.sidebar.header("Settings")
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        0.0, 1.0, 0.3, 0.01
    )
    
    # Frame skip for video processing
    frame_skip = st.sidebar.slider(
        "Frame Skip (process every Nth frame)", 
        1, 10, 5, 1
    )
    
    # Input selection
    input_option = st.selectbox(
        "Select Input Type",
        ["Image Upload", "Image URL", "Webcam", "Video Upload"]
    )

    # Store previous positions for speed calculation
    if 'prev_positions' not in st.session_state:
        st.session_state.prev_positions = []

    if input_option == "Image Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Detect Drones"):
                try:
                    results = ROBOFLOW_CLIENT.infer(image, model_id=MODEL_ID)
                    processed_image, drone_count, drone_info, _ = process_detections(
                        image, results, conf_threshold, [], 0, 0
                    )
                    st.image(processed_image, caption=f"Detection Result ({drone_count} drones detected)", use_column_width=True)
                    st.write(f"Number of drones detected: {drone_count}")
                    
                    # Display drone coordinates
                    for drone in drone_info:
                        st.write(f"Drone {drone['id']}:")
                        st.write(f"  Center Coordinates: ({drone['center'][0]}, {drone['center'][1]})")
                        st.write(f"  Bounding Box: ({drone['box'][0]}, {drone['box'][1]}) to ({drone['box'][2]}, {drone['box'][3]})")
                        st.write(f"  Confidence: {drone['confidence']:.2f}")
                    
                    # Gemini analysis
                    st.write("Analyzing image with Google Generative AI...")
                    gemini_result = analyze_frame_with_gemini(np.array(image))
                    st.write("Gemini Analysis:", gemini_result)
                except Exception as e:
                    logging.error(f"Error during image detection: {e}")
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
                            results = ROBOFLOW_CLIENT.infer(url, model_id=MODEL_ID)
                            processed_image, drone_count, drone_info, _ = process_detections(
                                image, results, conf_threshold, [], 0, 0
                            )
                            st.image(processed_image, caption=f"Detection Result ({drone_count} drones detected)", use_column_width=True)
                            st.write(f"Number of drones detected: {drone_count}")
                            
                            for drone in drone_info:
                                st.write(f"Drone {drone['id']}:")
                                st.write(f"  Center Coordinates: ({drone['center'][0]}, {drone['center'][1]})")
                                st.write(f"  Bounding Box: ({drone['box'][0]}, {drone['box'][1]}) to ({drone['box'][2]}, {drone['box'][3]})")
                                st.write(f"  Confidence: {drone['confidence']:.2f}")
                            
                            st.write("Analyzing image with Google Generative AI...")
                            gemini_result = analyze_frame_with_gemini(np.array(image))
                            st.write("Gemini Analysis:", gemini_result)
                        except Exception as e:
                            logging.error(f"Error during URL detection: {e}")
                            st.error(f"Error during detection: {str(e)}")
                else:
                    st.error("Failed to fetch image from URL.")
            except Exception as e:
                logging.error(f"Error processing URL: {e}")
                st.error(f"Error processing URL: {str(e)}")

    elif input_option == "Webcam":
        st.warning("Webcam feature works best when run locally")
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            st.image(image, caption="Webcam Image", use_column_width=True)
            
            if st.button("Detect Drones"):
                try:
                    results = ROBOFLOW_CLIENT.infer(image, model_id=MODEL_ID)
                    processed_image, drone_count, drone_info, _ = process_detections(
                        image, results, conf_threshold, [], 0, 0
                    )
                    st.image(processed_image, caption=f"Detection Result ({drone_count} drones detected)", use_column_width=True)
                    st.write(f"Number of drones detected: {drone_count}")
                    
                    for drone in drone_info:
                        st.write(f"Drone {drone['id']}:")
                        st.write(f"  Center Coordinates: ({drone['center'][0]}, {drone['center'][1]})")
                        st.write(f"  Bounding Box: ({drone['box'][0]}, {drone['box'][1]}) to ({drone['box'][2]}, {drone['box'][3]})")
                        st.write(f"  Confidence: {drone['confidence']:.2f}")
                    
                    st.write("Analyzing image with Google Generative AI...")
                    gemini_result = analyze_frame_with_gemini(np.array(image))
                    st.write("Gemini Analysis:", gemini_result)
                except Exception as e:
                    logging.error(f"Error during webcam detection: {e}")
                    st.error(f"Error during detection: {str(e)}")

    elif input_option == "Video Upload":
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("Error: Could not open video file.")
                logging.error("Could not open video file: %s", tfile.name)
                os.unlink(tfile.name)
                return
            
            # Get frame rate for speed calculation
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_time = (1 / fps) * frame_skip if fps > 0 else 0.033  # Default to 30 FPS if unknown
            
            stframe = st.empty()
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                
                frame = cv2.resize(frame, (640, 480))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                
                try:
                    results = ROBOFLOW_CLIENT.infer(pil_image, model_id=MODEL_ID)
                    processed_frame, drone_count, drone_info, current_positions = process_detections(
                        pil_image, results, conf_threshold, st.session_state.prev_positions, frame_time, frame_count
                    )
                    stframe.image(processed_frame, caption=f"Frame {frame_count} ({drone_count} drones detected)", channels="RGB")
                    st.write(f"Frame {frame_count}: {drone_count} drones detected")
                    
                    for drone in drone_info:
                        st.write(f"Drone {drone['id']}:")
                        st.write(f"  Center Coordinates: ({drone['center'][0]}, {drone['center'][1]})")
                        st.write(f"  Bounding Box: ({drone['box'][0]}, {drone['box'][1]}) to ({drone['box'][2]}, {drone['box'][3]})")
                        st.write(f"  Confidence: {drone['confidence']:.2f}")
                        st.write(f"  Speed: {drone['speed']:.2f} pixels/second")
                    
                    # Update previous positions
                    st.session_state.prev_positions = current_positions
                    
                    # Gemini analysis (every 10th processed frame)
                    if frame_count % (frame_skip * 10) == 0:
                        st.write(f"Analyzing frame {frame_count} with Google Generative AI...")
                        gemini_result = analyze_frame_with_gemini(frame)
                        st.write(f"Gemini Analysis for Frame {frame_count}:", gemini_result)
                except Exception as e:
                    logging.error(f"Error processing frame {frame_count}: {e}")
                    st.error(f"Error processing frame {frame_count}: {str(e)}")
                    break
            
            cap.release()
            os.unlink(tfile.name)

if __name__ == "__main__":
    main()
