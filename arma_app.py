import cv2
import numpy as np
import streamlit as st
import tempfile
import os
import torch
import torchvision.transforms as transforms
from torchvision import models

# Load the PyTorch model
model = torch.load('gun_detector_model.pt', map_location=torch.device('cpu'))
model.eval()

# Define a function to preprocess a frame for inference
def preprocess_frame(frame):
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                                    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
                                                                                     [0.229, 0.224, 0.225])])
    frame = transform(frame)
    frame = frame.unsqueeze(0)
    return frame

# Create a Streamlit app
st.title("Gun Detection App")

# Upload a video file
video_file = st.file_uploader("Upload a video file", type=["mp4"])
if video_file:
    # Save the uploaded video temporarily
    temp_video_path = os.path.join(tempfile.gettempdir(), video_file.name)
    with open(temp_video_path, "wb") as temp_file:
        temp_file.write(video_file.read())

    # Open the video file
    cap = cv2.VideoCapture(temp_video_path)

    if not cap.isOpened():
        st.error("Error: Unable to open the uploaded video.")
    else:
        st.success("Video opened successfully.")

    # Display the video feed and perform gun detection
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("End of video.")
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)

        # Pass the preprocessed frame through your PyTorch model for inference
        with torch.no_grad():
            prediction = model(preprocessed_frame)

        # Check the prediction to decide if it's a gun or not
        if prediction.item() >= 0.5:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)

        # Display the frame in Streamlit
        st.image(frame, channels="BGR", use_column_width=True)

    # Release the video capture
    cap.release()
    st.warning("Video closed.")

    # Remove the temporary video file
    os.remove(temp_video_path)
