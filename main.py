import streamlit as st
import glob
from PIL import Image
import cv2

# Set page configuration as the first Streamlit command
st.set_page_config(layout="wide")

# Author details
st.sidebar.markdown("Author: MobiNext Technologies")
st.sidebar.markdown("Task: Real-time object detection-GUN detection")

# Centered title with HTML and CSS
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <h1>VAMS-MobiNext</h1>
    </div>
    """,
    unsafe_allow_html=True
)

import cv2
import numpy as np
import streamlit as st
import tempfile
import os
import keras

# Load the model
cnn_model = keras.models.load_model('gun_detector_model.h5')

# Define a function to preprocess a frame for inference
def preprocess_frame(frame):
    frame = cv2.resize(frame, (416, 416))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
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

        # Pass the preprocessed frame through your CNN model for inference
        prediction = cnn_model.predict(preprocessed_frame)

        # Check the prediction to decide if it's a gun or not
        if prediction >= 0.5:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)

        # Display the frame in Streamlit
        st.image(frame, channels="BGR", use_column_width=True)

    # Release the video capture
    cap.release()
    st.warning("Video closed.")

    # Remove the temporary video file
    os.remove(temp_video_path)



if __name__ == "__main__":
    video_input()
