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

def video_input():
    vid_file = None
    vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
    if vid_bytes:
        try:
            # Use a unique filename based on current time to avoid conflicts
            import time
            timestamp = int(time.time())
            vid_file = f"data/uploaded_data/upload_{timestamp}.mp4"

            # Save the uploaded video to the specified path
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

            # Provide a success message to the user
            st.sidebar.success(f"Video uploaded successfully as {vid_file}")
        except Exception as e:
            st.sidebar.error(f"An error occurred while processing the uploaded video: {e}")
            st.write(e)  # Print the exception details to the app for debugging

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        st.markdown("---")
        output = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame...")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Add your image inference code here using the model
            # Replace 'model\Arma_(Gun)detector.pt' with the actual model path
            # Example: model_path = "path_to_your_model/model.pt"
            model_path = r"model\Arma_(Gun)detector.pt"
            # Load the model and perform inference
            # model = load_model(model_path)
            # prediction = infer_image(frame, model)
            # Display the prediction on the frame
            # frame_with_prediction = draw_prediction(frame, prediction)
            output.image(frame, use_column_width=True)  # Display the video on full screen

if __name__ == "__main__":
    video_input()
