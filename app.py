from collections import Counter
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import cv2
import mediapipe as mp

# Define a function to load the saved model
def load_model():
    with open('knn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Define a function to detect pose landmarks and store them in a DataFrame
def detect_landmarks(cap):
    # Initialize the MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # Extract the number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define an empty DataFrame to store the x, y, z, and visibility data
    columns = ['Frame', 'Joint', 'X', 'Y', 'Z', 'Visibility']
    df = pd.DataFrame(columns=columns)

    # Iterate through each frame of the video and use MediaPipe to detect the pose landmarks
    for frame in range(num_frames):
        ret, image = cap.read()
        if not ret:
            break

        # Convert the image to RGB and process it with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Extract the landmark data and store it in the DataFrame
        if results.pose_landmarks:
            landmarks = [[frame, i, lm.x, lm.y, lm.z, lm.visibility] for i, lm in enumerate(results.pose_landmarks.landmark)]
            df = df.append(pd.DataFrame(landmarks, columns=columns), ignore_index=True)

    return df

# Define a function to classify the pose using the loaded model
def classify_pose(df, model):
    test_set = df.iloc[:, :].values
    ypred = model.predict(df)
    new_data = Counter(ypred)
    result = new_data.most_common(1)
    if result[0][0] == 1:
        return "normal"
    elif result[0][0] == 2:
        return "stage 1"
    elif result[0][0] == 3:
        return "stage 2"
    else:
        return "stage 4"

# Define the Streamlit app
def app():
    st.title("Pose Classification App")

    # Load the saved model
    model = load_model()

    # Upload a video file
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov"])

    if uploaded_file is not None:
        # Open the video file using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        cap = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        # Detect pose landmarks and store them in a DataFrame
        df = detect_landmarks(cap)

        # Classify the pose using the loaded model
        pose_class = classify_pose(df, model)

        # Display the pose classification result
        st.write("Pose classification result: ", pose_class)

# Run the Streamlit app
if __name__ == '__main__':
    app()
