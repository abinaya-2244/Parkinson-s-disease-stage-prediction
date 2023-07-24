import cv2
import mediapipe as mp
import pandas as pd

# Initialize the MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Load the video file using OpenCV and extract the number of frames
cap = cv2.VideoCapture('STAGE4.mov')
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

# Save the DataFrame to a CSV file
df.to_csv('stage4.csv', index=False)
