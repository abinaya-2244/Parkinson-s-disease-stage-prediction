import pandas as pd
import pickle
import numpy as np
import cv2
import mediapipe as mp

# Load the saved model
with open('knn_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Initialize the MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Load the video file using OpenCV and extract the number of frames
cap = cv2.VideoCapture('test.mov')
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

print(df)

test_set = df.iloc[:, : ].values

ypred = model.predict(df)

print(len(ypred))

from collections import Counter
new_data = Counter(ypred)
print(new_data.most_common()) #returns all unique items and their counts 
result = new_data.most_common(1) #return the highest occuring item

new_data.most_common(1) #return the highest occuring item

print(type(result))

if result[0][0] == 1:
    print("normal")
elif result[0][0] == 2:
    print("stage 1")
elif result[0][0] == 3:
    print("stage 2")
else:
    print("stage 4")