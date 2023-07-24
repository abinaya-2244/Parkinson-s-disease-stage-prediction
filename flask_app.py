from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
import cv2
import mediapipe as mp

app = Flask(__name__)

# Load the saved model
with open('knn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize the MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)


ALLOWED_EXTENSIONS = ['mp4']
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    
    
    video = request.files['video']
    video_path='static/videos/' + video.filename
    video.save(video_path)
    
    # Load the video file using OpenCV and extract the number of frames
    cap = cv2.VideoCapture(video_path)
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

    test_set = df.iloc[:, :].values
    ypred = model.predict(df)

    from collections import Counter
    new_data = Counter(ypred)
    result = new_data.most_common(1)[0][0]

    if result == 1:
        result_str = "normal"
    elif result == 2:
        result_str = "stage 1"
    elif result == 3:
        result_str = "stage 2"
    else:
        result_str = "stage 4"

    return render_template('result.html', result=result_str)

if __name__ == '__main__':
    app.run(debug=True)
