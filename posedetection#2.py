import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import csv

# Define the headers for the DataFrame
headers = ['x_0', 'y_0', 'z_0', 'v_0', 'x_1', 'y_1', 'z_1', 'v_1', ... , 'x_32', 'y_32', 'z_32', 'v_32']


# create an empty dataframe to store the landmark values
keypoints = pd.DataFrame(columns=['x','y','z','visibility'])

# initialize mediapipe pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# take video input for pose detection
# you can put here video of your choice
cap = cv2.VideoCapture("STAGE1.mov")


# take live camera  input for pose detection
# cap = cv2.VideoCapture(0)

# read each frame/image from capture object
while True:
    ret, img = cap.read()
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # resize image/frame so we can accommodate it on our screen
    img = cv2.resize(img, (600
                           , 600
                           ))

    # do Pose detection
    results = pose.process(img)
    # draw the detected pose on original video/ live stream
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                           mp_draw.DrawingSpec((0, 250, 0), 2, 2)
                           )
    # Display pose on original video/live stream
    cv2.imshow("Pose Estimation", img)

    # Extract and draw pose on plain white image
    h, w, c = img.shape   # get shape of original frame
    opImg = np.zeros([h, w, c])  # create blank image with original frame size
    opImg.fill(0)  # set white background. put 0 if you want to make it black

    # draw extracted pose on black white image
    mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                           mp_draw.DrawingSpec((0, 250, 0), 2, 2)
                           )
    # display extracted pose on blank images
    cv2.imshow("Extracted Pose", opImg)

    # print all landmarks
    print(results.pose_landmarks)

    # Extract the landmark data from results.pose_landmarks
    landmark_data = []
    for landmark in results.pose_landmarks.landmark:
        landmark_data.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

    # Add the landmark data to the DataFrame
    keypoints.loc[len(keypoints)] = [item for sublist in landmark_data for item in sublist]

    # Write the DataFrame to a CSV file
    keypoints.to_csv('pose_landmarks.csv', index=False)
    
    cv2.waitKey(1)