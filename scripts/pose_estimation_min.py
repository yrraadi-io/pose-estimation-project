import cv2
import mediapipe as mp
import time

mp_draw = mp.solutions.drawing_utils
# create mp solutions pose object
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# create a cv2 VideoCapture object that reads from video file
cap = cv2.VideoCapture("./pose-videos/golf.mp4")
p_time = 0

while True:
    success, img = cap.read()
    # convert img to RGB format and store in img_RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_RGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    # put fps in the image
    cv2.putText(
        img,
        str(int(fps)),
        (70, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (255, 0, 255),
        3,
    )
    # display the image
    cv2.imshow("image", img)

    cv2.waitKey(1)
