import cv2 as cv
import mediapipe as mp
import time

capture=cv.VideoCapture(r'chapters\\pose_estimation\\assets\\5.mp4')

mp_pose=mp.solutions.pose
pose=mp_pose.Pose()
mp_draw=mp.solutions.drawing_utils

p_time=0
c_time=0

while True:
  isTrue, frame=capture.read()
  resized_frame=cv.resize(frame, (1000, 600), interpolation=cv.INTER_AREA)

  rgb_img=cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
  results=pose.process(rgb_img)
  
  if results.pose_landmarks:
    mp_draw.draw_landmarks(resized_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
      h, w, c=resized_frame.shape
      cx, cy=int(lm.x*w), int(lm.y*h)
      cv.circle(resized_frame, (cx, cy), 5, (0, 255, 0), cv.FILLED)

  c_time=time.time()
  fps=1/(c_time-p_time)
  p_time=c_time

  cv.putText(resized_frame, str(int(fps)), (30, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

  cv.imshow("Video", resized_frame)
  if cv.waitKey(20) & 0xFF==ord('d'):
    break

capture.release()
cv.destroyAllWindows()