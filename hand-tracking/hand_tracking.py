import cv2 as cv
import mediapipe as mp
import time

capture=cv.VideoCapture(0)

mp_hands=mp.solutions.hands
hands=mp_hands.Hands()
mp_draw=mp.solutions.drawing_utils

p_time=0
c_time=0

while True:
  isTrue, frame=capture.read()

  rgb_img=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
  results=hands.process(rgb_img)
  
  if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
      for id, lm in enumerate(handLms.landmark):
        h, w, c=frame.shape
        cx, cy=int(lm.x*w), int(lm.y*h)
        cv.circle(frame, (cx, cy), 8, (0, 255, 0), cv.FILLED)

      mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

  c_time=time.time()
  fps=1/(c_time-p_time)
  p_time=c_time

  cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

  cv.imshow("Capture", frame)
  if cv.waitKey(20) & 0xFF==ord('d'):
    break

capture.release()
cv.destroyAllWindows()