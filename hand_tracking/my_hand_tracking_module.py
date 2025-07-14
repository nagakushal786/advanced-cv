import cv2 as cv
import time
import hand_tracking_module as htm

capture=cv.VideoCapture(0)
p_time=0
c_time=0
detector=htm.HandDetector()

while True:
  isTrue, frame=capture.read()
  frame=detector.find_hands(frame)
  lm_list=detector.find_position(frame)

  c_time=time.time()
  fps=1/(c_time-p_time)
  p_time=c_time

  cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

  cv.imshow("Capture", frame)
  if cv.waitKey(20) & 0xFF==ord('d'):
    break

capture.release()
cv.destroyAllWindows()