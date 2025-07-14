import cv2 as cv
import time
import pose_estimation_module as psm

capture=cv.VideoCapture(r'chapters\\pose_estimation\\assets\\1.mp4')
p_time=0
c_time=0
detector=psm.PoseDetector()

while True:
  isTrue, frame=capture.read()
  resized_frame=cv.resize(frame, (1000, 600), interpolation=cv.INTER_AREA)
  resized_frame=detector.find_pose(resized_frame)
  lm_list=detector.get_position(resized_frame)

  c_time=time.time()
  fps=1/(c_time-p_time)
  p_time=c_time

  cv.putText(resized_frame, str(int(fps)), (30, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

  cv.imshow("Video", resized_frame)
  if cv.waitKey(20) & 0xFF==ord('d'):
    break

capture.release()
cv.destroyAllWindows()