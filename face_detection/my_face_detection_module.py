import cv2 as cv
import time
import face_detection_module as fdm

capture=cv.VideoCapture(r'chapters\\face_detection\\assets\\2.mp4')
c_time=0
p_time=0
detector=fdm.FaceDetector()

while True:
  isTrue, frame=capture.read()
  frame=detector.find_face(frame, draw=False)
  bbox_list=detector.get_position(frame)

  c_time=time.time()
  fps=1/(c_time-p_time)
  p_time=c_time

  cv.putText(frame, str(int(fps)), (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

  cv.imshow("Video", frame)
  if cv.waitKey(20) & 0xFF==ord('d'):
    break

capture.release()
cv.destroyAllWindows()