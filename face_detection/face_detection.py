import cv2 as cv
import mediapipe as mp
import time

mp_face_detection=mp.solutions.face_detection
face_detect=mp_face_detection.FaceDetection(0.75)
mp_draw=mp.solutions.drawing_utils

capture=cv.VideoCapture(r'chapters\\face_detection\\assets\\2.mp4')
c_time=0
p_time=0

while True:
  isTrue, frame=capture.read()
  rgb_img=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
  results=face_detect.process(rgb_img)

  if results.detections:
    for id, detection in enumerate(results.detections):
      # mp_draw.draw_detection(frame, detection)
      bbox=detection.location_data.relative_bounding_box
      h, w, c=frame.shape
      bboxC=int(bbox.xmin*w), int(bbox.ymin*h), \
            int(bbox.width*w), int(bbox.height*h)
      cv.rectangle(frame, bboxC, (255, 0, 255), 2)
      cv.putText(frame, f"Accuracy: {int(detection.score[0]*100)}%",
                 (bboxC[0], bboxC[1]-20), cv.FONT_HERSHEY_PLAIN,
                 3, (255, 0, 255), 3)

  c_time=time.time()
  fps=1/(c_time-p_time)
  p_time=c_time

  cv.putText(frame, str(int(fps)), (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

  cv.imshow("Video", frame)
  if cv.waitKey(20) & 0xFF==ord('d'):
    break

capture.release()
cv.destroyAllWindows()