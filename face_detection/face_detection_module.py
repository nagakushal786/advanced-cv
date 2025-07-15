import cv2 as cv
import mediapipe as mp
import time

class FaceDetector():
  def __init__(self, min_det_conf=0.75):
    self.min_det_conf=min_det_conf

    self.mp_face_detection=mp.solutions.face_detection
    self.face_detect=self.mp_face_detection.FaceDetection(
      min_detection_confidence=self.min_det_conf
    )
    self.mp_draw=mp.solutions.drawing_utils

  def find_face(self, frame, draw=True):
    rgb_img=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    self.results=self.face_detect.process(rgb_img)

    if self.results.detections:
      for id, detection in enumerate(self.results.detections):
        if draw:
          self.mp_draw.draw_detection(frame, detection)
    return frame
  
  def fancy_draw(self, frame, bbox, l=30, t=5, rt=1):
    x, y, w, h=bbox
    x1, y1=x+w, y+h

    cv.rectangle(frame, bbox, (255, 0, 255), rt)

    # Top left
    cv.line(frame, (x, y), (x+l, y), (255, 0, 255), t)
    cv.line(frame, (x, y), (x, y+l), (255, 0, 255), t)

    # Top right
    cv.line(frame, (x1, y), (x1-l, y), (255, 0, 255), t)
    cv.line(frame, (x1, y), (x1, y+l), (255, 0, 255), t)

    # Bottom left
    cv.line(frame, (x, y1), (x+l, y1), (255, 0, 255), t)
    cv.line(frame, (x, y1), (x, y1-l), (255, 0, 255), t)

    # Bottom right
    cv.line(frame, (x1, y1), (x1-l, y1), (255, 0, 255), t)
    cv.line(frame, (x1, y1), (x1, y1-l), (255, 0, 255), t)

    return frame
  
  def get_position(self, frame, draw=True):
    bbox_list=[]

    if self.results.detections:
      for id, detection in enumerate(self.results.detections):
        bbox=detection.location_data.relative_bounding_box
        h, w, c=frame.shape
        bboxC=int(bbox.xmin*w), int(bbox.ymin*h), \
              int(bbox.width*w), int(bbox.height*h)
        bbox_list.append([id, bboxC])
        if draw:
          frame=self.fancy_draw(frame, bboxC)
          cv.putText(frame, f"Accuracy: {int(detection.score[0]*100)}%",
                     (bboxC[0], bboxC[1]-20), cv.FONT_HERSHEY_PLAIN,
                     3, (255, 0, 255), 3)
    
    return bbox_list

def main():
  capture=cv.VideoCapture(r'chapters\\face_detection\\assets\\6.mp4')
  c_time=0
  p_time=0
  detector=FaceDetector()

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

if __name__=="__main__":
  main()