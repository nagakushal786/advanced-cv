import cv2 as cv
import mediapipe as mp
import time

class HandDetector():
  def __init__(self, mode=False, max_hands=2, min_det_conf=0.5, min_track_conf=0.5):
    self.mode=mode
    self.max_hands=max_hands
    self.min_det_conf=min_det_conf
    self.min_track_conf=min_track_conf

    self.mp_hands=mp.solutions.hands
    self.hands=self.mp_hands.Hands(
      static_image_mode=self.mode,
      max_num_hands=self.max_hands,
      min_detection_confidence=self.min_det_conf,
      min_tracking_confidence=self.min_track_conf)
    self.mp_draw=mp.solutions.drawing_utils

  def find_hands(self, frame, draw=True):
    rgb_img=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    self.results=self.hands.process(rgb_img)
  
    if self.results.multi_hand_landmarks:
      for handLms in self.results.multi_hand_landmarks:
        if draw:
          self.mp_draw.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)
    return frame
  
  def find_position(self, frame, hand_no=0, draw=True):
    lm_list=[]

    if self.results.multi_hand_landmarks:
      my_hand=self.results.multi_hand_landmarks[hand_no]
      for id, lm in enumerate(my_hand.landmark):
        h, w, c=frame.shape
        cx, cy=int(lm.x*w), int(lm.y*h)
        lm_list.append([id, cx, cy])
        if draw:
          cv.circle(frame, (cx, cy), 8, (0, 255, 0), cv.FILLED)

    return lm_list

def main():
  capture=cv.VideoCapture(0)
  p_time=0
  c_time=0
  detector=HandDetector()

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

if __name__=="__main__":
  main()