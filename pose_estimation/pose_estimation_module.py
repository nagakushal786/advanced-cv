import cv2 as cv
import mediapipe as mp
import time

class PoseDetector():
  def __init__(self, mode=False, smooth=True, min_det_conf=0.5, min_track_conf=0.5):
    self.mode=mode
    self.smooth=smooth
    self.min_det_conf=min_det_conf
    self.min_track_conf=min_track_conf

    self.mp_pose=mp.solutions.pose
    self.pose=self.mp_pose.Pose(
      static_image_mode=self.mode,
      smooth_landmarks=self.smooth,
      min_detection_confidence=self.min_det_conf,
      min_tracking_confidence=self.min_track_conf
    )
    self.mp_draw=mp.solutions.drawing_utils

  def find_pose(self, img, draw=True):
    rgb_img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    self.results=self.pose.process(rgb_img)
  
    if self.results.pose_landmarks:
      if draw:
        self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)  
    return img
  
  def get_position(self, img, draw=True):
    lm_list=[]

    if self.results.pose_landmarks:
      for id, lm in enumerate(self.results.pose_landmarks.landmark):
        h, w, c=img.shape
        cx, cy=int(lm.x*w), int(lm.y*h)
        lm_list.append([id, cx, cy])
        if draw:
          cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)
    return lm_list

def main():
  capture=cv.VideoCapture(r'chapters\\pose_estimation\\assets\\5.mp4')
  p_time=0
  c_time=0
  detector=PoseDetector()

  while True:
    isTrue, frame=capture.read()
    resized_frame=cv.resize(frame, (1000, 600), interpolation=cv.INTER_AREA)
    resized_frame=detector.find_pose(resized_frame)

    # For all landmarks
    lm_list=detector.get_position(resized_frame)

    # For a particular landmark
    # lm_list=detector.get_position(resized_frame, draw=False)
    # cv.circle(resized_frame, (lm_list[14][1], lm_list[14][2]), 5, (0, 255, 0), cv.FILLED)

    c_time=time.time()
    fps=1/(c_time-p_time)
    p_time=c_time

    cv.putText(resized_frame, str(int(fps)), (30, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv.imshow("Video", resized_frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
      break

  capture.release()
  cv.destroyAllWindows()

if __name__=="__main__":
  main()