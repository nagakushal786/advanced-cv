import cv2 as cv
import mediapipe as mp
import time

class FaceMesh():
  def __init__(self, mode=False, max_faces=2, min_det_conf=0.5, min_track_conf=0.5, t=1, cr=1):
    self.mode=mode
    self.max_faces=max_faces
    self.min_det_conf=min_det_conf
    self.min_track_conf=min_track_conf
    self.thickness=t
    self.circ_rad=cr

    self.mp_draw=mp.solutions.drawing_utils
    self.mp_face_mesh=mp.solutions.face_mesh
    self.face_mesh=self.mp_face_mesh.FaceMesh(
      static_image_mode=self.mode,
      max_num_faces=self.max_faces,
      min_detection_confidence=self.min_det_conf,
      min_tracking_confidence=self.min_track_conf)
    self.draw_specs=self.mp_draw.DrawingSpec(
      thickness=self.thickness,
      circle_radius=self.circ_rad)

  def find_mesh(self, frame, draw=True):
    rgb_img=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    self.results=self.face_mesh.process(rgb_img)
    faces=[]

    if self.results.multi_face_landmarks:
      # Looping through multiple faces
      for faceLms in self.results.multi_face_landmarks:
        if draw:
          self.mp_draw.draw_landmarks(frame, faceLms, self.mp_face_mesh.FACEMESH_TESSELATION, self.draw_specs, self.draw_specs)
          self.mp_draw.draw_landmarks(frame, faceLms, self.mp_face_mesh.FACEMESH_CONTOURS, self.draw_specs, self.draw_specs)
          # Looping through various landmarks in a face
        face=[]
        for id, lm in enumerate(faceLms.landmark):
          h, w, c=frame.shape
          cx, cy=int(lm.x*w), int(lm.y*h)
          # If you want to see the landmarks in the image
          # cv.putText(frame, str(int(id)), (cx, cy), cv.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 1)
          face.append([id, cx, cy])
        faces.append(face)

    return frame, faces

def main():
  capture=cv.VideoCapture(r'chapters\\face_mesh\\assets\\2.mp4')
  p_time=0
  c_time=0
  mesh=FaceMesh()

  while True:
    isTrue, frame=capture.read()
    frame, faces=mesh.find_mesh(frame)

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