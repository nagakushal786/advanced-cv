import cv2 as cv
import mediapipe as mp
import time

mp_draw=mp.solutions.drawing_utils
mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(max_num_faces=2)
draw_specs=mp_draw.DrawingSpec(thickness=1, circle_radius=1)

capture=cv.VideoCapture(r'chapters\\face_mesh\\assets\\6.mp4')
p_time=0
c_time=0

while True:
  isTrue, frame=capture.read()
  rgb_img=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
  results=face_mesh.process(rgb_img)

  if results.multi_face_landmarks:
    # Looping through multiple faces
    for faceLms in results.multi_face_landmarks:
      mp_draw.draw_landmarks(frame, faceLms, mp_face_mesh.FACEMESH_TESSELATION, draw_specs, draw_specs)
      mp_draw.draw_landmarks(frame, faceLms, mp_face_mesh.FACEMESH_CONTOURS, draw_specs, draw_specs)
      # Looping through various landmarks in a face
      for id, lm in enumerate(faceLms.landmark):
        h, w, c=frame.shape
        cx, cy=int(lm.x*w), int(lm.y*h)

  c_time=time.time()
  fps=1/(c_time-p_time)
  p_time=c_time

  cv.putText(frame, str(int(fps)), (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

  cv.imshow("Video", frame)
  if cv.waitKey(20) & 0xFF==ord('d'):
    break

capture.release()
cv.destroyAllWindows()