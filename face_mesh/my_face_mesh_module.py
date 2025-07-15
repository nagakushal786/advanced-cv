import cv2 as cv
import time
import face_mesh_module as fmm

capture=cv.VideoCapture(r'chapters\\face_mesh\\assets\\2.mp4')
p_time=0
c_time=0
mesh=fmm.FaceMesh()

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