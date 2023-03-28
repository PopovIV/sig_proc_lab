import numpy as np
import cv2
APPLICATION_STATE = "MENU"

MENU_STATE = "MENU"
CALIBRATION_STATE = "CALIBRATION"
VISUALISE_STATE = "VISUALISATION"

MAIN_WINDOW_NAME = "Main menu"
CALIBRATION_WINDOW_NAME = "Calibration"
VISUALISE_WINDOW_NAME = "Visualisation"

def to_list(vector):
    return [vector[i] for i in range(vector.shape[0])]

camera_src = 0
fisheye_K = np.array([])
fisheye_D = np.array([])
fisheye_DIM = (0, 0)
window_DIM = (640, 480)

#camera_src="rtsp://admin:88888888@192.168.0.43:10554/tcp/av0_0"
#fisheye_K = np.array([[1356.5131044815923, 0.0, 966.3469607973187], [0.0, 1350.8751885658182, 586.936968600807], [0.0, 0.0, 1.0]])
#fisheye_D = np.array([[-0.036212854182096015], [0.2329306759601983], [-1.344989668751758], [1.3384974285832354]])
#fisheye_DIM = (1920, 1080)
#window_DIM = (1280, 720)

fisheye_is_calculated = False
fisheye_map1 = None
fisheye_map2 = None

def undistort(img):
    global fisheye_map1, fisheye_map2, fisheye_is_calculated
    if fisheye_DIM == (0, 0):
        return img
    h,w = img.shape[:2]
    if not fisheye_is_calculated:
        fisheye_map1, fisheye_map2 = cv2.fisheye.initUndistortRectifyMap(fisheye_K, fisheye_D, np.eye(3), fisheye_K, fisheye_DIM, cv2.CV_16SC2)
        fisheye_is_calculated = True
    return cv2.remap(img, fisheye_map1, fisheye_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# buferless videoCapture
import cv2, queue, threading, time
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()
