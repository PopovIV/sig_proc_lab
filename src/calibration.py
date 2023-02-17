import cv2
import shared

back_button = [0,40,0,100]
save_button = [0,40,120,220]
videoStream = None

def calibration_mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > back_button[0] and y < back_button[1] and x > back_button[2] and x < back_button[3]:
            shared.APPLICATION_STATE = shared.MENU_STATE
        elif y > save_button[0] and y < save_button[1] and x > save_button[2] and x < save_button[3]:
            shared.APPLICATION_STATE = shared.MENU_STATE
            # also save it
        else:
            pass
            # remember positions

def build_calibration():
    # open video stream
    global videoStream
    cv2.namedWindow(shared.CALIBRATION_WINDOW_NAME)
    cv2.setMouseCallback(shared.CALIBRATION_WINDOW_NAME, calibration_mouse_callback)
    videoStream = cv2.VideoCapture(0)


def show_calibration():
    success, image = videoStream.read()
    image[back_button[0]:back_button[1], back_button[2]:back_button[3]] = 180
    cv2.putText(image, 'back', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0), 3)
    image[save_button[0]:save_button[1], save_button[2]:save_button[3]] = 180
    cv2.putText(image, 'save', (130, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0), 3)
    if success:
        cv2.imshow(shared.CALIBRATION_WINDOW_NAME, image)
