import cv2
import shared

back_button = [0,40,0,100]
videoStream = None

def visualisation_mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > back_button[0] and y < back_button[1] and x > back_button[2] and x < back_button[3]:
            shared.APPLICATION_STATE = shared.MENU_STATE

def build_visualisation():
    # open video stream
    global videoStream
    cv2.namedWindow(shared.VISUALISE_WINDOW_NAME)
    cv2.setMouseCallback(shared.VISUALISE_WINDOW_NAME, visualisation_mouse_callback)
    videoStream = cv2.VideoCapture(0)


def show_visualisation():
    success, image = videoStream.read()
    image[back_button[0]:back_button[1], back_button[2]:back_button[3]] = 180
    cv2.putText(image, 'back', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0), 3)
    if success:
        cv2.imshow(shared.VISUALISE_WINDOW_NAME, image)
