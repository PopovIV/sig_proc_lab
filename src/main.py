import cv2
import numpy as np
import shared
from calibration import build_calibration, show_calibration
from visualisation import build_visualisation, show_visualisation

calibration_button = [20,60,0,300]
visualise_button = [70,110,0,300]
main_menu_image = None

def main_menu_mouse_callback(event, x, y,flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > calibration_button[0] and y < calibration_button[1] and x > calibration_button[2] and x < calibration_button[3]:
            shared.APPLICATION_STATE = shared.CALIBRATION_STATE
        elif y > visualise_button[0] and y < visualise_button[1] and x > visualise_button[2] and x < visualise_button[3]:
            shared.APPLICATION_STATE = shared.VISUALISE_STATE

def build_main_menu():
    global main_menu_image
    cv2.namedWindow(shared.MAIN_WINDOW_NAME)
    cv2.setMouseCallback(shared.MAIN_WINDOW_NAME, main_menu_mouse_callback)
    main_menu_image = np.zeros((160, 300), np.uint8)
    main_menu_image[calibration_button[0]:calibration_button[1], calibration_button[2]:calibration_button[3]] = 180
    cv2.putText(main_menu_image, 'Calibration', (60, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0), 3)
    main_menu_image[visualise_button[0]:visualise_button[1], visualise_button[2]:visualise_button[3]] = 180
    cv2.putText(main_menu_image, 'Visualisation', (60, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0), 3)

def show_menu():
    cv2.imshow(shared.MAIN_WINDOW_NAME, main_menu_image)

def run_main_loop():
    is_run = True
    shared.APPLICATION_STATE = shared.MENU_STATE
    PREVIOUS_STATE = ""
    cv2.namedWindow(shared.MAIN_WINDOW_NAME)
    cv2.namedWindow(shared.CALIBRATION_WINDOW_NAME)
    cv2.namedWindow(shared.VISUALISE_WINDOW_NAME)
    while is_run:
        # if state changed
        if (PREVIOUS_STATE != shared.APPLICATION_STATE):
            # close all already opened windows
            cv2.destroyAllWindows()
            PREVIOUS_STATE = shared.APPLICATION_STATE
            # build each state
            if shared.APPLICATION_STATE == shared.MENU_STATE:
                build_main_menu()
            elif shared.APPLICATION_STATE == shared.CALIBRATION_STATE:
                build_calibration()
            elif shared.APPLICATION_STATE == shared.VISUALISE_STATE:
                build_visualisation()

        # check if we need to stop
        if cv2.getWindowProperty(shared.MAIN_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1 and \
           cv2.getWindowProperty(shared.CALIBRATION_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1 and \
           cv2.getWindowProperty(shared.VISUALISE_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            is_run = False
            cv2.destroyAllWindows()
            continue

        # visualise using state
        if shared.APPLICATION_STATE == shared.MENU_STATE:
            show_menu()
        elif shared.APPLICATION_STATE == shared.CALIBRATION_STATE:
            show_calibration()
        elif shared.APPLICATION_STATE == shared.VISUALISE_STATE:
            show_visualisation()

        # update
        cv2.waitKey(1)

if __name__ == "__main__":
    run_main_loop()
