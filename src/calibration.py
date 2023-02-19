import cv2
import shared
import copy
import numpy as np
import tkinter as tk

back_button = [0,40,0,100]
save_button = [0,40,120,220]
camera_button = [0,40,240,380]
result_square = [[0,0], [0, 480], [640, 480], [640, 0]]
calibr_square = None
point_clicked = 0
camera_coordinates = [0, 0, 0]
videoStream = None

def widget_callback(x_entry, y_entry, z_entry, win):
    global camera_coordinates
    if x_entry is not None and y_entry is not None and z_entry is not None:
        camera_coordinates[0] = float(x_entry.get())
        camera_coordinates[1] = float(y_entry.get())
        camera_coordinates[2] = float(z_entry.get())
        win.destroy()


def calibration_mouse_callback(event, x, y, flags, params):
    global point_clicked
    global result_square
    global calibr_square
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > back_button[0] and y < back_button[1] and x > back_button[2] and x < back_button[3]:
            shared.APPLICATION_STATE = shared.MENU_STATE
            calibr_square = result_square
            point_clicked = 0
            print(camera_coordinates)
        elif y > save_button[0] and y < save_button[1] and x > save_button[2] and x < save_button[3]:
            shared.APPLICATION_STATE = shared.MENU_STATE
            result_square = calibr_square
            point_clicked = 0
        elif y > camera_button[0] and y < camera_button[1] and x > camera_button[2] and x < camera_button[3]:
            win = tk.Tk()
            win.title("Camera calibration")
            tk.Label(win, text = "X-coordinate").grid(row = 0)
            tk.Label(win, text = "Y-coordinate").grid(row = 1)
            tk.Label(win, text = "Z-coordinate").grid(row = 2)
            x_entry = tk.Entry(win)
            x_entry.grid(row = 0, column = 1)
            y_entry = tk.Entry(win)
            y_entry.grid(row = 1, column = 1)
            z_entry = tk.Entry(win)
            z_entry.grid(row = 2, column = 1)
            x_entry.insert(0, str(camera_coordinates[0]))
            y_entry.insert(0, str(camera_coordinates[1]))
            z_entry.insert(0, str(camera_coordinates[2]))
            tk.Button(win, text = "Ok", command = lambda : widget_callback(x_entry, y_entry, z_entry, win)).grid(row = 4, column = 1, sticky= tk.W, pady = 4)
            win.mainloop()
            

        else:
            calibr_square[point_clicked % 4][0] = x
            calibr_square[point_clicked % 4][1] = y
            point_clicked += 1

def build_calibration():
    # open video stream
    global videoStream
    global calibr_square
    calibr_square =  copy.deepcopy(result_square)
    cv2.namedWindow(shared.CALIBRATION_WINDOW_NAME)
    cv2.setMouseCallback(shared.CALIBRATION_WINDOW_NAME, calibration_mouse_callback)
    videoStream = cv2.VideoCapture(0)

def show_calibration():
    global calibr_square
    success, image = videoStream.read()
    image[back_button[0]:back_button[1], back_button[2]:back_button[3]] = 180
    cv2.putText(image, 'back', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0), 3)
    image[save_button[0]:save_button[1], save_button[2]:save_button[3]] = 180
    cv2.putText(image, 'save', (130, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0), 3)
    image[camera_button[0]:camera_button[1], camera_button[2]:camera_button[3]] = 180
    cv2.putText(image, 'camera', (250, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0), 3)
    pts = np.array([calibr_square[0], calibr_square[1], calibr_square[2], calibr_square[3]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], True, (0, 255, 0))
    if success:
        cv2.imshow(shared.CALIBRATION_WINDOW_NAME, image)
