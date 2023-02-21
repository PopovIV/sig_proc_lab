import cv2
import shared
import mediapipe as mp
from mediapipe.framework.formats import location_data_pb2

back_button = [0,40,0,100]
video_stream = None
mp_pose_model = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_face_detection_model = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def visualisation_mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > back_button[0] and y < back_button[1] and x > back_button[2] and x < back_button[3]:
            shared.APPLICATION_STATE = shared.MENU_STATE

def build_visualisation():
    # open video stream
    global video_stream
    cv2.namedWindow(shared.VISUALISE_WINDOW_NAME)
    cv2.setMouseCallback(shared.VISUALISE_WINDOW_NAME, visualisation_mouse_callback)
    video_stream = cv2.VideoCapture(0)

def show_visualisation():
    success, image = video_stream.read()

    if success:
        # transform image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # run model-s
        pose_results = mp_pose_model.process(rgb_image)
        face_results = mp_face_detection_model.process(rgb_image)

        faces_bound_boxes = []
        heads_points = []
        feet_points = []

        # collect faces bound-boxes
        if face_results.detections != None:
            for face_detection in face_results.detections:
                location_data = face_detection.location_data
                if location_data.format == location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
                    bb = location_data.relative_bounding_box
                    faces_bound_boxes.append([
                        int(bb.xmin * image.shape[1]),
                        int(bb.ymin * image.shape[0]),
                        int(bb.width * image.shape[1]),
                        int(bb.height * image.shape[0])])

        # draw faces bound boxes
        for box in faces_bound_boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 10)

    image[back_button[0]:back_button[1], back_button[2]:back_button[3]] = 180
    cv2.putText(image, 'back', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0), 3)
    if success:
        cv2.imshow(shared.VISUALISE_WINDOW_NAME, image)
