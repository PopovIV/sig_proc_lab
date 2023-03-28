import json

import cv2
import shared
import mediapipe as mp
from mediapipe.framework.formats import location_data_pb2
import dlib
from sklearn import neighbors as KNN
from math import sqrt
from calibration import load_matrix, tranform_screen_point_to_floor, get_camera_coordinates

back_button = [0,40,0,100]
video_stream = None
is_face_detection = True
is_face_recognition = True
is_pose_detection = False

class Human:
    #screenspace data
    ss_face_bound_box = (0, 0, 0, 0)
    ss_left_leg_pos = (0, 0)
    ss_right_leg_pos = (0, 0)
    ss_height = 0

    face_image = None
    features = None
    id = -1

    #worldspace data
    ws_height = 0
    ws_position = [0, 0]

class Recognizer:
    # models
    mp_pose_model = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_face_detection_model = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    face_recognition_model = dlib.face_recognition_model_v1("bin/rec.dat")
    face_shape_predictor_model = dlib.shape_predictor("bin/pred68.dat")

    # pairs of (feature, id). For classification
    database = []
    face_classifier = None
    names = {}

    def loadClassifier(self):
        if len(self.database) == 0:
            return
        self.face_classifier = KNN.KNeighborsClassifier(n_neighbors=max(1, int(sqrt(len(self.database)))), algorithm='ball_tree', weights='distance')
        features, ids, _ = zip(*self.database)
        self.face_classifier.fit(features, ids)

        for record in self.database:
            self.names[record[1]] = record[2]

    # both arguments coordinates are from 0 to 1
    def getWsPosition(self, point_ss_pos, floor_ss_pos):
        cam_pos = get_camera_coordinates()
        cam_floor = [cam_pos[0], 0, cam_pos[1]]

        # ss - screen space
        # ws - world space
        point_projected_pos = tranform_screen_point_to_floor(point_ss_pos[0], point_ss_pos[1])
        point_projected_pos = [point_projected_pos[0], 0, point_projected_pos[1]]

        floor_ws_pos = tranform_screen_point_to_floor(floor_ss_pos[0], floor_ss_pos[1])
        floor_ws_pos = [floor_ws_pos[0], 0, floor_ws_pos[1]]

        # we have similar triangles

        #                         * cam_pos
        #                       / |
        #                      /  |
        #              point  /   |
        #                    *    |
        #                   /|    |
        #                  / |    |
        # projected_point *__*____*  cam_floor
        #                   floor_point

        # to get head we need linear interpolate between projected_point and cam_pos
        # with alpha = len(projected_point - floor_point) / len(projected_point - cam_floor)

        numenator = sqrt(
            (point_projected_pos[0] - floor_ws_pos[0]) ** 2 +
            (point_projected_pos[2] - floor_ws_pos[2]) ** 2)
        denumenator = sqrt(
            (point_projected_pos[0] - cam_floor[0]) ** 2 +
            (point_projected_pos[2] - cam_floor[2]) ** 2)
        alpha = numenator / denumenator

        point_ws = [
            (1 - alpha) * point_projected_pos[0] + alpha * cam_pos[0],
            (1 - alpha) * point_projected_pos[1] + alpha * cam_pos[1],
            (1 - alpha) * point_projected_pos[2] + alpha * cam_pos[2]]
        return point_ws
    def process(self, image):
        # transform image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # run model-s
        if is_face_detection:
            face_results = self.mp_face_detection_model.process(rgb_image)
        if is_pose_detection:
            pose_results = self.mp_pose_model.process(rgb_image)
        # deal with faces first
        if is_face_detection and face_results.detections != None:
            result = []

            for face_detection in face_results.detections:
                location_data = face_detection.location_data
                if location_data.format == location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
                    bb = location_data.relative_bounding_box

                    h = Human()
                    # setup bound-box
                    h.ss_face_bound_box = [
                        int(bb.xmin * image.shape[1]),
                        int(bb.ymin * image.shape[0]),
                        int(bb.width * image.shape[1]),
                        int(bb.height * image.shape[0])]
                    # setup image
                    h.face_image = image[
                                   h.ss_face_bound_box[1]: h.ss_face_bound_box[1] + h.ss_face_bound_box[3],
                                   h.ss_face_bound_box[0]: h.ss_face_bound_box[0] + h.ss_face_bound_box[2]]

                    # calculate features from detector. they will be simular for same people
                    rect = dlib.rectangle(
                        h.ss_face_bound_box[0],
                        h.ss_face_bound_box[1],
                        h.ss_face_bound_box[0] + h.ss_face_bound_box[2],
                        h.ss_face_bound_box[1] + h.ss_face_bound_box[3])
                    if is_face_recognition:
                        shape = self.face_shape_predictor_model(rgb_image, rect)
                        h.features = shared.to_list(self.face_recognition_model.compute_face_descriptor(rgb_image, shape, 0))

                    # classify them
                    if is_face_recognition and self.face_classifier != None:
                        data = self.face_classifier.kneighbors([h.features], n_neighbors=1)
                        dist = data[0][0][0]
                        ind = self.face_classifier.predict([h.features])[0]
                        if (dist < 0.5):
                            h.id = ind


                    result.append(h)
            return result
        return []

recognizer = None

def visualisation_mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > back_button[0] and y < back_button[1] and x > back_button[2] and x < back_button[3]:
            shared.APPLICATION_STATE = shared.MENU_STATE
def build_visualisation():
    # open video stream
    global video_stream
    global recognizer
    cv2.namedWindow(shared.VISUALISE_WINDOW_NAME)
    cv2.setMouseCallback(shared.VISUALISE_WINDOW_NAME, visualisation_mouse_callback)
    video_stream = shared.VideoCapture(shared.camera_src)#rtsp://admin:88888888@192.168.0.43:10554/tcp/av0_0")

    recognizer = Recognizer()

    # create classifier
    with open("database.json", "r") as f:
        recognizer.database = json.loads(f.read())
        recognizer.loadClassifier()

    load_matrix()

def show_visualisation():
    global recognizer
    image = video_stream.read()

    image = shared.undistort(image)
    #process our algorithm
    visualised_humans_data = recognizer.process(image)

    # draw faces bound boxes
    for human in visualised_humans_data:
        box = human.ss_face_bound_box
        cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 10)
        text = "id " + str(human.id)
        if human.id in recognizer.names:
            text += "(" + recognizer.names[human.id] + ")"
        cv2.putText(image, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    image = cv2.resize(image, shared.window_DIM)

    image[back_button[0]:back_button[1], back_button[2]:back_button[3]] = 180
    cv2.putText(image, 'back', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0), 3)
    cv2.imshow(shared.VISUALISE_WINDOW_NAME, image)
