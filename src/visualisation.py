import json

import cv2
import shared
import mediapipe as mp
from mediapipe.framework.formats import location_data_pb2
import dlib
from sklearn import neighbors as KNN
from math import sqrt

back_button = [0,40,0,100]
video_stream = None

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

    def process(self, image):
        # transform image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # run model-s
        pose_results = self.mp_pose_model.process(rgb_image)
        face_results = self.mp_face_detection_model.process(rgb_image)

        # deal with faces first
        if face_results.detections != None:
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
                    shape = self.face_shape_predictor_model(rgb_image, rect)
                    h.features = shared.to_list(self.face_recognition_model.compute_face_descriptor(rgb_image, shape, 0))

                    # classify them
                    if self.face_classifier != None:
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
    video_stream = cv2.VideoCapture(0)

    recognizer = Recognizer()

    # create classifier
    with open("database.json", "r") as f:
        recognizer.database = json.loads(f.read())
        recognizer.loadClassifier()

def show_visualisation():
    global recognizer
    success, image = video_stream.read()

    if success:
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

    image[back_button[0]:back_button[1], back_button[2]:back_button[3]] = 180
    cv2.putText(image, 'back', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0), 3)
    if success:
        cv2.imshow(shared.VISUALISE_WINDOW_NAME, image)
