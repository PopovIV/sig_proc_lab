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
out_video_stream = None
is_face_detection = False
is_face_recognition = False
is_pose_detection = True

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class Human:
    #screenspace data
    ss_face_bound_box = (0, 0, 0, 0)

    ss_left_leg_pos = (0, 0)
    ss_right_leg_pos = (0, 0)
    ss_floor = (0, 0)
    ss_height = 0
    ss_head = (0,0)

    face_image = None
    features = None
    id = -1

    pose_landmarks = None

    #worldspace data
    ws_height = 0
    ws_position = [0, 0]

class Recognizer:
    # models
    mp_pose_model = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_face_detection_model = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1)
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

    def cross_prod(self, a, b):
        return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]

    # both arguments coordinates are from 0 to 1
    def getWsPosition(self, point_ss_pos, floor_ss_pos):
        cam_pos = get_camera_coordinates()
        cam_floor = [cam_pos[0], 0, cam_pos[2]]

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
        fix_plane = True
        if fix_plane:
            # however, our projected_point may be not in plane of (floor_point, cam_floor, cam_pos). to fix this,
            # we reproject our "projected_point" again =)
            normal = self.cross_prod([cam_pos[0] - cam_floor[0], cam_pos[1] - cam_floor[1], cam_pos[2] - cam_floor[2]],
                                     [floor_ws_pos[0] - cam_floor[0],
                                      floor_ws_pos[1] - cam_floor[1],
                                      floor_ws_pos[2] - cam_floor[2]])

            normal_len = sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
            normal = [normal[0]/normal_len, normal[1]/normal_len, normal[2]/normal_len]

            d1 = cam_floor[0] * normal[0] + cam_floor[1] * normal[1] + cam_floor[2] * normal[2]
            d2 = point_projected_pos[0] * normal[0] + point_projected_pos[1] * normal[1] + point_projected_pos[2] * normal[2]

            point_projected_pos = [
                point_projected_pos[0] - normal[0] * (d2 - d1),
                point_projected_pos[1] - normal[1] * (d2 - d1),
                point_projected_pos[2] - normal[2] * (d2 - d1)]

            d3 = point_projected_pos[0] * normal[0] + point_projected_pos[1] * normal[1] + point_projected_pos[2] * normal[2]

        numenator = (point_projected_pos[0] - floor_ws_pos[0]) ** 2 + (point_projected_pos[2] - floor_ws_pos[2]) ** 2
        denumenator = (point_projected_pos[0] - cam_floor[0]) ** 2 + (point_projected_pos[2] - cam_floor[2]) ** 2
        alpha = sqrt(numenator / denumenator)

        point_ws = [
            (1 - alpha) * point_projected_pos[0] + alpha * cam_pos[0],
            (1 - alpha) * point_projected_pos[1] + alpha * cam_pos[1],
            (1 - alpha) * point_projected_pos[2] + alpha * cam_pos[2]]
        return point_ws
    def process(self, image):
        # transform image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # run model-s
        face_results = None
        pose_results = None
        if is_face_detection:
            face_results = self.mp_face_detection_model.process(rgb_image)
        if is_pose_detection:
            pose_results = self.mp_pose_model.process(rgb_image)

        result = []
        if is_pose_detection and pose_results.pose_landmarks != None:
            result = []
            # we need 30 and 29 for feet and 0 for head
            h = Human()
            h.ss_left_leg_pos = [pose_results.pose_landmarks.landmark[29].x, pose_results.pose_landmarks.landmark[29].y]
            h.ss_right_leg_pos = [pose_results.pose_landmarks.landmark[30].x, pose_results.pose_landmarks.landmark[30].y]
            h.ss_floor = [
                (h.ss_left_leg_pos[0] + h.ss_right_leg_pos[0]) / 2.0,
                (h.ss_left_leg_pos[1] + h.ss_right_leg_pos[1]) / 2.0]
            h.pose_landmarks = pose_results.pose_landmarks
            ss_nose = [pose_results.pose_landmarks.landmark[0].x, pose_results.pose_landmarks.landmark[0].y]
            h.ss_head = [ss_nose[0] + 0.02, ss_nose[1] - 0.07]
            h.ws_position = self.getWsPosition(floor_ss_pos=h.ss_floor, point_ss_pos=h.ss_head)
            h.ws_height = h.ws_position[1]
            result.append(h)

        # deal with faces first
        if is_face_detection and face_results.detections != None:
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


recognizer = None

def visualisation_mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > back_button[0] and y < back_button[1] and x > back_button[2] and x < back_button[3]:
            shared.APPLICATION_STATE = shared.MENU_STATE
            out_video_stream.release()
def build_visualisation():
    # open video stream
    global video_stream
    global out_video_stream
    global recognizer
    cv2.namedWindow(shared.VISUALISE_WINDOW_NAME)
    cv2.setMouseCallback(shared.VISUALISE_WINDOW_NAME, visualisation_mouse_callback)
    video_stream = shared.VideoCapture(shared.camera_src)#rtsp://admin:88888888@192.168.0.43:10554/tcp/av0_0")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video_stream = cv2.VideoWriter("output.avi", fourcc, 25.0, shared.window_DIM)

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
        if human.pose_landmarks != None:
            print("ws_height", human.ws_height)
            print("ws_pos", human.ws_position)

            estimated_head_position = [int(human.ss_head[0] * image.shape[1]), int(human.ss_head[1] * image.shape[0])]
            estimated_floor_position = [int(human.ss_floor[0] * image.shape[1]),
                                        int(human.ss_floor[1] * image.shape[0])]
            cv2.circle(image, estimated_head_position, 10, (0, 0, 255), -1)
            cv2.circle(image, estimated_floor_position, 10, (0, 255, 0), -1)
            text = "coord:" + str([round(human.ws_position[0], 1), round(human.ws_position[2], 1)])
            cv2.putText(image, text, (estimated_head_position[0] + 20, estimated_head_position[1] + 20), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
            text = "height:" + str(round(human.ws_height * 100)) + "cm"
            cv2.putText(image, text, (estimated_head_position[0] + 20, estimated_head_position[1] + 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        if human.ss_face_bound_box != (0, 0, 0, 0):
            box = human.ss_face_bound_box
            cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 10)
            text = "id " + str(human.id)
            if human.id in recognizer.names:
                text += "(" + recognizer.names[human.id] + ")"
            cv2.putText(image, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)


    image = cv2.resize(image, shared.window_DIM)
    if out_video_stream != None:
        out_video_stream.write(image)

    image[back_button[0]:back_button[1], back_button[2]:back_button[3]] = 180
    cv2.putText(image, 'back', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0), 3)
    cv2.imshow(shared.VISUALISE_WINDOW_NAME, image)
