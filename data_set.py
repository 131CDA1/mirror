import json
import os

import cv2
import mediapipe as mp


def load_images_from_folder(folder,face_detection,face_mesh):
    name = {'bao':0, 'ding':1, 'zhang':2, 'zhu':3}
    images = []
    labels = []
    for filename in os.listdir(folder):
        image = {}
        keypoints = []
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            keypoint = detect_faces(img, face_detection, face_mesh)
            label = name[filename[0:-5]]
            keypoints = keypoint
            labels = label
        image['label'] = labels
        image['keypoints'] = keypoints
        images.append(image)
    with open('data.json', 'w') as json_file:
        json.dump(images, json_file)
    return images

def detect_faces(image, face_detection, face_mesh):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    face_landmarks = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            # 进行关键点检测
            results_mesh = face_mesh.process(image_rgb)
            if results_mesh.multi_face_landmarks:
                for face_landmark in results_mesh.multi_face_landmarks:
                    for landmark in face_landmark.landmark:
                        x = int(landmark.x * iw)
                        y = int(landmark.y * ih)
                        z = int(landmark.z * iw)
                        face_landmarks.append([x, y, z])

    return face_landmarks




if __name__ == '__main__':
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection()
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)


    data = load_images_from_folder('faces', face_detection, face_mesh)
