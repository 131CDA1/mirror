import mediapipe as mp

from compare import *

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

database_images = load_images_from_folder('face', face_detection, face_mesh)

frame = cv2.imread('faces/bao5.jpg')
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_detection.process(frame_rgb)
if results:
    closest_image = calculate_by_torch(frame_rgb, face_detection, face_mesh)
    print(closest_image)