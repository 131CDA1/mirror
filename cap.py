import json

import cv2
import mediapipe as mp

# 初始化MediaPipe人脸检测和面部关键点模型
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection()
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# 初始化存储数据的列表
data = []

# 设置视频捕获
cap = cv2.VideoCapture(0)

# 用于存储人脸关键点的函数

# 采集人脸图像和关键点
name = {'bao': 0, 'ding': 1, 'zhang': 2, 'zhu': 3}
# 等待用户输入标签
label = name[input("输入标签并按回车键: ")]
while len(data) < 1000:
    ret, frame = cap.read()
    if not ret:
        print("无法读取图像，退出程序。")
        break
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    face_landmarks = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
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


        # 保存数据
        data.append({
            "label": label,
            "keypoints": face_landmarks
        })

    # 显示图像
    cv2.imshow('Face Keypoints', frame)

    # 检查是否按下 'p' 或 'q' 键
    if cv2.waitKey(1) & 0xFF in [ord('p'), ord('q')]:
        break

# print(data)

# 保存到JSON文件
filename = "face_data.json"
# 检查JSON文件是否存在，如果存在则加载现有数据
try:
    with open(filename, 'r') as infile:
        existing_data = json.load(infile)
except FileNotFoundError:
    existing_data = []

# 将新数据添加到现有数据列表中
data_to_add = {

    "label": label,
    "keypoints": face_landmarks
}
existing_data.append(data_to_add)

# 将合并后的数据写入JSON文件
with open(filename, 'w') as outfile:
    json.dump(existing_data, outfile, indent=4)
# 释放资源
cap.release()
cv2.destroyAllWindows()
print(f"数据已保存至 {filename}")
