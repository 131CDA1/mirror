import mediapipe as mp

from compare import *

if __name__ == '__main__':
    # 初始化Mediapipe的Face模型
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection()
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    database_images = load_images_from_folder('face', face_detection, face_mesh)
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    n = 0
    while cap.isOpened():
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            break

        # 将图像转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 进行人脸检测
        results = face_detection.process(frame_rgb)
        if results.detections:
            n += 1
            if n >= 30:
                try:
                    closest_image = calculate_by_torch(frame_rgb, face_detection, face_mesh)
                    print(closest_image)
                except:
                    pass
                n = 0
        # 绘制关键点
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)

                # 绘制人脸关键点
                mp_drawing.draw_detection(frame, detection)

                # 输出人脸位置信息
                print("Bounding box:", bbox)

                # 进行人脸关键点检测
                face_results = face_mesh.process(frame_rgb)
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        for landmark in face_landmarks.landmark:
                            x, y = int(landmark.x * iw), int(landmark.y * ih)
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # 显示结果
        cv2.imshow('Face Detection', frame)

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
