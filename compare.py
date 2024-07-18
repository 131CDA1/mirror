import os

import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.linear_model import RANSACRegressor


# from train import *
def load_images_from_folder(folder,face_detection,face_mesh):
    images = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images[filename] = detect_faces(img,face_detection,face_mesh)
    # with open('data.json', 'w') as json_file:
    #     json.dump(images, json_file)
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
def compare_faces(input_image, database_images, face_detection, face_mesh):
    input_face = detect_faces(input_image, face_detection, face_mesh)
    min_distance = 5000.0
    closest_image = None
    closest_image_filename = None

    for filename, database_image in database_images.items():
        # database_faces = detect_faces(database_image, face_detection, face_mesh)
        # for database_face in database_faces:
            # 计算人脸相似度（这里用的是人脸位置的欧氏距离）
        # print(input_face)
        # print(len(database_image))
        input_face = np.array(input_face)
        database_image = np.array(database_image)

        ransac_model = ransac_similarity_transform(input_face, database_image)
        similarity = hausdorff_distance(input_face, database_image)
        print(similarity)
        if similarity < min_distance:
            min_distance = similarity
            closest_image = database_image
            closest_image_filename = filename
        # print(filename)
    return closest_image_filename


def cosine_similarity(matrix1, matrix2):
    vector1 = matrix1.flatten()  # 将二维列表展平为一维数组
    vector2 = matrix2.flatten()  # 将二维列表展平为一维数组

    dot_product = np.dot(vector1, vector2)  # 计算点积
    norm1 = np.linalg.norm(vector1)  # 计算向量1的模
    norm2 = np.linalg.norm(vector2)  # 计算向量2的模

    similarity = dot_product / (norm1 * norm2)  # 计算余弦相似度
    return similarity
def hausdorff_distance(matrix1, matrix2):
    distance1 = directed_hausdorff(matrix1, matrix2)[0]
    distance2 = directed_hausdorff(matrix2, matrix1)[0]
    similarity = max(distance1, distance2)
    return similarity


def euclid_score(points1, points2):
    points1 = np.array(points1)
    points2 = np.array(points2)
    # if points1.shape != (486, 3) or points2.shape != (486, 3):
    #     raise ValueError("输入数组的形状应该是 (486, 3)")

    # 计算每一对关键点之间的欧几里得距离
    distances = np.linalg.norm(points1 - points2, axis=1)

    # 计算所有距离的平均值作为置信度分数
    confidence_score = np.mean(distances)

    return confidence_score


def calculate_similarity_transform(src_points, tgt_points):
    """
    使用PCA计算两组点之间的相似性变换（包括旋转和平移）。

    参数:
    - src_points: 第一组面部关键点的二维数组，形状为 (N, 3)。
    - tgt_points: 第二组面部关键点的二维数组，形状为 (N, 3)。

    返回:
    - 旋转矩阵：一个 (3, 3) 的数组。
    - 平移向量：一个 (3,) 的数组。
    """
    src_points = np.array(src_points)
    tgt_points = np.array(tgt_points)
    if src_points.shape != tgt_points.shape:
        raise ValueError("输入点集的大小必须相同")
    # 计算质心
    centroid_src = src_points.mean(axis=0)
    centroid_tgt = tgt_points.mean(axis=0)

    # 移动点集使其质心在原点
    src_points_centered = src_points - centroid_src
    tgt_points_centered = tgt_points - centroid_tgt

    # 使用PCA找到旋转矩阵
    A = np.dot(src_points_centered.T, tgt_points_centered)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    rotation_matrix = np.dot(Vt.T, U.T)

    # 计算平移向量
    translation_vector = centroid_tgt - np.dot(rotation_matrix, centroid_src)

    return rotation_matrix, translation_vector


def calculate_confidence_score(src_points, tgt_points):
    """
    使用相似性变换计算两组面部关键点的置信度分数。

    参数:
    - src_points: 第一组面部关键点的二维数组，形状为 (486, 3)。
    - tgt_points: 第二组面部关键点的二维数组，形状为 (486, 3)。

    返回:
    - 置信度分数：一个标量，表示变换误差的平均值。
    """
    # 计算相似性变换
    rotation_matrix, translation_vector = calculate_similarity_transform(src_points, tgt_points)

    # 应用变换到源点
    transformed_src_points = np.dot(src_points, rotation_matrix) + translation_vector

    # 计算变换后的误差
    errors = tgt_points - transformed_src_points

    # 计算置信度分数，即误差的平均值
    confidence_score = np.mean(np.linalg.norm(errors, axis=1))

    return confidence_score


def ransac_similarity_transform(src_points, tgt_points):
    """
    使用RANSAC算法计算两组点之间的相似性变换（包括旋转和平移）。

    参数:
    - src_points: 第一组面部关键点的二维数组，形状为 (N, 3)。
    - tgt_points: 第二组面部关键点的二维数组，形状为 (N, 3)。

    返回:
    - RANSAC回归器：拟合了变换的模型。
    """
    if src_points.shape != tgt_points.shape or src_points.shape[1] != 3:
        raise ValueError("输入数组的形状必须为 (N, 3)")

    # 使用RANSAC回归器来拟合变换
    ransac = RANSACRegressor(base_estimator=None, min_samples=3, residual_threshold=1.0,
                             random_state=0)
    ransac.fit(src_points, tgt_points)

    return ransac


# def calculate_confidence_score(src_points, tgt_points, ransac_model):
#     """
#     使用RANSAC算法计算的相似性变换来评估置信度分数。
#
#     参数:
#     - src_points: 第一组面部关键点的二维数组，形状为 (N, 3)。
#     - tgt_points: 第二组面部关键点的二维数组，形状为 (N, 3)。
#     - ransac_model: 使用ransac_similarity_transform方法拟合的RANSAC回归器。
#
#     返回:
#     - 置信度分数：一个标量，表示变换误差的平均值。
#     """
#     # 使用RANSAC模型预测变换后的点
#     transformed_src_points = ransac_model.predict(src_points)
#
#     # 计算变换后的误差
#     errors = transformed_src_points - tgt_points
#
#     # 计算置信度分数，即误差的平均值
#     confidence_score = np.sqrt(mean_squared_error(tgt_points, transformed_src_points))
#
#     return confidence_score

def calculate_by_torch(input_image, face_detection, face_mesh):
    class_labels = {0:'bao', 1:'ding', 2:'zhang', 3:'zhu'}
    model = FacialKeypointsClassifier(input_size=3*468, num_classes=4)
    model.load_state_dict(torch.load('facial_keypoints_classifier.pth'))
    input_face = detect_faces(input_image, face_detection, face_mesh)
    input_face = np.array(input_face).astype(np.float32)
    points_tensor = torch.tensor(input_face)
    points_tensor_flat = points_tensor.flatten()
    model.eval()
    # points_tensor = torch.from_numpy(input_face).to(DEVICE)
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        predictions = model(points_tensor_flat)
    predicted_indices = np.argmax(predictions).item()
    predicted_class_labels = class_labels[predicted_indices]
    return predicted_class_labels
# if __name__ == '__main__':
#     database_images = load_images_from_folder('faces')
#     mp_face_detection = mp.solutions.face_detection
#     face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
#
#     input_image = cv2.imread('bao.jpg')
#     closest_image = compare_faces(input_image, database_images)
#     cv2.imshow('Closest Image', closest_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
