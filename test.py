import cv2
import numpy as np

# （1）读取骨骼图像得到img1并显示
img1 = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Image 1', img1)
cv2.imwrite('1.jpg', img1)
# （2）对读取的图像进行Laplacian滤波得到img2
lap = cv2.Laplacian(img1, cv2.CV_16S)
lap = np.absolute(lap)
img2 = cv2.convertScaleAbs(lap)  # 对img2进行对比度拉伸处理
cv2.imshow('Image 2', img2)
cv2.imwrite('2.jpg', img2)

# （3）将前两个图像相加得到img3
img3 = cv2.add(img1, img2)
cv2.imshow('Image 3', img3)
cv2.imwrite('3.jpg', img3)

# （4）用sobel算子对img3分别计算其x、y方向的梯度
sobelx = cv2.Sobel(img3, cv2.CV_16S, 1, 0)
sobely = cv2.Sobel(img3, cv2.CV_16S, 0, 1)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)

# 计算两个梯度和
img4 = cv2.add(sobelx, sobely)
img4 = np.clip(img4, 0, 255).astype('uint8')  # 使用clip函数限制其值在[0,255]
cv2.imshow('Image 4', img4)
cv2.imwrite('1.jpg', img1)

# （5）对img4进行5*5中值滤波，得到img5
img5 = cv2.medianBlur(img4, 5)
cv2.imshow('Image 5', img5)
cv2.imwrite('1.jpg', img1)

# （6）将img3与img5相乘得到img6
img6 = cv2.multiply(img3, img5)
cv2.imshow('Image 6', img6)
cv2.imwrite('1.jpg', img1)

# （7）将img1与img6相加得到img7
img7 = cv2.add(img1, img6)
cv2.imshow('Image 7', img7)
cv2.imwrite('1.jpg', img1)

# （8）对img7执行幂率运算，幂率为0.5
# 先将图像转换为浮点数类型
img7_float = img7.astype(np.float32) / 255.0
# 执行幂运算
img8 = np.power(img7_float, 0.5) * 255  # 注意：幂运算后需要乘以255，再转换回uint8类型
img8 = np.clip(img8, 0, 255).astype('uint8')
cv2.imshow('Image 8', img8)
cv2.imwrite('1.jpg', img1)

cv2.waitKey(0)  # 等待用户按键
cv2.destroyAllWindows()  # 关闭所有窗口