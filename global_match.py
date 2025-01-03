import cv2
import numpy as np

def template_matching(input_image_path, template_image_path, threshold=0.8):
    # 读取输入图像和模板图像
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)

    # 获取模板图像的尺寸
    template_height, template_width = template_image.shape

    # 进行模板匹配
    result = cv2.matchTemplate(input_image, template_image, cv2.TM_CCOEFF_NORMED)

    # 创建与输入图像相同大小的空白图像用于绘制匹配结果
    match_result = np.zeros_like(input_image)

    # 找到匹配程度大于阈值的所有位置
    locations = np.where(result >= threshold)

    # 在匹配位置绘制矩形
    for pt in zip(*locations[::-1]):
        cv2.rectangle(match_result, pt, (pt[0] + template_width, pt[1] + template_height), 255, -1)

    # 二值化匹配结果
    _, binary_match_result = cv2.threshold(match_result, 1, 255, cv2.THRESH_BINARY)

    return binary_match_result

# 示例使用
binary_result = template_matching('origin.jpg', 'template_00.jpg', threshold=0.6)

# 显示结果
cv2.imshow('Binary Match Result', binary_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
