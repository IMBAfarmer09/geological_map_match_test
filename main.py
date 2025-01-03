import cv2
import numpy as np

def multi_scale_template_matching(map_image_path, template_image_path, threshold=0.8):
    # 读取地图和模板图像
    map_img = cv2.imread(map_image_path)
    template_img = cv2.imread(template_image_path)

    # 将图像转换为灰度图像
    map_gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    # 获取模板的宽度和高度
    tH, tW = template_gray.shape[:2]

    # 初始化变量以存储最佳匹配结果
    best_match = None
    best_scale = 1.0
    best_loc = None

    # 定义多尺度范围
    scales = np.linspace(0.2, 1.0, 20)[::-1]

    for scale in scales:
        # 缩放模板图像
        resized_template = cv2.resize(template_gray, (int(tW * scale), int(tH * scale)))
        rH, rW = resized_template.shape[:2]

        # 如果缩放后的模板尺寸大于地图图像，则跳过
        if rH > map_gray.shape[0] or rW > map_gray.shape[1]:
            continue

        # 使用模板匹配
        result = cv2.matchTemplate(map_gray, resized_template, cv2.TM_CCOEFF_NORMED)

        # 获取匹配结果中大于阈值的位置
        loc = np.where(result >= threshold)

        # 如果找到了匹配结果，更新最佳匹配
        if len(loc[0]) > 0:
            best_match = result
            best_scale = scale
            best_loc = loc
            break

    # 如果未找到匹配结果，返回None
    if best_match is None:
        print("未找到匹配区域")
        return None

    # 创建与地图图像相同大小的空白图像用于绘制匹配结果
    match_result = np.zeros_like(map_gray)

    # 在匹配位置绘制矩形
    for pt in zip(*best_loc[::-1]):
        cv2.rectangle(match_result, pt, (pt[0] + int(tW * best_scale), pt[1] + int(tH * best_scale)), 255, -1)

    # 二值化匹配结果
    _, binary_match_result = cv2.threshold(match_result, 1, 255, cv2.THRESH_BINARY)

    return binary_match_result

# 示例使用
binary_result = multi_scale_template_matching('origin.jpg', 'template_00.jpg')

# 显示结果
if binary_result is not None:
    cv2.imshow('Binary Match Result', binary_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
