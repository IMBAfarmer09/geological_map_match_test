import cv2
import numpy as np

# 1. 读取图像
# map_path 为完整地图的路径
# legend_path 为给定图例（例如红底+黑色加号纹理）的图像路径
map_path = 'origin.jpg'
legend_path = 'template_00.jpg'

# 读入原始地图图像
img_map = cv2.imread(map_path)
# 读入图例图像
img_legend = cv2.imread(legend_path)

# 如果图像读取失败，可以抛出异常或给出提示
if img_map is None:
    raise FileNotFoundError(f'无法读取地图图像: {map_path}')
if img_legend is None:
    raise FileNotFoundError(f'无法读取图例图像: {legend_path}')

# 转换为灰度图像（模板匹配通常先做灰度匹配）
gray_map = cv2.cvtColor(img_map, cv2.COLOR_BGR2GRAY)
gray_legend = cv2.cvtColor(img_legend, cv2.COLOR_BGR2GRAY)

# 2. 模板匹配
# 这里使用 TM_CCOEFF_NORMED 方法
res = cv2.matchTemplate(gray_map, gray_legend, cv2.TM_CCOEFF_NORMED)

# 设置一个匹配阈值（需要根据实际情况调整）
# 阈值越高说明对模板相似程度要求越高
threshold = 0.6

# 找到所有大于等于阈值的位置
loc = np.where(res >= threshold)

# 3. 构建一个与地图同尺寸的二值化空图，用于最终可视化
binary_mask = np.zeros_like(gray_map, dtype=np.uint8)

h_legend, w_legend = gray_legend.shape[:2]

# 将匹配到的位置在 binary_mask 上标记为 255
for pt in zip(*loc[::-1]):
    # pt 是匹配到的左上角顶点坐标
    top_left = pt
    bottom_right = (pt[0] + w_legend, pt[1] + h_legend)

    # 在二值图上，将对应模板区域内像素设置为白色
    # 也可以只标记矩形范围，或根据匹配强度对局部像素做更精细的处理
    binary_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255

# 4. 保存或显示结果
cv2.imwrite('matched_binary_mask.png', binary_mask)

# 如果需要在窗口中可视化，可以使用以下代码
cv2.imshow('Binary Mask', binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
