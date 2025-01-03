import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

# 1. 定义ResNet50的特征提取器，保留高分辨率空间信息
class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # 去掉全局池化和全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)

# 2. 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet50Backbone().to(device)
model.eval()

# 3. 定义输入变换，分别针对 legend 和 map
legend_transform = T.Compose([
    T.Resize((68, 41)),  # 使用原始比例缩放
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

map_transform = T.Compose([
    T.Resize((1435, 1320)),  # 使用原始比例缩放
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. 读取图像并提取特征
def get_feature_map(img_path, transform, model):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)  # 增加 batch 维度
    with torch.no_grad():
        feat = model(x)  # 提取特征
    return feat.squeeze(0).cpu().numpy()  # 移除 batch 维度

# 获取原始图像尺寸
map_img = Image.open("map.jpg")
map_width, map_height = map_img.size

legend_feat = get_feature_map("legend.jpg", legend_transform, model)  # [2048, hL, wL]
map_feat = get_feature_map("map.jpg", map_transform, model)  # [2048, hM, wM]

# 5. 滑动窗口相似度计算
def sliding_window_similarity(map_feat, legend_feat):
    c, hM, wM = map_feat.shape
    _, hL, wL = legend_feat.shape

    sim_map = np.zeros((hM - hL + 1, wM - wL + 1))
    for i in range(hM - hL + 1):
        for j in range(wM - wL + 1):
            # 提取滑动窗口
            window = map_feat[:, i:i + hL, j:j + wL]
            # 计算余弦相似度
            sim = np.sum(window * legend_feat) / (
                np.linalg.norm(window) * np.linalg.norm(legend_feat) + 1e-8
            )
            sim_map[i, j] = sim
    return sim_map

sim_map = sliding_window_similarity(map_feat, legend_feat)

# 6. 可视化相似度热图
sim_map_norm = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
sim_map_resized = cv2.resize(sim_map_norm, (map_width, map_height), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("similarity_heatmap.png", (sim_map_resized * 255).astype(np.uint8))

# 7. 根据阈值生成二值掩码
threshold = 0.6
mask = (sim_map_resized > threshold).astype(np.uint8) * 255
cv2.imwrite("prediction_mask.png", mask)

# 8. 分别将热力图和二值化掩码叠加到 map.jpg
def overlay_heatmap_on_map(map_path, heatmap, output_path):
    map_img = cv2.imread(map_path)
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(map_img, 0.5, heatmap_colored, 0.5, 0)
    cv2.imwrite(output_path, overlay)

def overlay_mask_on_map(map_path, binary_mask, output_path):
    map_img = cv2.imread(map_path)
    binary_mask_colored = cv2.merge([binary_mask, binary_mask, binary_mask])
    overlay = cv2.addWeighted(map_img, 0.5, binary_mask_colored, 0.5, 0)
    cv2.imwrite(output_path, overlay)

overlay_heatmap_on_map("map.jpg", sim_map_resized, "heatmap_overlay.png")
overlay_mask_on_map("map.jpg", mask, "mask_overlay.png")

print("相似度热图、二值化掩码及分别叠加图已保存，尺寸与 map.jpg 保持一致！")
