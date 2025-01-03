import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from skimage import measure
import matplotlib.pyplot as plt

# 定义ResNet50特征提取器，保留到C3阶段
class ResNet50C3Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # 保留到layer3的前部分，即C3阶段
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # 第一次pooling
            resnet.layer1,
            resnet.layer2  # 第二次pooling
        )

    def forward(self, x):
        return self.features(x)

# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet50C3Backbone().to(device)
model.eval()

# 定义输入变换
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 读取图像并提取特征
def get_feature_map(img_path, model):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x)
    return feat.squeeze(0).cpu().numpy()

# 获取原始图像尺寸
map_img = Image.open("map.jpg")
map_width, map_height = map_img.size

legend_feat = get_feature_map("legend.jpg", model)
map_feat = get_feature_map("map.jpg", model)

# 滑动窗口相似度计算
def sliding_window_similarity(map_feat, legend_feat, stride=1):
    c, hM, wM = map_feat.shape
    _, hL, wL = legend_feat.shape

    output_h = (hM - hL) // stride + 1
    output_w = (wM - wL) // stride + 1
    sim_map = np.zeros((output_h, output_w))

    # 遍历滑动窗口并计算相似度
    for i in range(0, hM - hL + 1, stride):
        for j in range(0, wM - wL + 1, stride):
            window = map_feat[:, i:i + hL, j:j + wL]
            sim = np.sum(window * legend_feat) / (
                np.linalg.norm(window) * np.linalg.norm(legend_feat) + 1e-8
            )
            sim_map[i // stride, j // stride] = sim

    return sim_map

sim_map = sliding_window_similarity(map_feat, legend_feat, stride=1)

# 插值回原始图像尺寸，使用平滑的上采样方法
sim_map_resized = cv2.resize(sim_map, (map_width, map_height), interpolation=cv2.INTER_LINEAR)
cv2.imwrite("similarity_heatmap.png", (sim_map_resized * 255).astype(np.uint8))

# 使用 MSER 或类似算法生成二值化掩码
def generate_mask_from_similarity(sim_map, threshold=0.5):
    # 将相似度大于阈值的区域作为候选区域
    binary_map = (sim_map > threshold).astype(np.uint8)

    # 使用连通域分析进一步处理
    labeled_map = measure.label(binary_map, connectivity=2)
    properties = measure.regionprops(labeled_map)

    # 构建掩码
    mask = np.zeros_like(sim_map, dtype=np.uint8)
    for prop in properties:
        if prop.area > 50:  # 忽略小区域
            coords = prop.coords
            mask[coords[:, 0], coords[:, 1]] = 255

    return mask

# 生成掩码
mask = generate_mask_from_similarity(sim_map_resized, threshold=0.45)
cv2.imwrite("prediction_mask.png", mask)

# 叠加热力图和掩码到原始地图
def overlay_on_map_bw(map_path, overlay_data, output_path):
    map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
    overlay = cv2.addWeighted(map_img, 0.5, overlay_data.astype(np.uint8), 0.5, 0)
    cv2.imwrite(output_path, overlay)

# 黑白图叠加
overlay_on_map_bw("map.jpg", (sim_map_resized * 255).astype(np.uint8), "heatmap_overlay_bw.png")
overlay_on_map_bw("map.jpg", mask, "mask_overlay_bw.png")

# 分析 sim_map 的数值分布
def analyze_similarity_distribution(sim_map):
    flat_sim_map = sim_map.flatten()
    bins = np.arange(0, 1.05, 0.05)  # 0.05 为最小统计单元
    hist, bin_edges = np.histogram(flat_sim_map, bins=bins)
    proportions = hist / flat_sim_map.size

    # 绘制分布图
    plt.figure(figsize=(8, 6))
    plt.bar(
        [f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}" for i in range(len(bin_edges) - 1)],
        proportions,
        width=0.8,
        color="blue",
        alpha=0.7
    )
    plt.xlabel("Similarity Range")
    plt.ylabel("Proportion")
    plt.title("Distribution of Similarity Scores in sim_map")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("similarity_distribution.png")
    plt.show()

# 调用分析函数
analyze_similarity_distribution(sim_map_resized)

print("相似度热图和掩码已更新，并以黑白形式叠加到原图，同时输出了相似度分布图。")
