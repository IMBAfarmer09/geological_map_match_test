import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

# 1. 准备一个特征提取网络 (backbone)，这里用ResNet50举例
#   去掉最后的分类层，只保留特征层
class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # 去掉最后的全连接层 fc
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # [B, 2048, 1, 1] 最后是全局池化前的特征

    def forward(self, x):
        # x shape: [B, 3, H, W]
        # output shape: [B, 2048, something...]
        # 但resnet50里在layer4之后会有全局平均池化一下...
        # 这里也可能要做一些修改，使得我们能拿到空间分辨率更高的特征图
        return self.features(x)

# 2. 初始化
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet50Backbone().to(device)
model.eval()

# 3. 定义输入变换
transform = T.Compose([
    T.ToTensor(),
    T.Resize((512, 512)),  # 简化处理，实际可根据分辨率情况调整
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet上常用的均值方差
                std=[0.229, 0.224, 0.225])
])

# 4. 读取图例和地图图像，并提取特征
def get_feature_map(img_path, model):
    """
    读取图像并提取深度特征图（简化示例）
    """
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)  # [1,3,512,512]
    with torch.no_grad():
        feat = model(x) # [1, 2048, H', W'] or [1, 2048, 1, 1] depends on the model arch
    return feat

legend_feat = get_feature_map("legend.jpg", model)  # [1, 2048, hL, wL]
map_feat = get_feature_map("map.jpg", model)        # [1, 2048, hM, wM]

# 5. 计算“legend”的聚合特征
#   如果 feature map 只有 [1, 2048, 1, 1]，那就只有一个向量
#   如果保留更多空间分辨率，比如 [1,2048,16,16]，可以对所有像素做平均，以得到 "legend" 区域的整体特征向量
legend_feat_np = legend_feat.squeeze(0).cpu().numpy()  # [2048, hL, wL] or [2048] if global
if len(legend_feat_np.shape) == 3:
    # 做一个全局平均
    legend_vec = legend_feat_np.mean(axis=(1,2))  # [2048]
else:
    # 说明已经是 [2048] 了
    legend_vec = legend_feat_np

# 6. 计算“map”的像素级（或特征图级）相似度
map_feat_np = map_feat.squeeze(0).cpu().numpy()  # [2048, hM, wM] or [2048]
if len(map_feat_np.shape) == 3:
    c, h, w = map_feat_np.shape
    # 先把 map_feat_np reshape 为 [h*w, c]
    map_feat_flat = map_feat_np.reshape(c, -1).T  # [h*w, 2048]
    legend_vec_norm = legend_vec / (np.linalg.norm(legend_vec) + 1e-8)  # 归一化
    map_feat_norm = map_feat_flat / (np.linalg.norm(map_feat_flat, axis=1, keepdims=True) + 1e-8)
    # 计算余弦相似度
    sim = np.dot(map_feat_norm, legend_vec_norm)  # [h*w]
    # 再reshape回 [h, w]
    sim_map = sim.reshape(h, w)
else:
    # 如果是 [2048] 就只能得到一个标量相似度
    sim_map = np.dot(
        map_feat_np / (np.linalg.norm(map_feat_np) + 1e-8),
        legend_vec / (np.linalg.norm(legend_vec) + 1e-8)
    )
    # 这里就只是一维，没什么可分割的

# 7. 将相似度 sim_map 二值化
if len(sim_map.shape) == 2:
    # 取 [0,1] 范围
    sim_map_norm = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
    # 选个阈值，比如 0.6
    mask = (sim_map_norm > 0.6).astype(np.uint8)*255

    # 8. 可视化输出
    # sim_map 的大小是 (hM, wM), 取决于网络最后的特征图大小
    # 如果需要和512x512相同尺寸对齐，可以先放大到512x512再写出
    # 如果需要和原图尺寸对齐，可以再做一次插值
    mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)  # 这里根据需要修改
    cv2.imwrite("prediction_mask.png", mask_resized)
else:
    print("Warning: 只能得到一个标量相似度，无法输出分割掩码。请修改backbone以保留空间维度。")
