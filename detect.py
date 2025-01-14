import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import yaml
from PIL import Image
import numpy as np
import cv2

class ResNet50C2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )

    def forward(self, x):
        return self.features(x)

# 读取配置
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

# 加载配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet50C2Backbone().to(device)
model.eval()

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=config["parameters"]["mean"], std=config["parameters"]["std"]),
])

def get_feature_map(img_path, model):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x)
    return feat.squeeze(0).cpu().numpy()

def sliding_window_similarity(map_feat, legend_feat, stride):
    c, hM, wM = map_feat.shape
    _, hL, wL = legend_feat.shape
    output_h = (hM - hL) // stride + 1
    output_w = (wM - wL) // stride + 1
    sim_map = np.zeros((output_h, output_w))

    for i in range(0, hM - hL + 1, stride):
        for j in range(0, wM - wL + 1, stride):
            window = map_feat[:, i:i + hL, j:j + wL]
            sim = np.sum(window * legend_feat) / (
                np.linalg.norm(window) * np.linalg.norm(legend_feat) + 1e-8
            )
            sim_map[i // stride, j // stride] = sim

    return sim_map

# 运行特征提取和相似度计算
legend_feat = get_feature_map(config["inputs"]["legend_image"], model)
map_feat = get_feature_map(config["inputs"]["map_image"], model)
sim_map = sliding_window_similarity(map_feat, legend_feat, config["parameters"]["stride"])

# 保存相似度热图
sim_map_resized = cv2.resize(
    sim_map,
    (Image.open(config["inputs"]["map_image"]).size),
    interpolation=cv2.INTER_LINEAR
)
cv2.imwrite(config["output"]["similarity_heatmap"], (sim_map_resized * 255).astype(np.uint8))
np.save("sim_map.npy", sim_map_resized)
