import numpy as np
import cv2
import yaml
from skimage import measure
import matplotlib.pyplot as plt

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

def generate_mask_from_similarity(sim_map, threshold, min_area):
    binary_map = (sim_map > threshold).astype(np.uint8)
    labeled_map = measure.label(binary_map, connectivity=2)
    properties = measure.regionprops(labeled_map)

    mask = np.zeros_like(sim_map, dtype=np.uint8)
    for prop in properties:
        if prop.area > min_area:
            coords = prop.coords
            mask[coords[:, 0], coords[:, 1]] = 255

    return mask

def overlay_on_map_bw(map_path, overlay_data, output_path):
    map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    overlay = cv2.addWeighted(map_img, 0.5, overlay_data.astype(np.uint8), 0.5, 0)
    cv2.imwrite(output_path, overlay)

def analyze_similarity_distribution(sim_map, output_path):
    flat_sim_map = sim_map.flatten()
    bins = np.arange(0, 1.05, 0.05)
    hist, bin_edges = np.histogram(flat_sim_map, bins=bins)
    proportions = hist / flat_sim_map.size

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
    plt.savefig(output_path)
    plt.show()

# 加载热图数据
sim_map = np.load("sim_map.npy")

# 生成掩码并保存
mask = generate_mask_from_similarity(
    sim_map,
    config["parameters"]["threshold"],
    config["parameters"]["min_area"]
)
cv2.imwrite(config["output"]["prediction_mask"], mask)

# 渲染叠加图
overlay_on_map_bw(config["inputs"]["map_image"], (sim_map * 255).astype(np.uint8), config["output"]["heatmap_overlay"])
overlay_on_map_bw(config["inputs"]["map_image"], mask, config["output"]["mask_overlay"])

# 分析分布
analyze_similarity_distribution(sim_map, config["output"]["similarity_distribution"])
