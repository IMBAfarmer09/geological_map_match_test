import numpy as np
import cv2

def overlay_on_map(map_path, overlay_data, output_path, colormap=cv2.COLORMAP_JET):
    map_img = cv2.imread(map_path)
    if colormap:
        overlay_colored = cv2.applyColorMap((overlay_data * 255).astype(np.uint8), colormap)
    else:
        overlay_colored = cv2.merge([overlay_data, overlay_data, overlay_data])
    overlay = cv2.addWeighted(map_img, 0.5, overlay_colored, 0.5, 0)
    cv2.imwrite(output_path, overlay)

overlay_on_map("map.jpg", sim_map_resized, "heatmap_overlay.png", colormap=None)
overlay_on_map("map.jpg", mask, "mask_overlay.png", colormap=None)

print("相似度热图和掩码已更新，并分别叠加到原图。")