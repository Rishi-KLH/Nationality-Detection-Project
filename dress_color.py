import cv2
import numpy as np
from sklearn.cluster import KMeans

def color_name(rgb):
    r, g, b = rgb
    if r < 60 and g < 60 and b < 60:
        return "Black"
    if r > 200 and g > 200 and b > 200:
        return "White"
    if r > g and r > b:
        return "Red"
    if g > r and g > b:
        return "Green"
    return "Blue"

def get_dress_color(img_np):
    h, w, _ = img_np.shape
    roi = img_np[int(h*0.5):h, :]
    roi = cv2.resize(roi, (200,200))

    pixels = roi.reshape(-1,3)
    kmeans = KMeans(n_clusters=3, n_init=10).fit(pixels)
    dominant = kmeans.cluster_centers_[kmeans.labels_[0]]

    return color_name(dominant.astype(int))