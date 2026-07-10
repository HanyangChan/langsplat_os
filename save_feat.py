import numpy as np
from PIL import Image

f = np.load("sample_data/sample_feature_3d.npy")
f = (f - f.min()) / (f.max() - f.min())
img = Image.fromarray((f * 255).astype(np.uint8))
img.save("feat_rgb.png")
