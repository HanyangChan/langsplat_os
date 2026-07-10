import torch
import torchvision.transforms as T
import open_clip
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os
import glob

device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", device=device)
model.eval()

img_paths = [f'sample_data/gt/{i:05d}.png' for i in range(32, 42)]

for img_path in img_paths:
    if not os.path.exists(img_path): continue
    
    original_img = Image.open(img_path).convert("RGB")
    W_orig, H_orig = original_img.size

    # Resize image so that crop extraction is faster, say max 600 px
    scale = 600 / max(W_orig, H_orig)
    if scale < 1.0:
        new_w, new_h = int(W_orig * scale), int(H_orig * scale)
        img_resized = original_img.resize((new_w, new_h), Image.BILINEAR)
    else:
        img_resized = original_img
        new_w, new_h = W_orig, H_orig

    crop_size = 112
    stride = 56

    my_preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    img_tensor = my_preprocess(img_resized) # [3, H, W]

    crops = []
    coords = []
    for y in range(0, new_h - crop_size + 1, stride):
        for x in range(0, new_w - crop_size + 1, stride):
            crop = img_tensor[:, y:y+crop_size, x:x+crop_size]
            crops.append(T.Resize((224,224))(crop))
            coords.append((x, y, x+crop_size, y+crop_size))

    if len(crops) == 0:
        crops.append(T.Resize((224,224))(img_tensor))
        coords.append((0, 0, new_w, new_h))

    crops = torch.stack(crops).to(device)

    print(f"[{img_path}] Extracting features for {len(crops)} crops...")
    with torch.no_grad():
        features = []
        chunk_size = 16
        for i in range(0, len(crops), chunk_size):
            chunk = crops[i:i+chunk_size]
            feat = model.encode_image(chunk)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat)
        features = torch.cat(features, dim=0) # [N, 512]

    # Reconstruct feature map
    feature_map = torch.zeros((new_h, new_w, 512), device=device)
    counts = torch.zeros((new_h, new_w, 1), device=device)

    for i, (x1, y1, x2, y2) in enumerate(coords):
        feature_map[y1:y2, x1:x2] += features[i].unsqueeze(0).unsqueeze(0)
        counts[y1:y2, x1:x2] += 1

    feature_map = feature_map / counts.clamp(min=1)
    feature_map = feature_map / feature_map.norm(dim=-1, keepdim=True).clamp(min=1e-5)

    # Save to _512.npy
    feature_map_np = feature_map.cpu().numpy()
    out_path = img_path.replace('gt', 'renders_npy').replace('.png', '_512.npy')
    np.save(out_path, feature_map_np)
    print(f"Saved {out_path}")

