import torch
import torchvision.transforms as T
import open_clip
from PIL import Image
import torch.nn.functional as F
import numpy as np
import base64

device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", device=device)
model.eval()

img_path = 'sample_data/gt/00032.png'
original_img = Image.open(img_path).convert("RGB")
W_orig, H_orig = original_img.size

# Let's resize image so that crop extraction is faster, say max 1000 px
scale = 1000 / max(W_orig, H_orig)
if scale < 1.0:
    new_w, new_h = int(W_orig * scale), int(H_orig * scale)
    img_resized = original_img.resize((new_w, new_h), Image.BILINEAR)
else:
    img_resized = original_img
    new_w, new_h = W_orig, H_orig

crop_size = 224
stride = 112

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
        crops.append(crop)
        coords.append((x, y, x+crop_size, y+crop_size))

# handle right/bottom edges if needed... let's just keep it simple

if len(crops) == 0:
    # Image smaller than 224, just use original
    crops.append(T.Resize((224,224))(img_tensor))
    coords.append((0, 0, new_w, new_h))

crops = torch.stack(crops).to(device)

print(f"Extracting features for {len(crops)} crops...")
with torch.no_grad():
    # Batch process in chunks to avoid memory issues
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
feature_map = feature_map / feature_map.norm(dim=-1, keepdim=True)

# Resize to H_orig, W_orig
feature_map = feature_map.permute(2, 0, 1).unsqueeze(0)
feature_map = F.interpolate(feature_map, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
feature_512 = feature_map.squeeze(0).permute(1, 2, 0).reshape(-1, 512)

# Query
tokenizer = open_clip.get_tokenizer('ViT-B-16')
negatives = ["object", "things", "stuff", "texture"]
phrases = ['car'] + negatives
text_tokens = tokenizer(phrases).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

embed = feature_512
p = text_features
output = torch.mm(embed, p.T)
positive_vals = output[..., 0:1]
negative_vals = output[..., 1:]
repeated_pos = positive_vals.repeat(1, len(negatives))
sims = torch.stack((repeated_pos, negative_vals), dim=-1)
softmax = torch.softmax(10 * sims, dim=-1)
best_id = softmax[..., 0].argmin(dim=1)
relevancy = torch.where(best_id == 0, sims[..., 0, 0], sims[..., 0, 1])

relevancy = relevancy.view(H_orig, W_orig).cpu().numpy()
# compute max and min inside the script to see
print(f"Relevancy min: {relevancy.min():.3f}, max: {relevancy.max():.3f}")

# Save debug image
import cv2
relevancy = (relevancy - relevancy.min()) / (relevancy.max() - relevancy.min() + 1e-5)
heatmap = cv2.applyColorMap(np.uint8(255 * relevancy), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
img_orig_np = np.array(original_img)
alpha = 0.5
out = cv2.addWeighted(img_orig_np, alpha, heatmap, 1 - alpha, 0)
Image.fromarray(out).save('debug_crop_car.png')
print('Saved debug_crop_car.png')

