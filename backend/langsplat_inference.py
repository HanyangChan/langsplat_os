import os
import torch
import numpy as np
import open_clip
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import io
import base64

# Define the Decoder model architecture matching LangSplat's autoencoder
class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading OpenCLIP model on {device}...")
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", device=device)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-16")

# Pre-load decoder to save time during requests if possible, 
# but original code loads it per frame. Let's keep it per request or load once if path is constant.
decoder_path_default = os.path.join("..", "sample_data", "decoder.pth")
decoder = None
if os.path.exists(decoder_path_default):
    decoder = Decoder([3, 16, 32, 64, 128, 256, 256, 512]).to(device)
    ckpt = torch.load(decoder_path_default, map_location=device, weights_only=True)
    new_state_dict = {}
    for k, v in ckpt.items():
        if k.startswith('decoder.'):
            new_key = k.replace('decoder.', 'network.')
            new_state_dict[new_key] = v
    decoder.load_state_dict(new_state_dict)
    decoder.eval()
    print("Pre-loaded LangSplat Decoder.")

def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def process_query(text_query: str, threshold: float, frame_id: str):
    base_dir = os.path.join("..", "sample_data")
    image_path = os.path.join(base_dir, "gt", f"{frame_id}.png")
    feature_npy_path = os.path.join(base_dir, "renders_npy", f"{frame_id}.npy")

    if not os.path.exists(feature_npy_path) or not os.path.exists(image_path):
        return None, "Error: Required feature or image not found."
    if decoder is None:
        return None, "Error: Decoder not loaded."

    # 1. Load Original Image
    original_img = Image.open(image_path).convert("RGB")
    
    # 2. Check for pre-computed 512-dim feature map (for sharp 2D fallbacks)
    feature_512_path = feature_npy_path.replace('.npy', '_512.npy')
    if os.path.exists(feature_512_path):
        feature_512_map = np.load(feature_512_path) # [H, W, 512]
        H, W = feature_512_map.shape[:2]
        feature_512 = torch.tensor(feature_512_map, dtype=torch.float32).view(-1, 512).to(device)
    else:
        # Load 3-dim Rendered Feature
        feature_3d = np.load(feature_npy_path) # shape: (H, W, 3)
        if feature_3d.max() == 0 and feature_3d.min() == 0:
            import torchvision.transforms as T
            import torch.nn.functional as F
            W_orig, H_orig = original_img.size
            target_size = 448
            grid_size = target_size // 16
            my_preprocess = T.Compose([
                T.Resize((target_size, target_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
            img_tensor = my_preprocess(original_img).unsqueeze(0).to(device)
            with torch.no_grad():
                x = model.visual.conv1(img_tensor)
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)
                pos_embed = model.visual.positional_embedding.to(x.dtype)
                cls_pos = pos_embed[0:1, :]
                patch_pos = pos_embed[1:, :].reshape(1, 14, 14, 768).permute(0, 3, 1, 2)
                patch_pos = F.interpolate(patch_pos, size=(grid_size, grid_size), mode='bicubic', align_corners=False)
                patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(grid_size**2, 768)
                pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
                x = torch.cat([model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
                x = x + pos_embed
                x = model.visual.ln_pre(x)
                x = x.permute(1, 0, 2)
                x = model.visual.transformer(x)
                x = x.permute(1, 0, 2)
                patch_tokens = x[:, 1:, :]
                patch_tokens = model.visual.ln_post(patch_tokens)
                if model.visual.proj is not None:
                    patch_tokens = patch_tokens @ model.visual.proj
                feature_512 = patch_tokens.squeeze(0)
                feature_512 = feature_512 / feature_512.norm(dim=-1, keepdim=True)
                feature_512_map = feature_512.reshape(grid_size, grid_size, 512).permute(2, 0, 1).unsqueeze(0)
                feature_512_map = F.interpolate(feature_512_map, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
                feature_512 = feature_512_map.squeeze(0).permute(1, 2, 0).reshape(-1, 512)
                feature_512 = feature_512 / feature_512.norm(dim=-1, keepdim=True)
            H, W = H_orig, W_orig
        else:
            H, W, _ = feature_3d.shape
            feature_3d_tensor = torch.tensor(feature_3d, dtype=torch.float32).to(device)
            feature_3d_tensor = feature_3d_tensor.view(-1, 3) # shape: (H*W, 3)

            # 3. Get 512-dim CLIP feature from decoder
            with torch.no_grad():
                feature_512 = decoder(feature_3d_tensor)
                feature_512 = feature_512 / feature_512.norm(dim=-1, keepdim=True) # Normalize

    # 4. Process Text Query and Negatives
    negatives = ["object", "things", "stuff", "texture"]
    phrases = [text_query] + negatives
    text_tokens = tokenizer(phrases).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 5. Compute Relevancy
    similarity = (feature_512 @ text_features.T) # (H*W, 5)
    
    positive_vals = similarity[:, 0:1] # (H*W, 1)
    negative_vals = similarity[:, 1:]  # (H*W, 4)
    repeated_pos = positive_vals.repeat(1, len(negatives))
    
    sims = torch.stack((repeated_pos, negative_vals), dim=-1) # (H*W, 4, 2)
    softmax = torch.softmax(10 * sims, dim=-1) # (H*W, 4, 2)
    best_id = softmax[..., 0].argmin(dim=1)
    
    relevancy = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(negatives), 2))[:, 0, 0]
    
    # 6. Apply colormap
    relevancy_map = relevancy.view(H, W).cpu().numpy()
    
    p_i = np.clip(relevancy_map - threshold, 0, 1)
    max_p = p_i.max()
    if max_p > 0:
        p_i_scaled = p_i / max_p
    else:
        p_i_scaled = p_i
        
    cmap = plt.get_cmap('turbo')
    heatmap = cmap(p_i_scaled)[..., :3] # RGB
    heatmap = (heatmap * 255).astype(np.uint8)
    
    original_img_np = np.array(original_img)
    if original_img_np.shape[:2] != (H, W):
        original_img_np = np.array(original_img.resize((W, H)))
        
    mask = relevancy_map < threshold
    
    composited = heatmap.copy()
    composited[mask] = original_img_np[mask] * 0.3
    
    blended = Image.fromarray(composited.astype(np.uint8))

    return image_to_base64(blended), "Success"
