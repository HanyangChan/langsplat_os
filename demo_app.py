import os
import torch
import numpy as np
import clip
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn

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

import open_clip

# Load CLIP model (Must use the exact same one LangSplat used!)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading OpenCLIP model on {device}...")
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", device=device)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-16")

def process_query(text_query, threshold, frame_id, decoder_path):
    base_dir = "sample_data"
    image_path = os.path.join(base_dir, "gt", f"{frame_id}.png")
    feature_npy_path = os.path.join(base_dir, "renders_npy", f"{frame_id}.npy")

    if not os.path.exists(feature_npy_path) or not os.path.exists(image_path) or not os.path.exists(decoder_path):
        return None, "Error: Required feature, image, or decoder file not found."

    # 1. Load Original Image
    original_img = Image.open(image_path).convert("RGB")
    
    # 2. Load 3-dim Rendered Feature
    feature_3d = np.load(feature_npy_path) # shape: (H, W, 3)
    H, W, _ = feature_3d.shape
    feature_3d = torch.tensor(feature_3d, dtype=torch.float32).to(device)
    feature_3d = feature_3d.view(-1, 3) # shape: (H*W, 3)

    # 3. Load Decoder and get 512-dim CLIP feature
    # The decoder dims used in LangSplat: 3 -> 16 -> 32 -> 64 -> 128 -> 256 -> 256 -> 512
    decoder = Decoder([3, 16, 32, 64, 128, 256, 256, 512]).to(device)
    
    ckpt = torch.load(decoder_path, map_location=device)
    new_state_dict = {}
    for k, v in ckpt.items():
        if k.startswith('decoder.'):
            # map 'decoder.0.weight' -> 'network.0.weight'
            new_key = k.replace('decoder.', 'network.')
            new_state_dict[new_key] = v
            
    decoder.load_state_dict(new_state_dict)
    decoder.eval()

    with torch.no_grad():
        feature_512 = decoder(feature_3d)
        feature_512 = feature_512 / feature_512.norm(dim=-1, keepdim=True) # Normalize

    # 4. Process Text Query and Negatives
    negatives = ["object", "things", "stuff", "texture"]
    phrases = [text_query] + negatives
    text_tokens = tokenizer(phrases).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 5. Compute Relevancy (LERF style)
    similarity = (feature_512 @ text_features.T) # (H*W, 5)
    
    positive_vals = similarity[:, 0:1] # (H*W, 1)
    negative_vals = similarity[:, 1:]  # (H*W, 4)
    repeated_pos = positive_vals.repeat(1, len(negatives))
    
    sims = torch.stack((repeated_pos, negative_vals), dim=-1) # (H*W, 4, 2)
    softmax = torch.softmax(10 * sims, dim=-1) # (H*W, 4, 2)
    best_id = softmax[..., 0].argmin(dim=1)
    
    relevancy = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(negatives), 2))[:, 0, 0]
    similarity_map = relevancy.view(H, W).cpu().numpy()

    # 6. Apply colormap (LERF composite style)
    relevancy_map = relevancy.view(H, W).cpu().numpy()
    
    # Scale p_i for the colormap
    p_i = np.clip(relevancy_map - threshold, 0, 1)
    max_p = p_i.max()
    if max_p > 0:
        p_i_scaled = p_i / max_p
    else:
        p_i_scaled = p_i
        
    cmap = plt.get_cmap('turbo')
    heatmap = cmap(p_i_scaled)[..., :3] # RGB
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Create the composite image
    original_img_np = np.array(original_img)
    if original_img_np.shape[:2] != (H, W):
        original_img_np = np.array(original_img.resize((W, H)))
        
    mask = relevancy_map < threshold
    
    composited = heatmap.copy()
    composited[mask] = original_img_np[mask] * 0.3
    
    blended = Image.fromarray(composited.astype(np.uint8))

    return blended, "Success"

def gradio_interface(text_query, threshold, frame_id):
    decoder_path = os.path.join("sample_data", "decoder.pth")
    if not os.path.exists(decoder_path):
        return None
        
    blended_img, msg = process_query(text_query, threshold, frame_id, decoder_path)
    return blended_img

if __name__ == "__main__":
    frames = [f"{i:05d}" for i in range(31)] # 00000 to 00030
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.Textbox(lines=1, placeholder="Enter text query here (e.g., 'a red apple')..."),
            gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Relevancy Threshold"),
            gr.Dropdown(choices=frames, value="00017", label="Select Viewpoint (Frame ID)")
        ],
        outputs=gr.Image(type="pil"),
        title="LangSplat Interactive Demo",
        description="Enter a text query to search for the object in the 3D rendered scene. Adjust the threshold slider if your object is not highlighted (some objects have naturally lower CLIP confidence)."
    )
    iface.launch(share=False)
