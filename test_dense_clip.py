import torch
import open_clip
from PIL import Image
import torch.nn.functional as F

device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", device=device)
model.eval()

img = Image.new("RGB", (448, 448))
img_tensor = preprocess(img).unsqueeze(0)
# preprocess resizes to 224x224 by default!
# We can bypass preprocess:
import torchvision.transforms as T
my_preprocess = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])
img_tensor = my_preprocess(img).unsqueeze(0)

x = model.visual.conv1(img_tensor)  # [1, 768, 28, 28]
print(x.shape)
x = x.reshape(x.shape[0], x.shape[1], -1)  # [1, 768, 784]
x = x.permute(0, 2, 1)  # [1, 784, 768]

# interpolate pos embedding
pos_embed = model.visual.positional_embedding.to(x.dtype)
cls_pos = pos_embed[0:1, :]
patch_pos = pos_embed[1:, :] # [196, 768]
patch_pos = patch_pos.reshape(1, 14, 14, 768).permute(0, 3, 1, 2) # [1, 768, 14, 14]
patch_pos = F.interpolate(patch_pos, size=(28, 28), mode='bicubic', align_corners=False)
patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(784, 768)
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

print("Patch tokens shape:", patch_tokens.shape)
