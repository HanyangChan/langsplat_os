import os
import sys
import base64
from PIL import Image
import io

os.chdir('backend')
sys.path.append('.')
from langsplat_inference import process_query

frames = []
for i in range(32, 42):
    frame_id = f"{i:05d}"
    print(f"Processing frame {frame_id}...")
    img_b64, msg = process_query('car', 0.25, frame_id)
    if img_b64:
        img_data = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        frames.append(img)
    else:
        print(f"Failed on {frame_id}: {msg}")

if frames:
    out_path = '../langsplat_driving_car.gif'
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=150, # ms per frame
        loop=0
    )
    print(f"Saved GIF to {out_path}")
else:
    print("No frames processed.")
