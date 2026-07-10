from PIL import Image
import glob

files = [f'sample_data/gt/{i:05d}.png' for i in range(32, 42)]
frames = [Image.open(f).convert("RGB") for f in files]

if frames:
    out_path = 'langsplat_driving_orig.gif'
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=150, # ms per frame
        loop=0
    )
    print(f"Saved GIF to {out_path}")
