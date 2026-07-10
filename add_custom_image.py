import os
import numpy as np
from PIL import Image

image_path = "sample_data/gt/00030.png"
feature_npy_path = "sample_data/renders_npy/00030.npy"

if not os.path.exists(image_path):
    print(f"Error: {image_path} 가 존재하지 않습니다. 먼저 이미지를 이 위치에 저장해주세요.")
else:
    # Load the image to get its dimensions
    img = Image.open(image_path)
    W, H = img.size
    print(f"이미지 크기 확인됨: {W}x{H}")

    # Generate a dummy feature map (H, W, 3) 
    # Using random noise or copying existing feature
    print("더미 3D 특징(.npy) 파일을 생성합니다...")
    dummy_feature = np.zeros((H, W, 3), dtype=np.float32)
    
    os.makedirs(os.path.dirname(feature_npy_path), exist_ok=True)
    np.save(feature_npy_path, dummy_feature)
    print(f"완료! {feature_npy_path} 가 생성되었습니다. 이제 프레임 ID '00030'을 사용하여 테스트하실 수 있습니다.")
