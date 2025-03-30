import os
import shutil
import pathlib
import pycolmap
import numpy as np
import cv2

"""
TODO 1. find default focal length of pyclomap
TODO 2. change focal length option to my camera's
"""

folder = pathlib.Path("benchmark")
data_path = pathlib.Path("tiny_nerf_data.npz")
output_path = pathlib.Path("output")
image_dir = pathlib.Path("images")# 재구성 결과 폴더

if __name__=="__main__":
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        print(f"{output_path} is removed.")
    
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
        print(f"{image_dir} is removed.")

    data = np.load(data_path)

    images = data['images']  # shape: (N, H, W, 3)
    N = images.shape[0]

    # 만약 images의 dtype이 float라면 [0,1] 범위로 가정 후 uint8로 변환
    if images.dtype != np.uint8:
        images = (np.clip(images, 0, 1) * 255).astype(np.uint8)
    
    os.mkdir(image_dir)
    print(f"{image_dir} is created.")
    
    # save image
    for i in range(N):
        # 이미지 배열을 가져옴
        img = images[i]
        # 만약 이미지 색상 순서가 RGB라면, OpenCV는 BGR로 사용하므로 변환 필요
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 파일 이름 지정 (예: image_000.png, image_001.png, ...)
        filename = os.path.join(image_dir, f"image_{i:03d}.png")
        
        # 이미지 파일 저장
        cv2.imwrite(filename, img_bgr)

    print(f"Saving images done.")

    # 출력 폴더 생성
    output_path.mkdir(exist_ok=True)
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"

    # Sparse reconstruction
    print("Extracting features...")
    pycolmap.extract_features(database_path, image_dir)
    print("Matching features...")
    pycolmap.match_exhaustive(database_path)
    print("Running incremental mapping...")
    # incremental_mapping은 sparse reconstruction을 수행하며, 여러 맵(reconstructions)을 반환합니다.
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    # 첫 번째 reconstruction 결과를 저장합니다.
    maps[0].write(output_path)
