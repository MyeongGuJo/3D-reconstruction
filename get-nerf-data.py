import os
import cv2
import numpy as np

def load_images_from_folder(folder_path):
    """
    폴더 내의 모든 *.png 파일을 정렬된 순서대로 읽어 (N, H, W, 3) numpy array로 반환합니다.
    """
    # 파일 이름 정렬 (예: image_000.png, image_001.png, ...)
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
    images = []
    for fname in image_files:
        path = os.path.join(folder_path, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load image: {fname}")
            continue
        # OpenCV는 기본적으로 BGR 순서이므로, RGB로 변환
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
    return np.array(images)

def compute_focal_from_image(images, focal_scale=1.2):
    """
    이미지 배열의 첫 이미지의 크기를 이용하여 focal 값을 산출합니다.
    focal = focal_scale * max(H, W)
    """
    if images.shape[0] == 0:
        raise ValueError("No images found!")
    H, W = images[0].shape[:2]
    focal = focal_scale * max(H, W)
    return focal

def create_llff_data(image_folder, poses_bounds_path, output_npz="data.npz"):
    """
    image_folder: 모든 이미지가 있는 폴더 (예: "images")
    poses_bounds_path: LLFF 형식의 poses_bounds.npy 파일 경로 (각 행에 17개 값)
    
    위 두 데이터를 사용하여 data 딕셔너리를 생성하고,
    data['images'], data['poses'], data['focal']로 저장 후 npz 파일로 출력합니다.
    """
    # 이미지 읽기
    images = load_images_from_folder(image_folder)
    print(f"Loaded {images.shape[0]} images from {image_folder}.")
    
    # focal 계산 (첫 이미지 기준)
    focal = compute_focal_from_image(images)
    print(f"Computed focal: {focal}")
    
    # poses_bounds 로드 (npy 파일)
    poses_bounds = np.load(poses_bounds_path)  # shape: (N, 17)
    print(f"Loaded poses_bounds with shape: {poses_bounds.shape}")
    
    # 딕셔너리 구성 (원하는 키로 저장)
    data = {
        'images': images,       # (N, H, W, 3) RGB 순서
        'poses': poses_bounds,  # LLFF 형식의 poses + bounds, (N, 17)
        'focal': focal          # focal 값 (scalar)
    }
    
    # npz 파일로 저장
    np.savez(output_npz, **data)
    print(f"Saved data to {output_npz}")

if __name__ == "__main__":
    # 사용자에 맞게 경로 수정
    image_folder = "images"             # 이미지가 들어있는 폴더
    poses_bounds_path = "poses_bounds.npy"  # LLFF 형식의 포즈 파일
    output_npz = "nerf_data.npz"
    
    create_llff_data(image_folder, poses_bounds_path, output_npz)
