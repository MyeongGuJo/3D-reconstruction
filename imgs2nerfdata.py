import os
import shutil
import pathlib
import numpy as np
import pycolmap
import cv2

def qvec2rotmat(q):
    """
    q: [x, y, z, w] 순서의 쿼터니언.
    3×3 회전 행렬 R을 반환합니다.
    """
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2),  2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])
    return R

def run_colmap(image_dir, database_path, output_path):
    """
    주어진 이미지 폴더에서 COLMAP의 sparse 재구성을 실행합니다.
    결과는 output_path에 저장됩니다.
    """
    # 기존 database와 output 폴더 삭제 (있다면)
    if os.path.exists(database_path):
        os.remove(database_path)
        print(f"{database_path} removed.")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        print(f"{output_path} removed.")
    os.makedirs(output_path, exist_ok=True)
    
    print("Extracting features...")
    pycolmap.extract_features(database_path, image_dir)
    print("Matching features (exhaustive)...")
    pycolmap.match_exhaustive(database_path)
    print("Running incremental mapping...")
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    # 첫 번째 재구성 결과 사용
    reconstruction = maps[0]
    reconstruction.write(output_path)
    return reconstruction

def extract_poses_from_reconstruction(reconstruction):
    """
    COLMAP 재구성 결과에서 각 이미지의 camera-to-world 포즈(4×4 행렬)를 추출합니다.
    COLMAP은 world-to-camera 정보를 저장하므로, 역행렬을 취하여 camera-to-world 행렬을 얻습니다.
    """
    poses = []
    image_names = []
    for _, image in reconstruction.images.items():
        # image.cam_from_world는 world-to-camera 변환 정보를 제공
        rigid = image.cam_from_world
        q = rigid.rotation.quat
        t = np.array(rigid.translation)
        T_wc = np.eye(4)
        T_wc[:3, :3] = qvec2rotmat(q)
        T_wc[:3, 3] = t
        # camera-to-world = inv(T_wc)
        T = np.linalg.inv(T_wc)
        poses.append(T)
        image_names.append(image.name)
    poses = np.stack(poses, axis=0)
    return image_names, poses

def apply_z_flip(poses):
    """
    NeRF에서는 카메라가 -z 방향을 바라보도록 하는 경우가 많습니다.
    COLMAP 결과가 +z 방향이라면, 각 포즈에 대해 T_flip = diag([1, 1, -1, 1])를 적용합니다.
    """
    T_flip = np.diag([1, 1, -1, 1])
    flipped = np.array([pose @ T_flip for pose in poses])
    return flipped

def recenter_poses(poses):
    """
    모든 포즈의 카메라 중심(translation)들을 평균으로 빼서 중앙화합니다.
    """
    centers = poses[:, :3, 3]
    center_mean = centers.mean(axis=0)
    poses_recentered = poses.copy()
    poses_recentered[:, :3, 3] -= center_mean
    return poses_recentered, center_mean

def scale_poses(poses, desired_scale=1.0):
    """
    각 포즈의 카메라 중심이 desired_scale 내에 들어오도록 스케일 보정합니다.
    여기서는 카메라 중심의 최대 거리를 기준으로 보정합니다.
    """
    centers = poses[:, :3, 3]
    max_dist = np.max(np.linalg.norm(centers, axis=1))
    scale_factor = desired_scale / max_dist
    poses_scaled = poses.copy()
    poses_scaled[:, :3, 3] *= scale_factor
    return poses_scaled, scale_factor

def load_images_from_folder(folder_path):
    """
    폴더 내의 모든 PNG 파일을 정렬된 순서대로 읽어 (N, H, W, 3) RGB numpy array로 반환합니다.
    """
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
    images = []
    for fname in image_files:
        path = os.path.join(folder_path, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load image: {fname}")
            continue
        # OpenCV는 BGR이므로 RGB로 변환
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
    return np.array(images)

def compute_focal_from_image(images, focal_scale=1.2):
    """
    첫 번째 이미지의 크기를 기준으로 focal을 계산합니다.
    focal = focal_scale * max(height, width)
    """
    if images.shape[0] == 0:
        raise ValueError("No images found!")
    H, W = images[0].shape[:2]
    focal = focal_scale * max(H, W)
    return focal

def create_nerf_data(image_dir, database_path, output_path, output_npz="nerf_data.npz", desired_scale=1.0):
    """
    주어진 이미지 폴더에서 COLMAP을 실행하여 포즈를 추출하고, 
    - z축 반전(T_flip) → recenter → scale 보정을 거친 후,
    이미지, 4×4 포즈, focal 정보를 NeRF 학습용 데이터 형식(딕셔너리)에 맞게 구성하여 npz 파일로 저장합니다.
    
    결과 딕셔너리:
      data['images']: (N, H, W, 3) RGB 이미지 배열
      data['poses']: (M, 4, 4) 카메라 포즈 배열 (NeRF 형식, -z forward, recentered, scaled)
      data['focal']: scalar, focal 값
    """
    # COLMAP 재구성 실행
    reconstruction = run_colmap(image_dir, database_path, output_path)
    
    # COLMAP 재구성 결과에서 포즈 추출 (이미지별로 등록된 포즈)
    image_names, poses = extract_poses_from_reconstruction(reconstruction)
    print("Extracted poses shape:", poses.shape)
    
    # NeRF에서 일반적으로 사용하는 좌표계: 카메라가 -z 방향을 바라보도록 변환
    poses = apply_z_flip(poses)
    
    # recenter 및 scale 보정 (desired_scale은 보통 1.0 등으로 설정)
    poses, center_mean = recenter_poses(poses)
    poses, scale_factor = scale_poses(poses, desired_scale=desired_scale)
    print("Poses recentered and scaled.")
    
    # 이미지 로드
    images = load_images_from_folder(image_dir)
    print(f"Loaded {images.shape[0]} images from {image_dir}.")
    
    # focal 계산 (첫 번째 이미지 기준)
    focal = compute_focal_from_image(images)
    print("Computed focal:", focal)
    
    # NeRF 데이터 딕셔너리 구성 (COLMAP에 의해 등록된 이미지 수와 실제 이미지 수가 다를 수 있음에 유의)
    data = {
        'images': images,   # (N, H, W, 3) 전체 이미지 배열
        'poses': poses,     # (M, 4, 4) 추출 및 변환된 카메라 포즈
        'focal': focal      # scalar focal 값
    }
    
    # npz 파일로 저장
    np.savez(output_npz, **data)
    print(f"Saved NeRF data to {output_npz}")

if __name__ == "__main__":
    # 경로 설정 (필요에 따라 수정)
    image_dir = "images"            # 입력 이미지 폴더
    database_path = "database.db"   # COLMAP 데이터베이스 파일 경로
    output_path = "colmap_output"   # COLMAP 결과 저장 폴더
    output_npz = "nerf_data.npz"      # 최종 npz 파일 이름
    
    create_nerf_data(image_dir, database_path, output_path, output_npz, desired_scale=1.0)
