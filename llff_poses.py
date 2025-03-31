import os
import shutil
import numpy as np
import pycolmap

def qvec2rotmat(q):
    """
    q: [x, y, z, w] 순서의 쿼터니언.
    3x3 회전 행렬 R을 반환.
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
    # 기존 출력 파일 삭제 (있다면)
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
    # 여러 재구성 중 첫 번째 결과 사용
    reconstruction = maps[0]
    reconstruction.write(output_path)  # optional: 저장
    return reconstruction

def extract_poses_from_reconstruction(reconstruction):
    """
    COLMAP 재구성 결과(reconstruction)에서 camera-to-world 포즈(4x4 행렬)를 추출합니다.
    COLMAP은 보통 world-to-camera 정보를 저장하므로, 여기서는 역행렬을 취합니다.
    """
    poses = []
    image_names = []
    for _, image in reconstruction.images.items():
        # image.cam_from_world는 world-to-camera 변환 정보
        rigid = image.cam_from_world
        q = rigid.rotation.quat
        t = np.array(rigid.translation)
        T_wc = np.eye(4)
        T_wc[:3, :3] = qvec2rotmat(q)
        T_wc[:3, 3] = t
        # camera-to-world
        T = np.linalg.inv(T_wc)
        poses.append(T)
        image_names.append(image.name)
    poses = np.stack(poses, axis=0)
    return image_names, poses

def recenter_poses(poses):
    """
    모든 포즈의 카메라 중심을 평균으로 빼서 중앙화합니다.
    """
    centers = poses[:, :3, 3]
    center_mean = centers.mean(axis=0)
    poses_recentered = poses.copy()
    poses_recentered[:, :3, 3] -= center_mean
    return poses_recentered, center_mean

def scale_poses(poses, desired_scale=1.0):
    """
    포즈의 카메라 중심들이 desired_scale 내에 들어오도록 스케일 보정합니다.
    (최대 거리를 기준으로 보정)
    """
    centers = poses[:, :3, 3]
    max_dist = np.max(np.linalg.norm(centers, axis=1))
    scale_factor = desired_scale / max_dist
    poses_scaled = poses.copy()
    poses_scaled[:, :3, 3] *= scale_factor
    return poses_scaled, scale_factor

def poses_to_llff_format(poses, bounds):
    """
    각 4x4 포즈에서 상위 3행 4열을 취해 3×4 행렬을 만든 후,
    한 열(1)을 추가하여 3×5 행렬 (15개 값)로 만든 뒤,
    bounds(2개 값)와 연결하여 LLFF 형식(총 17개 값)을 생성합니다.
    """
    N = poses.shape[0]
    llff_list = []
    for i in range(N):
        pose_3x4 = poses[i, :3, :4]  # 3x4 행렬
        # LLFF에서는 보통 마지막 열에 1이 채워진 3×5 행렬로 만듦
        pose_3x5 = np.concatenate([pose_3x4, np.ones((3, 1))], axis=1)
        llff_list.append(pose_3x5.flatten())  # 15개 값
    llff_poses = np.stack(llff_list, axis=0)  # (N, 15)
    # bounds: (N, 2) 배열 (예: 모든 뷰에 대해 같은 [near, far])
    llff = np.concatenate([llff_poses, bounds], axis=1)  # (N, 17)
    return llff

def main():
    # 경로 설정 (필요에 따라 수정)
    image_dir = "images"            # 모든 입력 이미지가 있는 폴더
    database_path = "database.db"   # COLMAP 데이터베이스 파일
    output_path = "colmap_output"   # COLMAP 결과가 저장될 폴더

    # COLMAP 재구성 실행 (이미지 폴더를 입력)
    reconstruction = run_colmap(image_dir, database_path, output_path)
    
    # 재구성 결과에서 포즈 추출
    image_names, poses = extract_poses_from_reconstruction(reconstruction)
    print("Extracted poses shape:", poses.shape)  # (M, 4, 4), M ≤ 56(실제 등록된 이미지 수)
    
    # recenter 및 scale 보정 (예: desired_scale = 1.0)
    poses, center_mean = recenter_poses(poses)
    poses, scale_factor = scale_poses(poses, desired_scale=1.0)
    print("Poses recentered and scaled.")

    # LLFF 형식으로 변환하기 위해 각 이미지에 대해 같은 bounds [near, far] 사용 (예: [0.1, 10.0])
    N = poses.shape[0]
    bounds = np.tile(np.array([0.1, 10.0])[None, :], (N, 1))
    llff_poses = poses_to_llff_format(poses, bounds)
    print("LLFF poses shape:", llff_poses.shape)  # (M, 17)

    # 결과 저장
    np.save("poses.npy", poses)
    print("Saved poses.npy in 4x4 format.")
    np.save("poses_bounds.npy", llff_poses)
    print("Saved poses_bounds.npy in LLFF format.")

if __name__ == "__main__":
    main()
