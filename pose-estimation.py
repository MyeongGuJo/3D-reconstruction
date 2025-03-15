import os
import subprocess
import pycolmap

import utils.video

video_path = "coffee.mp4"  # 비디오 파일 경로로 변경하세요
image_dir = "images"       # 예: "./images"
database_path = "colmap_database.db"      # COLMAP 데이터베이스 파일
sparse_dir = "sparse"                     # 재구성 결과 폴더

if __name__=="__main__":
    for file in [database_path, database_path+"-wal", database_path+"-shm"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"{file} is removed.")
    
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
        utils.video.save_video2images(video_path)

    if not os.path.exists(sparse_dir):
        os.mkdir(sparse_dir)

    # 1. Feature Extraction: pycolmap을 사용하여 SIFT 특징점 추출
    print("Extracting features...")
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir
    )

    # 2. Feature Matching: pycolmap의 exhaustive matching 사용
    print("\nMatching features...")
    pycolmap.match_exhaustive(
        database_path=database_path
    )

    # 3. COLMAP SfM 실행: COLMAP의 mapper CLI를 subprocess로 호출
    print("\nRunning COLMAP incremental SfM (mapper)...")
    mapper_command = [
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir
    ]

    subprocess.run(mapper_command, check=True)

    # 4. SfM 결과 불러오기: 재구성 결과는 보통 sparse/0 폴더에 저장됨
    recon_path = os.path.join(sparse_dir, "0")
    if os.path.exists(recon_path):
        reconstruction = pycolmap.Reconstruction(recon_path)
        print("\nEstimated camera poses:")
        for image_id, image in reconstruction.images.items():
            print(f"Image: {image.name}")
            print(image.pose)  # 4x4 카메라-to-world 변환 행렬
            print("-" * 30)
    else:
        print(f"\nReconstruction directory '{recon_path}' not found.")
