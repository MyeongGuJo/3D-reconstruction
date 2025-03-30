import cv2
import numpy as np
import random
from pathlib import Path
import pycolmap

fixed_x, fixed_y = 100, 100  # 창을 띄울 고정 좌표

def to_cv2_keypoints(kpts):
    """
    COLMAP의 keypoints 배열이 (N, 6) 형태로 [x, y, scale, orientation, response, octave]를 담고 있다고 가정합니다.
    OpenCV KeyPoint 객체 리스트로 변환할 때는 첫 4개의 값(x, y, scale, orientation)을 사용합니다.
    """
    keypoints = []
    for kp in kpts:
        x, y, scale, orientation = kp[:4]
        keypoints.append(cv2.KeyPoint(x, y, size=scale, angle=orientation))
    return keypoints

# 경로 설정 (프로젝트 구조에 맞게 조정)
database_path = Path("output/database.db")
image_dir = Path("images")
sparse_dir = Path("output/0")  # COLMAP sparse reconstruction 결과 폴더

# COLMAP 재구성 결과를 읽어 이미지 정보를 가져옵니다.
recon = pycolmap.Reconstruction(str(sparse_dir))
images_dict = recon.images  # {image_id: Image object}
if not images_dict:
    print("Reconstruction에 이미지 정보가 없습니다.")
    exit(1)

# 이미지 정보를 파일명 순으로 정렬합니다.
sorted_images = sorted(images_dict.items(), key=lambda x: x[1].name)

# 데이터베이스 객체 생성 (COLMAP 데이터베이스 파일)
db = pycolmap.Database(str(database_path))

# 연속된 이미지 쌍에 대해 매칭 결과 시각화: (0,1), (1,2), (2,3), ...
for idx in range(len(sorted_images) - 1):
    id1, image1 = sorted_images[idx]
    id2, image2 = sorted_images[idx + 1]
    print(f"Processing match between '{image1.name}' and '{image2.name}'")

    # 각 이미지의 keypoints 읽기 (배열 형상: (N, 6))
    kpts1 = db.read_keypoints(id1)
    kpts2 = db.read_keypoints(id2)
    if kpts1 is None or kpts2 is None:
        print(f"키포인트 정보가 없습니다: {image1.name} 또는 {image2.name}")
        continue
    kp1 = to_cv2_keypoints(kpts1)
    kp2 = to_cv2_keypoints(kpts2)

    # TwoViewGeometry 객체를 통해 두 이미지 사이의 매칭 정보 읽기
    # 공식 문서: https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.TwoViewGeometry
    two_view = db.read_two_view_geometry(id1, id2)
    if two_view is None or two_view.inlier_matches.size == 0:
        print(f"매칭 정보가 없습니다: {image1.name} - {image2.name}")
        continue

    # inlier_matches는 (M, 2) 배열로, 각 행은 [query_idx, train_idx] 형태입니다.
    matches_indices = two_view.inlier_matches
    matches = []
    for m in matches_indices:
        query_idx, train_idx = m
        matches.append(cv2.DMatch(_queryIdx=int(query_idx),
                                  _trainIdx=int(train_idx),
                                  _distance=0))
    
    # 원본 이미지 읽기 (images 폴더 내에 파일들이 있어야 함)
    img1_path = image_dir / image1.name
    img2_path = image_dir / image2.name
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        print(f"이미지 로드 실패: {image1.name} 또는 {image2.name}")
        continue

    # OpenCV의 drawMatches 함수를 사용하여 두 이미지를 좌우로 이어붙이고 매칭 결과를 표시합니다.
    out_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    window_name = "Matches"
    
    # 기존 창을 닫고 같은 창 이름으로 새 창 생성
    cv2.imshow(window_name, out_img)
    cv2.moveWindow(window_name, fixed_x, fixed_y)
    
    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC키 누르면 종료
        break
    cv2.destroyWindow(window_name)

db.close()
cv2.destroyAllWindows()
