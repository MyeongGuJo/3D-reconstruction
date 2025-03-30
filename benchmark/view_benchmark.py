import cv2
import os

# 이미지 파일들이 있는 폴더 경로를 지정합니다.
folder = "images"  # 예: "./images"

# 폴더 내의 모든 .png 파일을 정렬된 순서로 가져옵니다.
image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])

for img_file in image_files:
    img_path = os.path.join(folder, img_file)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Failed to load {img_file}")
        continue

    cv2.imshow("Image Viewer", img)
    key = cv2.waitKey(0) & 0xFF
    # ESC 키 (27번)가 눌리면 종료합니다.
    if key == 27:
        break

cv2.destroyAllWindows()
