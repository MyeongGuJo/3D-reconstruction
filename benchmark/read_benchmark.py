import cv2
import numpy as np

# 사용할 데이터 파일 지정 (예: tiny_nerf_data.npz)
data_path = "tiny_nerf_data.npz"
data = np.load(data_path)

images = data['images']  # shape: (N, H, W, 3)
poses = data['poses']    # shape: (N, 4, 4)
focal = data['focal']

N, H, W, _ = images.shape
cx = W / 2.0
cy = H / 2.0

# 카메라 내재치 행렬 (여기서는 principal point를 이미지 중심으로 가정)
K = np.array([
    [focal,   0,   cx],
    [  0,   focal, cy],
    [  0,     0,    1]
], dtype=np.float32)

# 재투영을 위한 상수 깊이 값 (데모용)
depth = 1.0

# 목표 프레임(타깃)의 픽셀 좌표 생성 (H x W)
u, v = np.meshgrid(np.arange(W), np.arange(H))
u = u.astype(np.float32)
v = v.astype(np.float32)

# 타깃 카메라 좌표계에서 픽셀을 정규화된 좌표로 변환한 후, 상수 깊이를 곱함
x = (u - cx) / focal
y = (v - cy) / focal
pts_cam_target = np.stack([x, y, np.ones_like(x)], axis=-1) * depth  # (H, W, 3)

# 동차좌표로 확장
pts_cam_target_homo = np.concatenate([pts_cam_target, np.ones((H, W, 1), dtype=np.float32)], axis=-1)
pts_cam_target_homo = pts_cam_target_homo.reshape(-1, 4).T  # shape: (4, H*W)

# 모든 연속 프레임에 대해 재투영 계산
for i in range(N - 1):
    # source: frame i, target: frame i+1
    pose_src = poses[i]    # source 이미지의 camera-to-world 행렬
    pose_tgt = poses[i+1]  # target 이미지의 camera-to-world 행렬

    # 1. target 카메라 좌표계에서 world 좌표계로 변환
    pts_world = pose_tgt @ pts_cam_target_homo  # (4, H*W)

    # 2. world 좌표계를 source 카메라 좌표계로 변환
    pts_src = np.linalg.inv(pose_src) @ pts_world  # (4, H*W)

    # 동차좌표를 일반 좌표로 변환
    pts_src = pts_src[:3, :]  # (3, H*W)
    x_src = pts_src[0, :]
    y_src = pts_src[1, :]
    z_src = pts_src[2, :]

    # 3. source 카메라 내재치를 통해 픽셀 좌표로 재투영
    u_src = (x_src / z_src) * focal + cx
    v_src = (y_src / z_src) * focal + cy

    # cv2.remap에 사용할 매핑 좌표 생성
    map_x = u_src.reshape(H, W).astype(np.float32)
    map_y = v_src.reshape(H, W).astype(np.float32)

    # 4. source 이미지(frame i)에서 역매핑(remap)을 통해 target 시점(frame i+1) 합성
    src_img = images[i]
    synthesized_img = cv2.remap(src_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 결과 출력: 합성된 이미지와 실제 target 이미지(frame i+1)를 나란히 보여줌
    cv2.imshow("Synthesized Frame (from frame {})".format(i), synthesized_img)
    cv2.imshow("Actual Frame (frame {})".format(i+1), images[i+1])

    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC키를 누르면 종료
        break

    cv2.destroyAllWindows()

cv2.destroyAllWindows()
