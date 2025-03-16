import os
import cv2
import pycolmap
import numpy as np

def qvec2rotmat(q):
    """
    q: [x, y, z, w] 순서의 쿼터니언
    회전 행렬 R (3x3)을 반환합니다.
    """
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
    ])
    return R

def get_poses(path):
    reconstruction = pycolmap.Reconstruction(path)
    # print(reconstruction.summary())
    
    image_files = []
    transformations = []
    
    for _, image in reconstruction.images.items():
        rigid = image.cam_from_world
        q = rigid.rotation.quat
        t = np.array(rigid.translation)
        
        # 쿼터니언을 회전 행렬로 변환
        R = qvec2rotmat(q)

        # 4x4 동차 변환 행렬 구성 (상단 좌측 3x3: 회전, 상단 우측: translation)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        image_files.append(image.name)
        transformations.append(T)
    
    return image_files, np.array(transformations)

def get_images(image_files):
    dir = "images/"
    images = []
    
    for image_file in image_files:
        image = cv2.imread(dir+image_file)
        if image is None:
            print(f"fail to load: {image_file}")
            continue
        images.append(image)
    
    return np.array(images)