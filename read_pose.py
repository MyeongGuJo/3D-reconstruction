import cv2
import numpy as np
from pathlib import Path

import utils.nerf_data_format

reconstruction_path = Path("output/0")

if __name__=="__main__":
    print("Get poses...")
    image_files, poses = utils.nerf_data_format.get_poses(reconstruction_path)
    print(poses.shape, type(poses))

    print("Get images...")
    images = utils.nerf_data_format.get_images(image_files)
    print(images.shape, type(images))

    N, W, H = images.shape[:3]
    homo_sub = np.ones((W, H, 1))

    for i in range(N-1):
        img = images[i]
        img_target = images[i+1]
        print(poses[i], '\n', poses[i+1])
        relative_pose = poses[i+1] @ np.linalg.inv(poses[i])

        # (R, G, B) -> (R, G, B, 1): homogeneous coordinate
        img_homo = np.concat((img, homo_sub), axis=-1)
        img_homo = img_homo.reshape(-1, 4)
        print(img_homo.shape)
        img_calculated = img_homo @ relative_pose
        img_calculated = img_calculated.reshape(W, H, 4).astype(np.uint8)
        print(img_calculated.shape)

        cv2.imshow("calculated", img_calculated[:, :, :3])
        cv2.imshow("target", img_target)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC키 누르면 종료
            break

        cv2.destroyAllWindows()

    cv2.destroyAllWindows()
