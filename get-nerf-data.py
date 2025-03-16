import pycolmap
import numpy as np
from pathlib import Path

import utils.nerf_data_format

reconstruction_path = Path("output/0")
output_path = 'coffee_data.npz'

if __name__=="__main__":
    print("Get poses...")
    image_files, poses = utils.nerf_data_format.get_poses(reconstruction_path)
    print(poses.shape, type(poses))

    print("Get images...")
    images = utils.nerf_data_format.get_images(image_files)
    print(images.shape, type(images))
    
    H = images.shape[1]
    W = images.shape[2]
    focal = 1.2 * max(H, W)
    print(f"focal length: {focal}, {type(focal)}")

    np.savez(output_path, images=images, poses=poses, focal=focal)
    print(f"Data saved to {output_path}")
