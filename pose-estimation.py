import os
import shutil
import pathlib
import pycolmap

import utils.video

"""
TODO 1. find default focal length of pyclomap
TODO 2. change focal length option to my camera's
"""

video_path = "megumin.mp4"  
output_path = pathlib.Path("output")
image_dir = pathlib.Path("images")# 재구성 결과 폴더

if __name__=="__main__":
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        print(f"{output_path} is removed.")
    
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
        print(f"{image_dir} is removed.")
        
    os.mkdir(image_dir)
    print(f"{image_dir} is created.")
    utils.video.save_video2images(video_path, is_skip=True)

    # # 출력 폴더 생성
    # output_path.mkdir(exist_ok=True)
    # mvs_path = output_path / "mvs"
    # database_path = output_path / "database.db"

    # # Sparse reconstruction
    # print("Extracting features...")
    # pycolmap.extract_features(database_path, image_dir)
    # print("Matching features...")
    # pycolmap.match_exhaustive(database_path)
    # print("Running incremental mapping...")
    # # incremental_mapping은 sparse reconstruction을 수행하며, 여러 맵(reconstructions)을 반환합니다.
    # maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    # # 첫 번째 reconstruction 결과를 저장합니다.
    # maps[0].write(output_path)
