import cv2
import numpy as np

def save_video2images(video_path, is_skip=False):
    i = 0
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
    else:
        while True:
            i += 1
            
            ret, frame = cap.read()
            if not ret:
                break  # 영상 끝에 도달하면 종료
            
            if is_skip and i%5 != 0:
                continue
            elif is_skip:
                H, W = frame.shape[:2]
                size = (W // 4, H // 4)
                frame = cv2.resize(frame, size)

            # save
            cv2.imwrite(f"images/image{i}.png", frame)

    cap.release()

    print("images ready")
