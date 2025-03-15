import cv2
import numpy as np

def save_video2images(video_path):
    i = 0
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 영상 끝에 도달하면 종료

            # save
            cv2.imwrite(f"images/image{i}.png", frame)
            i += 1

    cap.release()

    print("images ready")

def video2array(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []  # 모든 프레임을 저장할 리스트

    if not cap.isOpened():
        print("Error: Cannot open video file.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 영상 끝에 도달하면 종료
            
            # frame array
            frames.append(frame)

    cap.release()

    # 리스트를 numpy array로 변환 (shape: [frames, height, width, channels])
    frames_array = np.array(frames, dtype=np.uint8)
    print("Frames shape:", frames_array.shape)

    return frames_array
