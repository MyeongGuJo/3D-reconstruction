import os
import random
import cv2
import numpy as np
import rembg

def shuffle_frames(frames):
    """
    프레임 리스트를 섞어(shuffle) 반환합니다.
    """
    shuffled = frames[:]  # 원본 리스트를 복사
    random.shuffle(shuffled)
    return shuffled

def save_frames(frames, output_folder):
    """
    프레임 리스트의 각 프레임을 개별 이미지 파일로 저장합니다.
    """
    os.makedirs(output_folder, exist_ok=True)
    for i, frame in enumerate(frames):
        filename = os.path.join(output_folder, f"frame_{i:04d}.png")
        cv2.imwrite(filename, frame)

def save_video2images(video_path, is_skip=False):
    i = 0
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print("Error: Cannot open video file.")
    else:
        while True:
            i += 1
            
            ret, frame = cap.read()
            
            if not ret:
                break  # 영상 끝에 도달하면 종료
            
            if is_skip and i%3 != 0:
                continue
            H, W = frame.shape[:2]
            size = (W // 4, H // 4)
            frame = cv2.resize(frame, size)

            frames.append(frame)

    cap.release()

    shuffled = shuffle_frames(frames)
    save_frames(shuffled, "images")
    print("images ready")

def remove_background(img):
    """
    cv2 이미지(img)를 받아서 rembg를 사용해 배경을 제거한 후,
    결과 이미지를 (BGRA) 형태로 반환합니다.
    """
    # 이미지가 제대로 인코딩되는지 확인 (PNG 포맷)
    success, encoded_image = cv2.imencode('.png', img)
    if not success:
        raise ValueError("image encoding failed")
    
    # 바이트로 변환
    input_bytes = encoded_image.tobytes()
    
    # rembg를 이용해 배경 제거
    output_bytes = rembg.remove(input_bytes)
    
    # 바이트 스트림을 numpy 배열로 디코딩 (알파 채널 포함)
    output_array = np.frombuffer(output_bytes, np.uint8)
    result = cv2.imdecode(output_array, cv2.IMREAD_UNCHANGED)
    
    # 결과 이미지가 알파 채널(BGRA)이라면 알파 채널 제거 (배경은 검은색)
    if result.shape[2] == 4:
        result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
    
    return result
