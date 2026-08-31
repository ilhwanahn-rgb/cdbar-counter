import numpy as np
import cv2

# 1. YOLO 예측 실행
results = model(image, conf=0.35) # 기본 탐지
boxes = results[0].boxes

if len(boxes) > 0:
    # 각 박스의 너비와 높이 추출
    xyxy = boxes.xyxy.cpu().numpy()
    widths = xyxy[:, 2] - xyxy[:, 0]
    heights = xyxy[:, 3] - xyxy[:, 1]
    
    # 2. [도메인 지식 적용] 번들 대표 선경(중앙값) 산출
    median_w = np.median(widths)
    median_h = np.median(heights)
    target_radius = int((median_w + median_h) / 4) # 대표 반지름
    
    valid_centers = []
    
    for box, w, h in zip(xyxy, widths, heights):
        # 3. 대표 선경 대비 ±25% 범위를 벗어나는 오탐지(옆면/허공) 필터링
        if 0.75 * median_w <= w <= 1.25 * median_w and 0.75 * median_h <= h <= 1.25 * median_h:
            # 중심점 좌표 추출
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            valid_centers.append((cx, cy))
            
    # 4. 동일한 규격(target_radius)으로 정밀하게 원 그리기
    output_img = np.array(image).copy()
    for cx, cy in valid_centers:
        # 녹색 동일 크기 원 그리기
        cv2.circle(output_img, (cx, cy), target_radius, (0, 255, 0), 2)
        # 중심점 표시
        cv2.circle(output_img, (cx, cy), 3, (0, 0, 255), -1)

    # 최종 정제된 카운팅 수
    final_count = len(valid_centers)
