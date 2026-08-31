import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

st.set_page_config(page_title="CD-BAR 단면 정밀 카운터", layout="wide")

st.title("📦 CD-BAR 단면 정밀 카운터 & 중량 계산기")
st.write("옆면 빛 반사 및 결속 끈 오탐지를 차단하고 끝단면만 정확히 카운팅합니다.")

# ==========================================
# 사이드바 - 설정 패널
# ==========================================
st.sidebar.header("⚙️ 규격 및 재질 설정")

material_options = {
    "탄소강 / 합금강 (7.85 g/cm³)": 7.85,
    "STS 304 / 304L (7.93 g/cm³)": 7.93,
    "STS 316 / 316L (7.98 g/cm³)": 7.98,
    "STS 430 (7.70 g/cm³)": 7.70,
    "직접 입력": None
}

selected_material = st.sidebar.selectbox("재질 선택", list(material_options.keys()))
density = st.sidebar.number_input("비중 입력 (g/cm³)", value=7.85, step=0.01) if selected_material == "직접 입력" else material_options[selected_material]

diameter = st.sidebar.number_input("선경 / 직경 (mm)", value=16.0, step=0.5)
length = st.sidebar.number_input("1본당 길이 (m)", value=2.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("🎯 오체크 방지 옵션")

# 1. 영역 자르기 (ROI) 옵션
st.sidebar.subheader("1. 분석 영역(ROI) 지정")
use_roi = st.sidebar.checkbox("옆면 제외하고 끝단부 영역만 지정하기", value=True)
roi_x_min = st.sidebar.slider("X축 시작 (왼쪽 옆면 잘라내기)", 0, 100, 30, help="퍼센트(%) 단위로 왼쪽 바 몸통 영역을 제외합니다.")

# 2. 단면 내부 밝기 및 원형도 필터
st.sidebar.subheader("2. 단면 필터링 강도")
min_brightness = st.sidebar.slider("최소 단면 밝기 (어두운 옆면 제거)", 50, 220, 110)
min_dist = st.sidebar.slider("단면 간 최소 거리 (px)", 10, 80, 24)
min_radius = st.sidebar.slider("최소 단면 반지름 (px)", 5, 50, 10)
max_radius = st.sidebar.slider("최대 단면 반지름 (px)", 10, 100, 32)

# ==========================================
# 메인 로직
# ==========================================
uploaded_file = st.file_uploader("CD-BAR 단면 사진을 선택하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    h, w, _ = image.shape

    # 마스킹 영역 설정 (X축 자르기)
    mask = np.ones((h, w), dtype=np.uint8) * 255
    if use_roi:
        crop_x = int(w * (roi_x_min / 100.0))
        mask[:, :crop_x] = 0  # 왼쪽 몸통 영역 제외

    # 전처리 (그레이스케일 & 적응형 명암 대비)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_clahe, (9, 9), 2)

    # 원 탐지
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=50,
        param2=22,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    output_image = image.copy()
    valid_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cx, cy, r = c[0], c[1], c[2]

            # 1) ROI 범위 밖 필터링
            if use_roi and cx < int(w * (roi_x_min / 100.0)):
                continue

            # 2) 이미지 경계 체크
            if cx - r < 0 or cx + r >= w or cy - r < 0 or cy + r >= h:
                continue

            # 3) 단면 내부 평균 밝기 검증 (검은 배경/옆면 그림자 제거)
            circle_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(circle_mask, (cx, cy), int(r * 0.7), 255, -1)
            mean_val = cv2.mean(gray, mask=circle_mask)[0]

            if mean_val >= min_brightness:
                valid_circles.append((cx, cy, r))

    # 시각화 (유효한 단면만 초록 원 표시)
    count = len(valid_circles)
    for cx, cy, r in valid_circles:
        cv2.circle(output_image, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(output_image, (cx, cy), 2, (0, 0, 255), 3)

    # ROI 비활성화 영역 시각화 (잘려진 영역에 어둡게 표시)
    if use_roi and roi_x_min > 0:
        crop_x = int(w * (roi_x_min / 100.0))
        overlay = output_image.copy()
        cv2.rectangle(overlay, (0, 0), (crop_x, h), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.4, output_image, 0.6, 0, output_image)
        cv2.line(output_image, (crop_x, 0), (crop_x, h), (0, 0, 255), 2)

    # 중량 계산
    single_weight = (math.pi / 4000.0) * (diameter ** 2) * length * density
    total_weight_kg = single_weight * count
    total_weight_ton = total_weight_kg / 1000.0

    # 현황 요약
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("측정 수량", f"{count} 개")
    col2.metric("1본당 중량", f"{single_weight:.3f} kg")
    col3.metric("총 중량 (kg)", f"{total_weight_kg:.2f} kg")
    col4.metric("총 중량 (ton)", f"{total_weight_ton:.3f} ton")

    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    st.image(output_image_rgb, caption=f"분석 완료: 총 {count}개 감지됨", use_container_width=True)
else:
    st.info("👆 위 영역에 CD-BAR 사진을 선택해 주세요.")