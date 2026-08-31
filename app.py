import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

# 웹페이지 기본 설정
st.set_page_config(page_title="CD-BAR 카운터 Pro", layout="wide")

st.title("📦 CD-BAR 단면 카운터 & 중량 계산기")
st.write("사진을 업로드하면 CD-BAR 단면 개수를 자동 감지하고, 재질 및 규격에 맞게 중량을 즉시 산출합니다.")

# 사이드바 - 설정 패널
st.sidebar.header("⚙️ 규격 및 재질 설정")

# 재질 및 비중 선택 (탄소강/합금강 vs STS)
material_options = {
    "탄소강 / 합금강 (7.85 g/cm³)": 7.85,
    "STS 304 / 304L (7.93 g/cm³)": 7.93,
    "STS 316 / 316L (7.98 g/cm³)": 7.98,
    "STS 430 (7.70 g/cm³)": 7.70,
    "직접 입력": None
}

selected_material = st.sidebar.selectbox("재질 선택", list(material_options.keys()))

if selected_material == "직접 입력":
    density = st.sidebar.number_input("비중 입력 (g/cm³)", value=7.85, step=0.01)
else:
    density = material_options[selected_material]

# 선경 (직경, mm) 및 길이 (m)
diameter = st.sidebar.number_input("선경 / 직경 (mm)", value=16.0, step=0.5)
length = st.sidebar.number_input("1본당 길이 (m)", value=2.5, step=0.1)

# AI/영상처리 감도 옵션
st.sidebar.header("🔍 탐지 감도 조절")
min_dist = st.sidebar.slider("단면 간 최소 거리", 10, 100, 25)
param2 = st.sidebar.slider("원 탐지 민감도 (작을수록 더 많이 인식)", 10, 50, 20)
min_radius = st.sidebar.slider("최소 원 반지름 (px)", 5, 50, 8)
max_radius = st.sidebar.slider("최대 원 반지름 (px)", 10, 100, 30)

# 메인 화면 - 사진 업로드 및 카메라
uploaded_file = st.file_uploader("CD-BAR 단면 사진을 선택하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 읽기
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)

    # Hough Circles 기반 원형 탐지 알고리즘
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=50,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    output_image = image.copy()
    count = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        count = len(circles[0, :])
        for i, c in enumerate(circles[0, :]):
            # 초록색 원 그리기
            cv2.circle(output_image, (c[0], c[1]), c[2], (0, 255, 0), 2)
            # 중심점 빨간색 점 그리기
            cv2.circle(output_image, (c[0], c[1]), 2, (0, 0, 255), 3)

    # 중량 계산 공식: W(kg) = (pi / 4000) * d^2 * L * density * count
    single_weight = (math.pi / 4000.0) * (diameter ** 2) * length * density
    total_weight_kg = single_weight * count
    total_weight_ton = total_weight_kg / 1000.0

    # 현황 요약 카드 출력
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("측정 수량", f"{count} 개")
    col2.metric("1본당 중량", f"{single_weight:.3f} kg")
    col3.metric("총 중량 (kg)", f"{total_weight_kg:.2f} kg")
    col4.metric("총 중량 (ton)", f"{total_weight_ton:.3f} ton")

    # 결과 이미지 출력
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    st.image(output_image_rgb, caption=f"분석 완료: 총 {count}개 감지됨", use_container_width=True)
else:
    st.info("👆 위 영역에 사진을 끌어다 놓거나 선택해 주세요.")
