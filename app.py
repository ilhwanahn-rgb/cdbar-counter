import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from ultralytics import YOLO

# 1. 웹 페이지 기본 설정
st.set_page_config(page_title="CD-BAR 단면 카운터", layout="centered")
st.title("🔩 CD-BAR 단면 자동/수동 카운팅 시스템")
st.write(
    "AI 자동 감지 후, 누락된 단면은 **사진을 직접 클릭**하여 완벽하게 보정하세요."
)

# 세션 상태(클릭 좌표, 오프셋) 저장소 초기화
if "custom_points" not in st.session_state:
    st.session_state.custom_points = []
if "last_clicked" not in st.session_state:
    st.session_state.last_clicked = None

# 사이드바 컨트롤 설정
st.sidebar.header("⚙️ 탐지 민감도 및 보정")
conf_thresh = st.sidebar.slider(
    "AI 탐지 민감도 (낮출수록 어두운 단면도 감지)",
    min_value=0.05,
    max_value=0.70,
    value=0.20,
    step=0.05,
)
tolerance = (
    st.sidebar.slider("단면 크기 허용 오차 (%)", 10, 50, 35, 5) / 100.0
)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 수동 클릭 포인트 전체 초기화"):
    st.session_state.custom_points = []
    st.session_state.last_clicked = None
    st.rerun()


# 2. AI 모델(best.pt) 로드
@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:
    model = load_model()
except Exception as e:
    st.error(
        "❌ 'best.pt' 모델을 불러올 수 없습니다. GitHub 저장소를 확인해주세요."
    )
    st.stop()

# 3. 이미지 업로드 UI
uploaded_file = st.file_uploader(
    "CD-BAR 촬영 사진을 업로드하세요", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # 4. YOLOv8 AI 감지
    results = model(image, conf=conf_thresh)
    boxes = results[0].boxes

    valid_centers = []
    target_radius = 12  # 기본 반지름 예비값

    if len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]

        median_w = np.median(widths)
        median_h = np.median(heights)
        target_radius = max(3, int((median_w + median_h) / 4))

        for box, w, h in zip(xyxy, widths, heights):
            if (
                (1.0 - tolerance) * median_w <= w <= (1.0 + tolerance) * median_w
            ) and (
                (1.0 - tolerance) * median_h
                <= h
                <= (1.0 + tolerance) * median_h
            ):
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                valid_centers.append((cx, cy))

    # 5. 이미지 위에 AI 감지 결과(초록색) 및 수동 추가 결과(노란색) 그리기
    output_img = image.copy()
    draw = ImageDraw.Draw(output_img)

    # AI 감지 원 (초록색 + 빨간 점)
    for cx, cy in valid_centers:
        x0, y0 = cx - target_radius, cy - target_radius
        x1, y1 = cx + target_radius, cy + target_radius
        draw.ellipse([x0, y0, x1, y1], outline="lime", width=3)
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill="red")

    # 사용자 수동 추가 원 (노란색 + 파란 점)
    for cx, cy in st.session_state.custom_points:
        x0, y0 = cx - target_radius, cy - target_radius
        x1, y1 = cx + target_radius, cy + target_radius
        draw.ellipse([x0, y0, x1, y1], outline="yellow", width=3)
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill="blue")

    # 안내 및 인터랙티브 이미지 출력
    st.subheader("👇 사진의 미인식 단면을 터치/클릭하여 추가하세요")
    st.caption(
        "🟢 **초록색**: AI 자동 감지 | 🟡 **노란색**: 사용자 수동 클릭 추가"
    )

    # 마우스/손가락 클릭 좌표 수집
    value = streamlit_image_coordinates(output_img, key="pil_image")

    if value is not None and value != st.session_state.last_clicked:
        st.session_state.last_clicked = value
        click_x, click_y = value["x"], value["y"]
        st.session_state.custom_points.append((click_x, click_y))
        st.rerun()

    # 6. 수량 현황 표시
    ai_count = len(valid_centers)
    manual_count = len(st.session_state.custom_points)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏪ 마지막 클릭 수동 취소"):
            if len(st.session_state.custom_points) > 0:
                st.session_state.custom_points.pop()
                st.rerun()

    manual_offset = st.number_input(
        "🔢 숫자 미세 직접 입력 (+ / -)", value=0, step=1
    )

    total_count = ai_count + manual_count + manual_offset

    st.markdown("---")
    st.metric(
        label="🎉 최종 측정 CD-BAR 총 개수",
        value=f"{total_count} 개",
        delta=f"AI: {ai_count}개 / 수동 추가: {manual_count + manual_offset}개",
    )
