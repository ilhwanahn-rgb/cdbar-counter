import math
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from ultralytics import YOLO

# 1. 웹 페이지 기본 설정
st.set_page_config(page_title="CD-BAR 단면 카운터 & 중량 계산기", layout="centered")
st.title("🔩 CD-BAR 단면 카운터 & 중량 자동 계산기")
st.write(
    "AI 자동 감지 및 스마트 클릭 보정(추가/삭제) 후, **선경과 길이에 따른 총 중량까지 실시간 산출**합니다."
)


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

# 3. 사이드바 - AI 설정 및 중량 규격 입력
st.sidebar.header("⚙️ AI 탐지 설정")
conf_thresh = st.sidebar.slider(
    "AI 탐지 민감도", min_value=0.05, max_value=0.70, value=0.20, step=0.05
)
tolerance = (
    st.sidebar.slider("단면 크기 허용 오차 (%)", 10, 50, 35, 5) / 100.0
)

st.sidebar.markdown("---")
st.sidebar.header("📏 CD-BAR 규격 및 중량 설정")
bar_diameter = st.sidebar.number_input(
    "선경 (지름, mm)", min_value=1.0, max_value=200.0, value=12.0, step=0.5
)
bar_length = st.sidebar.number_input(
    "제품 길이 (L, m)", min_value=0.1, max_value=20.0, value=6.0, step=0.1
)
steel_density = st.sidebar.number_input(
    "철 비중 (g/cm³)", min_value=7.0, max_value=9.0, value=7.85, step=0.01
)

# 세션 상태 변수 초기화
if "image_id" not in st.session_state:
    st.session_state.image_id = None
if "active_centers" not in st.session_state:
    st.session_state.active_centers = []
if "target_radius" not in st.session_state:
    st.session_state.target_radius = 12
if "last_clicked" not in st.session_state:
    st.session_state.last_clicked = None

# 이미지 업로드 UI
uploaded_file = st.file_uploader(
    "CD-BAR 촬영 사진을 업로드하세요", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    current_id = f"{uploaded_file.name}_{uploaded_file.size}"

    # 사진이 새로 올라오면 최초 1회 AI 탐지 실행
    if st.session_state.image_id != current_id:
        st.session_state.image_id = current_id
        st.session_state.last_clicked = None

        image = Image.open(uploaded_file).convert("RGB")
        results = model(image, conf=conf_thresh)
        boxes = results[0].boxes

        ai_centers = []
        target_radius = 12

        if len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            widths = xyxy[:, 2] - xyxy[:, 0]
            heights = xyxy[:, 3] - xyxy[:, 1]

            median_w = np.median(widths)
            median_h = np.median(heights)
            target_radius = max(3, int((median_w + median_h) / 4))

            for box, w, h in zip(xyxy, widths, heights):
                if (
                    (1.0 - tolerance) * median_w
                    <= w
                    <= (1.0 + tolerance) * median_w
                ) and (
                    (1.0 - tolerance) * median_h
                    <= h
                    <= (1.0 + tolerance) * median_h
                ):
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    ai_centers.append((cx, cy, True))

        st.session_state.active_centers = ai_centers
        st.session_state.target_radius = target_radius

    image = Image.open(uploaded_file).convert("RGB")
    target_radius = st.session_state.target_radius

    # 리셋 버튼
    if st.sidebar.button("🧹 AI 감지 원본 상태로 리셋"):
        st.session_state.image_id = None
        st.rerun()

    # 이미지에 원 표시
    output_img = image.copy()
    draw = ImageDraw.Draw(output_img)

    for cx, cy, is_ai in st.session_state.active_centers:
        x0, y0 = cx - target_radius, cy - target_radius
        x1, y1 = cx + target_radius, cy + target_radius

        if is_ai:
            draw.ellipse([x0, y0, x1, y1], outline="lime", width=3)
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill="red")
        else:
            draw.ellipse([x0, y0, x1, y1], outline="yellow", width=3)
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill="blue")

    st.subheader("💡 클릭 인터랙티브 보정 모드")
    st.caption(
        "🟢 **녹색**: AI 탐지 | 🟡 **노란색**: 수동 추가 | **[원 클릭 = 삭제 ❌]** / **[빈 곳 클릭 = 추가 ➕]**"
    )

    # 클릭 인터랙션
    value = streamlit_image_coordinates(output_img, key="interactive_img")

    if value is not None and value != st.session_state.last_clicked:
        st.session_state.last_clicked = value
        click_x, click_y = value["x"], value["y"]

        hit_index = None
        threshold_dist = max(target_radius, 12)

        for idx, (cx, cy, is_ai) in enumerate(st.session_state.active_centers):
            dist = math.hypot(cx - click_x, cy - click_y)
            if dist <= threshold_dist:
                hit_index = idx
                break

        if hit_index is not None:
            st.session_state.active_centers.pop(hit_index)
        else:
            st.session_state.active_centers.append((click_x, click_y, False))

        st.rerun()

    # 4. 수량 및 중량 수식 계산
    total_count = len(st.session_state.active_centers)
    ai_count = sum(1 for item in st.session_state.active_centers if item[2])
    manual_count = sum(
        1 for item in st.session_state.active_centers if not item[2]
    )

    # 1본당 중량(kg) = 단면적(cm²) × 길이(cm) × 비중(g/cm³) / 1000
    radius_cm = (bar_diameter / 10.0) / 2.0
    area_cm2 = math.pi * (radius_cm**2)
    length_cm = bar_length * 100.0
    volume_cm3 = area_cm2 * length_cm
    unit_weight_kg = (volume_cm3 * steel_density) / 1000.0

    total_weight_kg = unit_weight_kg * total_count
    total_weight_ton = total_weight_kg / 1000.0

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="🎉 측정 CD-BAR 총 수량",
            value=f"{total_count} 개",
            delta=f"AI: {ai_count}개 | 수동: {manual_count}개",
        )
    with col2:
        st.metric(
            label="⚖️ 예상 총 중량",
            value=f"{total_weight_kg:,.1f} kg",
            delta=f"({total_weight_ton:.3f} Ton)",
        )

    st.info(
        f"📊 **적용 규격**: 선경 `{bar_diameter} mm` × 길이 `{bar_length} m` | **1본당 단중**: `{unit_weight_kg:.2f} kg` (비중 `{steel_density} g/cm³` 기준)"
    )
