import math
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 커스텀 CSS 스타일 적용
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CD-BAR 스마트 카운팅 & 중량 분석 시스템",
    page_icon="🔩",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff;
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .header-card h1 {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 6px !important;
    }
    .header-card p {
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0 !important;
    }

    .metric-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-val-main {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #10b981;
        font-weight: 600;
        margin-top: 4px;
    }

    .guide-box {
        background-color: #f0f9ff;
        border-left: 4px solid #0284c7;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #0369a1;
        margin-bottom: 16px;
    }
    
    .stImageCoordinates {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. 헤더 섹션
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-card">
        <h1>🔩 CD-BAR 스마트 카운팅 & 중량 산출 시스템</h1>
        <p>메인 묶음 선경 정규화 · 이중 인식 중복 제거 · 다중 이미지 통합 검수 보고서</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 3. AI 모델 로드
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception:
    st.error("❌ 'best.pt' 모델을 불러올 수 없습니다. GitHub 저장소를 확인해 주세요.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. 사이드바 - 설정 및 제조번호 입력 구역
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 시스템 설정")

    lot_number_input = st.text_input(
        "🏷️ 제조번호 (Lot No.)",
        value="",
        placeholder="예: P265MD123-01-10",
        help="검수 보고서 표기용 제조번호입니다.",
    )
    lot_number = lot_number_input.strip() if lot_number_input.strip() else "미입력"

    st.markdown("---")
    st.subheader("🎯 AI 탐지 및 묶음 제어")
    conf_thresh = st.slider(
        "탐지 민감도 (Confidence)",
        min_value=0.05,
        max_value=0.70,
        value=0.20,
        step=0.05,
    )
    tolerance = st.slider(
        "메인 묶음 선경 오차 범위 (%)",
        min_value=10,
        max_value=50,
        value=25,
        step=5,
        help="메인 묶음 대표 선경 대비 크기가 다른 소형/대형 바(하단 다른 묶음 등)를 자동 제외합니다.",
    ) / 100.0

    st.markdown("---")
    st.subheader("📏 제품 규격 및 중량 설정")
    bar_diameter = st.number_input(
        "선경 (지름, mm)",
        min_value=1.0,
        max_value=200.0,
        value=12.0,
        step=0.5,
    )
    bar_length_mm = st.number_input(
        "제품 길이 (L, mm)",
        min_value=100,
        max_value=30000,
        value=6000,
        step=1,
    )
    steel_density = st.number_input(
        "철 비중 (g/cm³)",
        min_value=7.0,
        max_value=9.0,
        value=7.85,
        step=0.01,
    )

    st.markdown("---")
    if st.button("🧹 전체 사진 분석 데이터 초기화", use_container_width=True):
        st.session_state.image_data = {}
        st.rerun()

if "image_data" not in st.session_state:
    st.session_state.image_data = {}

# -----------------------------------------------------------------------------
# 5. 다중 이미지 업로드 & 메인 연산 프로세스
# -----------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "CD-BAR 단면 촬영 사진을 업로드하세요 (여러 장 선택 가능)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    for file in uploaded_files:
        img_id = f"{file.name}_{file.size}"
        if img_id not in st.session_state.image_data:
            image = Image.open(file).convert("RGB")

            # 리사이징
            max_size = 1200
            if max(image.width, image.height) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            results = model(image, conf=conf_thresh)
            boxes = results[0].boxes

            raw_ai_centers = []
            target_radius = 12

            if len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                widths = xyxy[:, 2] - xyxy[:, 0]
                heights = xyxy[:, 3] - xyxy[:, 1]

                # 가장 많은 비중을 차지하는 메인 묶음의 대표 크기(중앙값) 산출
                median_w = np.median(widths)
                median_h = np.median(heights)
                target_radius = max(3, int((median_w + median_h) / 4))

                # 1단계: 대표 선경 범위를 벗어나는 바(하단 다른 소형 묶음 등) 필터링
                for box, w, h in zip(xyxy, widths, heights):
                    if (1.0 - tolerance) * median_w <= w <= (1.0 + tolerance) * median_w and \
                       (1.0 - tolerance) * median_h <= h <= (1.0 + tolerance) * median_h:
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        raw_ai_centers.append((cx, cy))

                # 2단계: 중심점 거리 기반 중복 제거 (이중 초록원 오류 해결)
                # 두 중심점 사이 거리가 반지름의 70% 이내면 동일 객체로 판단해 1개만 유지
                clean_ai_centers = []
                for cx, cy in raw_ai_centers:
                    is_duplicate = False
                    for kx, ky, _ in clean_ai_centers:
                        if math.hypot(cx - kx, cy - ky) < (target_radius * 0.7):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        clean_ai_centers.append((cx, cy, True))

            st.session_state.image_data[img_id] = {
                "file": file,
                "image": image,
                "centers": clean_ai_centers,
                "radius": target_radius,
                "last_clicked": None,
            }

    # 전체 수량 및 중량 종합 집계
    grand_total_count = sum(len(data["centers"]) for data in st.session_state.image_data.values())

    radius_cm = (bar_diameter / 10.0) / 2.0
    area_cm2 = math.pi * (radius_cm**2)
    length_cm = bar_length_mm / 10.0
    volume_cm3 = area_cm2 * length_cm
    unit_weight_kg = (volume_cm3 * steel_density) / 1000.0

    grand_total_weight_kg = unit_weight_kg * grand_total_count
    grand_total_weight_ton = grand_total_weight_kg / 1000.0

    # -----------------------------------------------------------------------------
    # 6. 상단 종합 대시보드 지표
    # -----------------------------------------------------------------------------
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">제조번호 (Lot No.)</div>
                <div class="metric-val-main" style="font-size: 1.1rem; padding-top: 8px; color: #0284c7;">{lot_number}</div>
                <div class="metric-sub">검수 대상 번호</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">총 검수 수량 (합계)</div>
                <div class="metric-val-main">{grand_total_count} <span style="font-size:1rem;">개</span></div>
                <div class="metric-sub">총 {len(uploaded_files)}장 메인 묶음 합계</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">총 중량 (합계)</div>
                <div class="metric-val-main">{grand_total_weight_kg:,.1f} <span style="font-size:1rem;">kg</span></div>
                <div class="metric-sub">{grand_total_weight_ton:.3f} Ton</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">1본당 단위 중량</div>
                <div class="metric-val-main">{unit_weight_kg:.2f} <span style="font-size:1rem;">kg</span></div>
                <div class="metric-sub">비중 {steel_density} 기준</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m5:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">설정 제품 규격</div>
                <div class="metric-val-main">Ø{bar_diameter:.1f}</div>
                <div class="metric-sub">{bar_length_mm:,} mm</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------------
    # 7. 개별 사진 선택 및 수동 클릭 보정 구역
    # -----------------------------------------------------------------------------
    selected_file = st.selectbox(
        "📸 검수 및 수동 클릭 보정을 진행할 사진을 선택하세요:",
        uploaded_files,
        format_func=lambda x: f"📷 {x.name} (현재 카운트: {len(st.session_state.image_data[f'{x.name}_{x.size}']['centers'])}개)",
    )

    current_id = f"{selected_file.name}_{selected_file.size}"
    active_data = st.session_state.image_data[current_id]

    st.markdown(
        """
        <div class="guide-box">
            👉 <b>터치/클릭 수동 보정 가이드:</b><br>
            • <b>[기존 원 클릭]</b> : 잘못 인식된 이중 원 및 오탐지를 즉시 삭제합니다. ❌<br>
            • <b>[빈 공간 클릭]</b> : 누락된 단면에 수동 원(노란색)을 새롭게 추가합니다. 🟡
        </div>
        """,
        unsafe_allow_html=True,
    )

    image = active_data["image"]
    target_radius = active_data["radius"]

    output_img = image.copy()
    draw = ImageDraw.Draw(output_img)

    for cx, cy, is_ai in active_data["centers"]:
        x0, y0 = cx - target_radius, cy - target_radius
        x1, y1 = cx + target_radius, cy + target_radius

        if is_ai:
            draw.ellipse([x0, y0, x1, y1], outline="#00FF66", width=3)
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill="#FF0033")
        else:
            draw.ellipse([x0, y0, x1, y1], outline="#FFCC00", width=3)
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill="#0066FF")

    value = streamlit_image_coordinates(
        output_img, key=f"canvas_{current_id}"
    )

    if value is not None and value != active_data["last_clicked"]:
        active_data["last_clicked"] = value
        click_x, click_y = value["x"], value["y"]

        hit_index = None
        threshold_dist = max(target_radius, 12)

        for idx, (cx, cy, is_ai) in enumerate(active_data["centers"]):
            dist = math.hypot(cx - click_x, cy - click_y)
            if dist <= threshold_dist:
                hit_index = idx
                break

        if hit_index is not None:
            active_data["centers"].pop(hit_index)
        else:
            active_data["centers"].append((click_x, click_y, False))

        st.rerun()

    # -----------------------------------------------------------------------------
    # 8. 종합 검수 명세표
    # -----------------------------------------------------------------------------
    with st.expander("📋 제조번호별 사진별 상세 측정 명세표 보기", expanded=True):
        spec_table = f"""
        | 사진 파일명 | 측정 수량 (메인 묶음) | 예상 중량 (kg) | 비고 |
        | :--- | :--- | :--- | :--- |
        """
        for img_key, data in st.session_state.image_data.items():
            cnt = len(data["centers"])
            wt = cnt * unit_weight_kg
            fname = data["file"].name
            spec_table += f"| **{fname}** | {cnt} EA | {wt:,.1f} kg | 메인 묶음 개별 산출 | \n"

        spec_table += f"""
        | **[합계 / Lot: {lot_number}]** | **{grand_total_count} EA** | **{grand_total_weight_kg:,.2f} kg ({grand_total_weight_ton:.3f} Ton)** | **총 {len(uploaded_files)}장 통합** |
        """
        st.markdown(spec_table)

else:
    st.info("👆 상단의 업로드 창을 통해 CD-BAR 단면 사진(1장 이상)을 등록해 주세요.")
