import math
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 커스텀 CSS
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

    html, body, p, span, label, div, .stMarkdown {
        font-size: 1.15rem !important;
    }

    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff;
        padding: 28px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .header-card h1 {
        color: #ffffff !important;
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        margin-bottom: 8px !important;
    }
    .header-card p {
        color: #94a3b8 !important;
        font-size: 1.25rem !important;
        margin-bottom: 0 !important;
    }

    .metric-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .metric-title {
        font-size: 1.15rem !important;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .metric-val-main {
        font-size: 2.1rem !important;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-sub {
        font-size: 1.1rem !important;
        color: #10b981;
        font-weight: 700;
        margin-top: 6px;
    }

    .guide-box {
        background-color: #f0f9ff;
        border-left: 5px solid #0284c7;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 1.2rem !important;
        color: #0369a1;
        margin-bottom: 20px;
        line-height: 1.6;
    }

    table, th, td {
        font-size: 1.15rem !important;
        padding: 12px !important;
    }

    .stSidebar label, .stSidebar p, .stSidebar div {
        font-size: 1.1rem !important;
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
# 2. 헤더
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-card">
        <h1>🔩 CD-BAR 스마트 카운팅 & 중량 산출 시스템</h1>
        <p>AI 자동 단면 인식 · 규격 공차 필터링 · 중복 제거 · 클릭 한 번으로 수동 보정</p>
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
    st.error("❌ 'best.pt' 모델을 불러올 수 없습니다. 저장소에 모델 파일이 있는지 확인해 주세요.")
    st.stop()

# 추론용 이미지 해상도 상한(px). 화면 표시용 썸네일보다 훨씬 크게 유지해야
# 촘촘하게 붙어있는 작은 단면들이 뭉개지지 않고 잘 분리 인식된다.
# ponytail: 정확도가 더 필요하면 이 값과 YOLO_IMGSZ를 함께 올리면 되지만, 느려진다.
INFER_MAX_SIDE = 2200
YOLO_IMGSZ = 1536
YOLO_IOU = 0.9  # 원형 단면은 서로 붙어있으면 박스가 크게 겹치므로, 기본 NMS(0.7)보다 느슨하게 잡아야
                # 진짜 이웃한 개별 철근이 하나로 병합/삭제되지 않는다.
DISPLAY_MAX_SIDE = 1200

# -----------------------------------------------------------------------------
# 4. 사이드바 - 설정 및 제조번호 입력
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 시스템 설정")

    lot_number_input = st.text_input(
        "🏷️ 제조번호 (Lot No.)",
        value="",
        placeholder="예: P265MD123-01-10",
        help="검수 보고서 작성용 제조번호입니다.",
    )
    lot_number = lot_number_input.strip() if lot_number_input.strip() else "미입력"

    st.markdown("---")
    st.subheader("🎯 AI 탐지 및 묶음 제어")
    conf_thresh = st.slider(
        "탐지 민감도 (Confidence)",
        min_value=0.03,
        max_value=0.70,
        value=0.15,
        step=0.01,
    )
    tolerance = st.slider(
        "규격 오차 허용 범위 (%)",
        min_value=10,
        max_value=60,
        value=35,
        step=5,
        help="단면 박스 크기가 중앙값 대비 이 범위를 벗어나면 오탐으로 간주해 제외합니다.",
    ) / 100.0

    st.markdown("---")
    st.subheader("📏 철근 규격 및 중량 설정")
    bar_diameter = st.number_input(
        "직경 (Diameter, mm)",
        min_value=1.0,
        max_value=200.0,
        value=12.0,
        step=0.5,
    )
    bar_length_mm = st.number_input(
        "절단 길이 (L, mm)",
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
    if st.button("🧹 전체 분석 데이터 초기화", use_container_width=True):
        st.session_state.image_data = {}
        st.rerun()

if "image_data" not in st.session_state:
    st.session_state.image_data = {}

# -----------------------------------------------------------------------------
# 5. 다중 이미지 업로드 & 메인 분석 프로세스
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
            original_image = Image.open(file).convert("RGB")

            # 추론용 이미지: 표시용 썸네일보다 큰 해상도를 유지해 작은 단면도 잘 잡히게 한다.
            infer_image = original_image.copy()
            if max(infer_image.width, infer_image.height) > INFER_MAX_SIDE:
                infer_image.thumbnail((INFER_MAX_SIDE, INFER_MAX_SIDE), Image.Resampling.LANCZOS)

            # 화면 표시/클릭 좌표용 썸네일 (더 작게, 반응성 확보)
            display_image = infer_image.copy()
            if max(display_image.width, display_image.height) > DISPLAY_MAX_SIDE:
                display_image.thumbnail((DISPLAY_MAX_SIDE, DISPLAY_MAX_SIDE), Image.Resampling.LANCZOS)
            scale = display_image.width / infer_image.width

            results = model(infer_image, conf=conf_thresh, imgsz=YOLO_IMGSZ, iou=YOLO_IOU, verbose=False)
            boxes = results[0].boxes

            raw_centers = []
            target_radius_infer = 12

            if len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                widths = xyxy[:, 2] - xyxy[:, 0]
                heights = xyxy[:, 3] - xyxy[:, 1]

                median_w = np.median(widths)
                median_h = np.median(heights)
                target_radius_infer = max(3, int((median_w + median_h) / 4))

                # 1단계: 규격(지름) 공차 필터링 - 중앙값과 크게 다른 박스는 오탐/파편으로 간주
                for box, w, h in zip(xyxy, widths, heights):
                    if (1.0 - tolerance) * median_w <= w <= (1.0 + tolerance) * median_w and \
                       (1.0 - tolerance) * median_h <= h <= (1.0 + tolerance) * median_h:
                        cx = (box[0] + box[2]) / 2.0
                        cy = (box[1] + box[3]) / 2.0
                        raw_centers.append((cx, cy))

                # 2단계: 같은 단면이 중복 탐지된 경우만 제거 (같은 위치에 여러 박스)
                clean_centers = []
                for cx, cy in raw_centers:
                    is_duplicate = False
                    for kx, ky in clean_centers:
                        if math.hypot(cx - kx, cy - ky) < (target_radius_infer * 0.75):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        clean_centers.append((cx, cy))
            else:
                clean_centers = []

            # 표시 좌표계로 환산해서 저장. 묶음 연결성으로 통째로 걸러내던 이전 로직은
            # 연결이 한 군데만 끊겨도 정상 탐지된 단면 수십 개가 통째로 사라지는 원인이었으므로 제거.
            # 남는 오탐(배경 이물 등)은 화면에서 클릭 한 번으로 지우는 편이 누락보다 훨씬 안전하다.
            final_centers = [
                (cx * scale, cy * scale, True) for cx, cy in clean_centers
            ]
            target_radius = max(3, int(round(target_radius_infer * scale)))

            st.session_state.image_data[img_id] = {
                "file": file,
                "image": display_image,
                "centers": final_centers,
                "radius": target_radius,
                "last_clicked": None,
            }

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
                <div class="metric-val-main" style="font-size: 1.4rem !important; padding-top: 6px; color: #0284c7;">{lot_number}</div>
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
                <div class="metric-val-main">{grand_total_count} <span style="font-size:1.2rem;">개</span></div>
                <div class="metric-sub">총 {len(uploaded_files)}장 합계</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">총 중량 (합계)</div>
                <div class="metric-val-main">{grand_total_weight_kg:,.1f} <span style="font-size:1.2rem;">kg</span></div>
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
                <div class="metric-val-main">{unit_weight_kg:.2f} <span style="font-size:1.2rem;">kg</span></div>
                <div class="metric-sub">비중 {steel_density} 기준</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m5:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">설정 철근 규격</div>
                <div class="metric-val-main">Ø{bar_diameter:.1f}</div>
                <div class="metric-sub">{bar_length_mm:,} mm</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------------
    # 7. 개별 사진 선택 및 수동 클릭 보정 구간
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
            💡 <b>터치/클릭 수동 보정 가이드:</b><br>
            • <b>[기존 점 클릭]</b> : 잘못 인식된 점을 즉시 삭제합니다. ❌<br>
            • <b>[빈 공간 클릭]</b> : 누락된 단면 위치에 새로운 점(노란색)을 추가합니다. 💡
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
    # 8. 종합 검수 명세서
    # -----------------------------------------------------------------------------
    with st.expander("📋 제조번호별 사진별 상세 측정 명세서 보기", expanded=True):
        spec_table = """
        | 사진 파일명 | 측정 수량 (단면) | 산출 중량 (kg) | 비고 |
        | :--- | :--- | :--- | :--- |
        """
        for img_key, data in st.session_state.image_data.items():
            cnt = len(data["centers"])
            wt = cnt * unit_weight_kg
            fname = data["file"].name
            spec_table += f"| **{fname}** | {cnt} EA | {wt:,.1f} kg | 사진별 산출 | \n"

        spec_table += f"""
        | **[합계 / Lot: {lot_number}]** | **{grand_total_count} EA** | **{grand_total_weight_kg:,.2f} kg ({grand_total_weight_ton:.3f} Ton)** | **총 {len(uploaded_files)}장 통합** |
        """
        st.markdown(spec_table)

else:
    st.info("👆 상단의 업로드 창을 통해 CD-BAR 단면 사진(1장 이상)을 등록해 주세요.")
