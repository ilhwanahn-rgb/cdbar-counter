import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 커스텀 CSS 스타일 적용 (폰트 +5pt 확대)
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
    
    /* 기본 전체 글자 크기 +5pt 확대 */
    html, body, p, span, label, div, .stMarkdown {
        font-size: 1.15rem !important;
    }
    
    /* 타이틀 헤더 카드 */
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

    /* 메트릭 카운터 카드 */
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

    /* 안내 배너 상자 */
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
    
    /* 명세표 테이블 및 사이드바 확대 */
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
# 2. 헤더 섹션
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-card">
        <h1>🔩 CD-BAR 스마트 카운팅 & 중량 산출 시스템</h1>
        <p>단면 순번 번호(Numbering) 오버레이 · 메인 묶음 공간 군집 추출 · 통합 검수 보고서 다운로드</p>
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
        min_value=0.03,
        max_value=0.70,
        value=0.15,
        step=0.01,
    )
    tolerance = st.slider(
        "메인 묶음 선경 오차 범위 (%)",
        min_value=10,
        max_value=60,
        value=35,
        step=5,
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
        help="1mm 단위 입력 가능 (예: 6,000 mm)",
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

            # 최적화 리사이징 (최대 1200px)
            max_size = 1200
            if max(image.width, image.height) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            results = model(image, conf=conf_thresh, imgsz=1024)
            boxes = results[0].boxes

            raw_ai_centers = []
            target_radius = 12

            if len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                widths = xyxy[:, 2] - xyxy[:, 0]
                heights = xyxy[:, 3] - xyxy[:, 1]

                median_w = np.median(widths)
                median_h = np.median(heights)
                target_radius = max(3, int((median_w + median_h) / 4))

                # 1단계: 선경 규격 필터링
                for box, w, h in zip(xyxy, widths, heights):
                    if (1.0 - tolerance) * median_w <= w <= (1.0 + tolerance) * median_w and \
                       (1.0 - tolerance) * median_h <= h <= (1.0 + tolerance) * median_h:
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        raw_ai_centers.append((cx, cy))

                # 2단계: 이중 원 중복 제거 (NMS)
                clean_ai_centers = []
                for cx, cy in raw_ai_centers:
                    is_duplicate = False
                    for kx, ky in clean_ai_centers:
                        if math.hypot(cx - kx, cy - ky) < (target_radius * 0.75):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        clean_ai_centers.append((cx, cy))

                # 3단계: 공간 연결 군집(Cluster) 분석으로 메인 묶음 추출
                if len(clean_ai_centers) > 0:
                    max_connect_dist = target_radius * 3.3
                    n = len(clean_ai_centers)
                    adj = [[] for _ in range(n)]

                    for i in range(n):
                        for j in range(i + 1, n):
                            d = math.hypot(clean_ai_centers[i][0] - clean_ai_centers[j][0], 
                                           clean_ai_centers[i][1] - clean_ai_centers[j][1])
                            if d <= max_connect_dist:
                                adj[i].append(j)
                                adj[j].append(i)

                    visited = [False] * n
                    components = []

                    for i in range(n):
                        if not visited[i]:
                            comp = []
                            queue = [i]
                            visited[i] = True
                            while queue:
                                curr = queue.pop(0)
                                comp.append(curr)
                                for neighbor in adj[curr]:
                                    if not visited[neighbor]:
                                        visited[neighbor] = True
                                        queue.append(neighbor)
                            components.append(comp)

                    if components:
                        largest_comp = max(components, key=len)
                        final_ai_centers = [(clean_ai_centers[idx][0], clean_ai_centers[idx][1], True) for idx in largest_comp]
                    else:
                        final_ai_centers = [(cx, cy, True) for cx, cy in clean_ai_centers]
                else:
                    final_ai_centers = []
            else:
                final_ai_centers = []

            st.session_state.image_data[img_id] = {
                "file": file,
                "image": image,
                "centers": final_ai_centers,
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
            • <b>[기존 원 클릭]</b> : 해당 번호의 원을 즉시 삭제합니다. ❌<br>
            • <b>[빈 공간 클릭]</b> : 누락된 단면에 수동 원(노란색)을 새롭게 추가합니다. 🟡
        </div>
        """,
        unsafe_allow_html=True,
    )

    image = active_data["image"]
    target_radius = active_data["radius"]

    # 번호 표기를 위한 공간 정렬 (위에서 아래로, 왼쪽에서 오른쪽 순)
    row_band = max(10, int(target_radius * 1.6))
    sorted_centers = sorted(
        active_data["centers"],
        key=lambda item: (int(item[1] // row_band), item[0])
    )
    active_data["centers"] = sorted_centers

    output_img = image.copy()
    draw = ImageDraw.Draw(output_img)

    # 반지름 기반 폰트 크기 자동 산출
    font_size = max(10, int(target_radius * 1.1))
    try:
        font = ImageFont.load_default(size=font_size)
    except Exception:
        font = ImageFont.load_default()

    # 순번 번호 오버레이
    for idx, (cx, cy, is_ai) in enumerate(active_data["centers"], start=1):
        x0, y0 = cx - target_radius, cy - target_radius
        x1, y1 = cx + target_radius, cy + target_radius

        if is_ai:
            draw.ellipse([x0, y0, x1, y1], outline="#FF3333", width=3)
        else:
            draw.ellipse([x0, y0, x1, y1], outline="#FFCC00", width=3)

        num_str = str(idx)
        try:
            bbox = draw.textbbox((0, 0), num_str, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = 8, 10

        tx, ty = cx - (tw / 2), cy - (th / 2)
        draw.text((tx + 1, ty + 1), num_str, fill="black", font=font)
        draw.text((tx, ty), num_str, fill="white", font=font)

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
    # 8. 종합 검수 명세표 및 리포트 다운로드
    # -----------------------------------------------------------------------------
    with st.expander("📋 제조번호별 사진별 상세 측정 명세표 및 보고서", expanded=True):
        spec_table = f"""
        | 사진 파일명 | 측정 수량 (순번 완료) | 예상 중량 (kg) | 비고 |
        | :--- | :--- | :--- | :--- |
        """
        report_list = []

        for img_key, data in st.session_state.image_data.items():
            cnt = len(data["centers"])
            wt = cnt * unit_weight_kg
            fname = data["file"].name
            spec_table += f"| **{fname}** | {cnt} EA | {wt:,.1f} kg | 메인 묶음 개별 산출 | \n"

            report_list.append({
                "제조번호(Lot No.)": lot_number,
                "파일명": fname,
                "선경(mm)": bar_diameter,
                "길이(mm)": bar_length_mm,
                "수량(EA)": cnt,
                "단위중량(kg)": round(unit_weight_kg, 2),
                "예상총중량(kg)": round(wt, 2),
            })

        spec_table += f"""
        | **[합계 / Lot: {lot_number}]** | **{grand_total_count} EA** | **{grand_total_weight_kg:,.2f} kg ({grand_total_weight_ton:.3f} Ton)** | **총 {len(uploaded_files)}장 통합** |
        """
        st.markdown(spec_table)

        # CSV/엑셀 다운로드 기능 제공
        df_report = pd.DataFrame(report_list)
        csv_data = df_report.to_csv(index=False).encode('utf-8-sig')

        st.download_button(
            label="📥 엑셀(CSV) 검수 보고서 다운로드",
            data=csv_data,
            file_name=f"CD_BAR_Report_{lot_number}.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.info("👆 상단의 업로드 창을 통해 CD-BAR 단면 사진(1장 이상)을 등록해 주세요.")
