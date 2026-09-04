import json
import math
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 커스텀 CSS 스타일 적용 (폰트 +5pt)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CD-BAR 스마트 카운팅 & 중량 분석 시스템",
    page_icon="🔩",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8fafc; }
    html, body, p, span, label, div, .stMarkdown { font-size: 1.15rem !important; }
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff; padding: 28px; border-radius: 16px; margin-bottom: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .header-card h1 { color: #ffffff !important; font-size: 2.3rem !important; font-weight: 800 !important; margin-bottom: 8px !important; }
    .header-card p { color: #94a3b8 !important; font-size: 1.25rem !important; margin-bottom: 0 !important; }
    .metric-container {
        background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 18px; text-align: center; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .metric-title { font-size: 1.15rem !important; color: #64748b; font-weight: 700; margin-bottom: 6px; }
    .metric-val-main { font-size: 2.1rem !important; font-weight: 800; color: #0f172a; }
    .metric-sub { font-size: 1.1rem !important; color: #10b981; font-weight: 700; margin-top: 6px; }
    
    .guide-box {
        background-color: #f0f9ff; border-left: 5px solid #0284c7; padding: 16px 20px;
        border-radius: 8px; font-size: 1.2rem !important; color: #0369a1; margin-bottom: 20px; line-height: 1.6;
    }
    
    div[data-testid="stRadio"] > label {
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        color: #0f172a !important;
    }
    
    table, th, td { font-size: 1.15rem !important; padding: 12px !important; }
    .stSidebar label, .stSidebar p, .stSidebar div { font-size: 1.1rem !important; }
    .stImageCoordinates { display: flex; justify-content: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="header-card">
        <h1>🔩 CD-BAR 스마트 카운팅 & 지속적 AI 재학습 시스템</h1>
        <p>강화된 중복 원 방지 엔진 · 박스 영역 일괄 편집 · 단면 순번 오버레이 · YOLO 라벨 자동 생성</p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception:
    st.error("❌ 'best.pt' 모델을 불러올 수 없습니다. GitHub 저장소를 확인해 주세요.")
    st.stop()

# -----------------------------------------------------------------------------
# 2. 사이드바 제어 설정
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 시스템 설정")

    lot_number_input = st.text_input("🏷️ 제조번호 (Lot No.)", value="", placeholder="예: P265MD123-01-10")
    lot_number = lot_number_input.strip() if lot_number_input.strip() else "미입력"

    st.markdown("---")
    st.subheader("🎯 AI 탐지 및 묶음 제어")
    conf_thresh = st.slider("탐지 민감도 (Confidence)", 0.03, 0.70, 0.12, 0.01)
    tolerance = st.slider("메인 묶음 선경 오차 범위 (%)", 10, 60, 40, 5) / 100.0

    use_tiling = st.checkbox("🧩 분할 타일링 스캔 (초밀집/작은 단면용)", value=True)
    use_cluster_filter = st.checkbox("✂️ 외곽 타 묶음 자동 제거 (군집 필터)", value=False)

    st.markdown("---")
    st.subheader("📏 제품 규격 및 중량 설정")
    bar_diameter = st.number_input("선경 (지름, mm)", 1.0, 200.0, 12.0, 0.5)
    bar_length_mm = st.number_input("제품 길이 (L, mm)", 100, 30000, 6000, 1)
    steel_density = st.number_input("철 비중 (g/cm³)", 7.0, 9.0, 7.85, 0.01)

    st.markdown("---")
    if st.button("🧹 전체 사진 분석 데이터 초기화", use_container_width=True):
        st.session_state.image_data = {}
        st.session_state.box_start_point = None
        st.rerun()

if "image_data" not in st.session_state:
    st.session_state.image_data = {}
if "box_start_point" not in st.session_state:
    st.session_state.box_start_point = None

# -----------------------------------------------------------------------------
# 3. 타일링 추론 및 데이터셋 저장 함수
# -----------------------------------------------------------------------------
def predict_with_slicing(img, model_obj, conf):
    w, h = img.size
    boxes_list = []

    if use_tiling:
        crop_w, crop_h = w // 2, h // 2
        overlaps = [
            (0, 0, crop_w + 50, crop_h + 50),
            (crop_w - 50, 0, w, crop_h + 50),
            (0, crop_h - 50, crop_w + 50, h),
            (crop_w - 50, crop_h - 50, w, h)
        ]

        for x1, y1, x2, y2 in overlaps:
            crop_img = img.crop((x1, y1, x2, y2))
            res = model_obj(crop_img, conf=conf, imgsz=1024)
            if len(res[0].boxes) > 0:
                b = res[0].boxes.xyxy.cpu().numpy()
                b[:, [0, 2]] += x1
                b[:, [1, 3]] += y1
                boxes_list.append(b)

        if boxes_list:
            all_boxes = np.vstack(boxes_list)
        else:
            all_boxes = np.empty((0, 4))
    else:
        res = model_obj(img, conf=conf, imgsz=1280)
        if len(res[0].boxes) > 0:
            all_boxes = res[0].boxes.xyxy.cpu().numpy()
        else:
            all_boxes = np.empty((0, 4))

    return all_boxes

def save_yolo_dataset():
    base_dir = "collected_dataset"
    img_dir = os.path.join(base_dir, "images")
    lbl_dir = os.path.join(base_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    saved_count = 0
    for img_key, data in st.session_state.image_data.items():
        fname = data["file"].name
        fname_no_ext = os.path.splitext(fname)[0]
        
        img_path = os.path.join(img_dir, fname)
        data["image"].save(img_path)

        txt_path = os.path.join(lbl_dir, f"{fname_no_ext}.txt")
        W, H = data["image"].size
        radius = data["radius"]
        box_w = radius * 2.0
        box_h = radius * 2.0

        with open(txt_path, "w") as f:
            for cx, cy, _ in data["centers"]:
                x_center = cx / W
                y_center = cy / H
                w_norm = box_w / W
                h_norm = box_h / H
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        saved_count += 1
    return saved_count

# -----------------------------------------------------------------------------
# 4. 이미지 프로세스 및 메인 연산
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

            max_size = 1400
            if max(image.width, image.height) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            all_boxes = predict_with_slicing(image, model, conf_thresh)

            raw_ai_centers = []
            target_radius = 12

            if len(all_boxes) > 0:
                widths = all_boxes[:, 2] - all_boxes[:, 0]
                heights = all_boxes[:, 3] - all_boxes[:, 1]

                median_w = np.median(widths)
                median_h = np.median(heights)
                target_radius = max(3, int((median_w + median_h) / 4))

                for box, w, h in zip(all_boxes, widths, heights):
                    if (1.0 - tolerance) * median_w <= w <= (1.0 + tolerance) * median_w and \
                       (1.0 - tolerance) * median_h <= h <= (1.0 + tolerance) * median_h:
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        raw_ai_centers.append((cx, cy))

                # [핵심 보완] 거리 기반 NMS 임계값을 반지름의 1.2배로 강화하여 이중 원 완벽 제거
                clean_ai_centers = []
                for cx, cy in raw_ai_centers:
                    is_duplicate = False
                    for kx, ky in clean_ai_centers:
                        if math.hypot(cx - kx, cy - ky) < (target_radius * 1.2):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        clean_ai_centers.append((cx, cy))

                if use_cluster_filter and len(clean_ai_centers) > 0:
                    max_connect_dist = target_radius * 3.5
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
                    final_ai_centers = [(cx, cy, True) for cx, cy in clean_ai_centers]
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

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="metric-container"><div class="metric-title">제조번호 (Lot No.)</div><div class="metric-val-main" style="font-size: 1.4rem !important; color: #0284c7;">{lot_number}</div><div class="metric-sub">검수 대상 번호</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-container"><div class="metric-title">총 검수 수량 (합계)</div><div class="metric-val-main">{grand_total_count} <span style="font-size:1.2rem;">개</span></div><div class="metric-sub">총 {len(uploaded_files)}장 통합</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-container"><div class="metric-title">총 중량 (합계)</div><div class="metric-val-main">{grand_total_weight_kg:,.1f} <span style="font-size:1.2rem;">kg</span></div><div class="metric-sub">{grand_total_weight_ton:.3f} Ton</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-container"><div class="metric-title">1본당 단위 중량</div><div class="metric-val-main">{unit_weight_kg:.2f} <span style="font-size:1.2rem;">kg</span></div><div class="metric-sub">비중 {steel_density} 기준</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="metric-container"><div class="metric-title">설정 제품 규격</div><div class="metric-val-main">Ø{bar_diameter:.1f}</div><div class="metric-sub">{bar_length_mm:,} mm</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    selected_file = st.selectbox(
        "📸 검수 및 수동 클릭 보정을 진행할 사진을 선택하세요:",
        uploaded_files,
        format_func=lambda x: f"📷 {x.name} (현재 카운트: {len(st.session_state.image_data[f'{x.name}_{x.size}']['centers'])}개)",
    )

    current_id = f"{selected_file.name}_{selected_file.size}"
    active_data = st.session_state.image_data[current_id]

    st.markdown("### 🛠️ 수동 보정 작업 모드 선택")
    work_mode = st.radio(
        "원하는 보정 작업 방식을 선택하세요:",
        ["🎯 1개씩 단일 클릭 추가/삭제", "📦 박스 영역 지정 일괄 삭제", "🔍 박스 영역 지정 AI 집중 재탐지"],
        horizontal=True,
        key="selected_work_mode"
    )

    if "last_mode" not in st.session_state or st.session_state.last_mode != work_mode:
        st.session_state.last_mode = work_mode
        st.session_state.box_start_point = None

    if work_mode == "🎯 1개씩 단일 클릭 추가/삭제":
        st.markdown('<div class="guide-box">👉 <b>단일 클릭 가이드:</b> 기존 원 클릭 ➔ 삭제 ❌ / 빈 공간 클릭 ➔ 1개 추가 🟡</div>', unsafe_allow_html=True)
    elif work_mode == "📦 박스 영역 지정 일괄 삭제":
        st.markdown('<div class="guide-box">👉 <b>박스 일괄 삭제 가이드:</b> [1차 클릭 = 시작 모서리 📍] ➔ [2차 클릭 = 대각선 반대 모서리 📍] 지정 시 <b>박스 안의 원이 모조리 삭제</b>됩니다.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="guide-box">👉 <b>박스 AI 집중 재탐지 가이드:</b> [1차 클릭 = 시작 모서리 📍] ➔ [2차 클릭 = 반대 모서리 📍] 지정 시 <b>해당 영역만 AI가 집중 스캔하여 놓친 원을 자동 생성</b>합니다.</div>', unsafe_allow_html=True)

    if st.session_state.box_start_point is not None:
        col_c1, col_c2 = st.columns([3, 1])
        with col_c1:
            st.info(f"📍 시작 모서리 지정 완료! (x: {st.session_state.box_start_point[0]}, y: {st.session_state.box_start_point[1]}) 대각선 반대쪽 모서리를 클릭하세요.")
        with col_c2:
            if st.button("❌ 박스 선택 취소", use_container_width=True):
                st.session_state.box_start_point = None
                st.rerun()

    image = active_data["image"]
    target_radius = active_data["radius"]

    row_band = max(10, int(target_radius * 1.6))
    sorted_centers = sorted(active_data["centers"], key=lambda item: (int(item[1] // row_band), item[0]))
    active_data["centers"] = sorted_centers

    output_img = image.copy()
    draw = ImageDraw.Draw(output_img)

    font_size = max(10, int(target_radius * 1.1))
    try: font = ImageFont.load_default(size=font_size)
    except: font = ImageFont.load_default()

    for idx, (cx, cy, is_ai) in enumerate(active_data["centers"], start=1):
        x0, y0 = cx - target_radius, cy - target_radius
        x1, y1 = cx + target_radius, cy + target_radius

        if is_ai: draw.ellipse([x0, y0, x1, y1], outline="#FF3333", width=3)
        else: draw.ellipse([x0, y0, x1, y1], outline="#FFCC00", width=3)

        num_str = str(idx)
        try:
            bbox = draw.textbbox((0, 0), num_str, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except: tw, th = 8, 10

        tx, ty = cx - (tw / 2), cy - (th / 2)
        draw.text((tx + 1, ty + 1), num_str, fill="black", font=font)
        draw.text((tx, ty), num_str, fill="white", font=font)

    if st.session_state.box_start_point is not None:
        bx, by = st.session_state.box_start_point
        draw.line([(bx - 20, by), (bx + 20, by)], fill="#00FFFF", width=3)
        draw.line([(bx, by - 20), (bx, by + 20)], fill="#00FFFF", width=3)
        draw.ellipse([bx - 6, by - 6, bx + 6, by + 6], outline="#00FFFF", width=3)

    value = streamlit_image_coordinates(output_img, key=f"canvas_{current_id}")

    if value is not None and value != active_data["last_clicked"]:
        active_data["last_clicked"] = value
        click_x, click_y = value["x"], value["y"]

        if work_mode == "🎯 1개씩 단일 클릭 추가/삭제":
            hit_index = None
            threshold_dist = max(target_radius, 12)

            for idx, (cx, cy, is_ai) in enumerate(active_data["centers"]):
                if math.hypot(cx - click_x, cy - click_y) <= threshold_dist:
                    hit_index = idx
                    break

            if hit_index is not None: active_data["centers"].pop(hit_index)
            else: active_data["centers"].append((click_x, click_y, False))
            st.rerun()

        elif work_mode in ["📦 박스 영역 지정 일괄 삭제", "🔍 박스 영역 지정 AI 집중 재탐지"]:
            if st.session_state.box_start_point is None:
                st.session_state.box_start_point = (click_x, click_y)
                st.rerun()
            else:
                x1, y1 = st.session_state.box_start_point
                x2, y2 = click_x, click_y
                
                min_x, max_x = min(x1, x2), max(x1, x2)
                min_y, max_y = min(y1, y2), max(y1, y2)

                if work_mode == "📦 박스 영역 지정 일괄 삭제":
                    new_centers = [
                        item for item in active_data["centers"]
                        if not (min_x <= item[0] <= max_x and min_y <= item[1] <= max_y)
                    ]
                    active_data["centers"] = new_centers
                
                elif work_mode == "🔍 박스 영역 지정 AI 집중 재탐지":
                    if (max_x - min_x) > 10 and (max_y - min_y) > 10:
                        roi_crop = image.crop((min_x, min_y, max_x, max_y))
                        roi_res = model(roi_crop, conf=max(0.03, conf_thresh * 0.7), imgsz=1024)
                        
                        if len(roi_res[0].boxes) > 0:
                            roi_boxes = roi_res[0].boxes.xyxy.cpu().numpy()
                            for box in roi_boxes:
                                cx = int((box[0] + box[2]) / 2) + min_x
                                cy = int((box[1] + box[3]) / 2) + min_y

                                # [동일 적용] ROI 재인식 시에도 1.2배 거리 적용으로 중복 생성 차단
                                is_dup = False
                                for ex_cx, ex_cy, _ in active_data["centers"]:
                                    if math.hypot(cx - ex_cx, cy - ex_cy) < (target_radius * 1.2):
                                        is_dup = True
                                        break
                                if not is_dup:
                                    active_data["centers"].append((cx, cy, True))

                st.session_state.box_start_point = None
                st.rerun()

    with st.expander("📋 상세 측정 명세표 및 AI 데이터 축적", expanded=True):
        spec_table = "| 사진 파일명 | 측정 수량 | 예상 중량 (kg) | 비고 |\n| :--- | :--- | :--- | :--- |\n"
        report_list = []

        for img_key, data in st.session_state.image_data.items():
            cnt = len(data["centers"])
            wt = cnt * unit_weight_kg
            fname = data["file"].name
            spec_table += f"| **{fname}** | {cnt} EA | {wt:,.1f} kg | 분석 개별 산출 |\n"
            report_list.append({
                "제조번호": lot_number, "파일명": fname, "선경(mm)": bar_diameter,
                "길이(mm)": bar_length_mm, "수량(EA)": cnt, "단위중량(kg)": round(unit_weight_kg, 2),
                "예상총중량(kg)": round(wt, 2)
            })

        spec_table += f"| **[합계 / Lot: {lot_number}]** | **{grand_total_count} EA** | **{grand_total_weight_kg:,.2f} kg ({grand_total_weight_ton:.3f} Ton)** | **총 {len(uploaded_files)}장 통합** |"
        st.markdown(spec_table)

        col_dl, col_train = st.columns(2)
        with col_dl:
            df_report = pd.DataFrame(report_list)
            st.download_button("📥 엑셀(CSV) 검수 보고서 다운로드", df_report.to_csv(index=False).encode('utf-8-sig'), f"CD_BAR_Report_{lot_number}.csv", "text/csv", use_container_width=True)

        with col_train:
            if st.button("🧠 현재 보정 결과를 AI 재학습 데이터셋으로 저장", use_container_width=True):
                n_saved = save_yolo_dataset()
                st.success(f"✅ 총 {n_saved}장의 보정 완료 사진과 YOLO 라벨(.txt)이 'collected_dataset/' 폴더에 자동 축적되었습니다!")

else:
    st.info("👆 상단의 업로드 창을 통해 CD-BAR 단면 사진(1장 이상)을 등록해 주세요.")
