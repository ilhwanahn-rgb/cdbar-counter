import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="선재/이형재 인발 소성가공 종합 산출 도구", layout="wide")

# --- 탭 및 KEY-IN 입력창 시안성 극대화 CSS ---
st.markdown("""
    <style>
    /* 전체 배경: 눈이 편안한 모던 슬레이트 톤 */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    
    /* 1. 탭(Tab) 전체 영역 컨테이너 여백 */
    div[data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: transparent !important;
        padding-bottom: 8px !important;
        border-bottom: 2px solid #cbd5e1 !important;
    }

    /* 탭 버튼 기본 스타일 (입체 카드 형태) */
    button[data-baseweb="tab"], [data-testid="stTab"] {
        font-size: 19px !important;
        font-weight: 800 !important;
        color: #334155 !important;
        background-color: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 10px 10px 0px 0px !important;
        padding: 12px 24px !important;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease-in-out !important;
    }

    /* 탭 마우스 호버 효과 */
    button[data-baseweb="tab"]:hover {
        background-color: #e2e8f0 !important;
        color: #1e3a8a !important;
        border-color: #94a3b8 !important;
    }

    /* 선택된 활성화 탭 (강력한 부각 및 딥블루 하이라이트) */
    button[data-baseweb="tab"][aria-selected="true"], [data-testid="stTab"][aria-selected="true"] {
        color: #ffffff !important;
        background-color: #1e40af !important; /* 강렬한 딥블루 */
        border: 2px solid #1d4ed8 !important;
        border-bottom: 5px solid #2563eb !important;
        box-shadow: 0px 4px 12px rgba(30, 64, 175, 0.35) !important;
        transform: translateY(-2px) !important;
    }
    
    /* 2. 입력창 라벨(제목) 글자 강조 */
    div[data-testid="stWidgetLabel"] label p {
        font-weight: 800 !important;
        color: #1e3a8a !important; /* 진한 남색 */
        font-size: 16px !important;
    }

    /* 3. KEY-IN (숫자 입력창) 전용 하이라이트 - 엑셀 입력셀 스타일 (연노랑 배경 + 파란 테두리) */
    .stNumberInput input {
        background-color: #fefce8 !important; /* 파스텔 연노랑 */
        color: #0f172a !important;
        border: 2.5px solid #2563eb !important; /* 선명한 파란색 테두리 */
        border-radius: 8px !important;
        font-weight: 800 !important;
        font-size: 17px !important;
        padding: 8px 12px !important;
    }
    .stNumberInput input:focus {
        background-color: #ffffff !important;
        border-color: #1d4ed8 !important;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.25) !important;
    }

    /* 4. 드롭다운 (Selectbox) KEY-IN 하이라이트 - (파스텔 스카이블루 배경 + 딥사이안 테두리) */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #f0f9ff !important; /* 파스텔 스카이블루 */
        color: #0f172a !important;
        border: 2.5px solid #0284c7 !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
        font-size: 16px !important;
    }

    /* 5. 계산 결과 지표 카드(Metric) 입체 스타일 (화이트 배경으로 입력창과 확실히 구분) */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-size: 15px !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricValue"] {
        color: #1e40af !important;
        font-weight: 800 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛠️ 선재/이형재 인발 소성가공 종합 산출 도구")
st.markdown("단면 감면율 및 3D 시각화, 인발력(형상별 & 95% 설비검증), 중량 계산, 직진도 환산 연산 통합 도구입니다.")

# --- 탭 구성 (4개 독립 모듈) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📐 1. 형상별 감면율 & 3D", 
    "⚡ 2. 인발력 산출 (형상/TS/95%설비검증)", 
    "⚖️ 3. 중량 계산 (원형/사각/육각)",
    "📏 4. 직진도 환산"
])

# ==========================================
# 공통 헬퍼 함수 (2D 단면 정점 생성)
# ==========================================
def generate_shape_points(shape, w, h, r, n_points=120):
    if shape == "정육각형":
        pts = []
        r_center = (w - 2 * r) / np.sqrt(3)
        for i in range(6):
            angle_c = np.pi/6 + i * np.pi/3
            cx, cy = r_center * np.cos(angle_c), r_center * np.sin(angle_c)
            arc_angles = np.linspace(angle_c - np.pi/6, angle_c + np.pi/6, n_points // 6)
            for a in arc_angles:
                pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
        pts = np.array(pts)
        return pts[:, 0], pts[:, 1]
        
    elif shape == "사각형 (정/직사각)":
        hw, hh = w / 2.0 - r, h / 2.0 - r
        centers = [(hw, hh), (-hw, hh), (-hw, -hh), (hw, -hh)]
        arcs = [(0, np.pi/2), (np.pi/2, np.pi), (np.pi, 3*np.pi/2), (3*np.pi/2, 2*np.pi)]
        pts = []
        for (cx, cy), (sa, ea) in zip(centers, arcs):
            arc = np.linspace(sa, ea, n_points // 4)
            for a in arc:
                pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
        pts = np.array(pts)
        return pts[:, 0], pts[:, 1]
        
    else: # 트랙형 (장원형)
        r_track = h / 2.0
        straight_len = max(0.0, (w - h) / 2.0)
        pts = []
        for a in np.linspace(-np.pi/2, np.pi/2, n_points // 2):
            pts.append([straight_len + r_track * np.cos(a), r_track * np.sin(a)])
        for a in np.linspace(np.pi/2, 3*np.pi/2, n_points // 2):
            pts.append([-straight_len + r_track * np.cos(a), r_track * np.sin(a)])
        pts = np.array(pts)
        return pts[:, 0], pts[:, 1]

# ==========================================
# [TAB 1] 형상별 감면율 및 3D 시각화
# ==========================================
with tab1:
    st.subheader("1. 형상별 감면율 산출 및 2D/3D 시각화")
    st.markdown("✍️ **[노란색/파란색 박스]**에 입력 치수를 KEY-IN하여 연산 결과를 확인하세요.")
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        d_in = st.number_input("✍️ 입력 원형 선경 d (mm)", value=30.0, min_value=1.0, step=0.5, key="t1_din")
        shape_type = st.selectbox("📌 목표 단면 형상 선택", ["정육각형", "사각형 (정/직사각)", "이형 (트랙/장원형)"], key="t1_shape")

    with col_in2:
        if shape_type == "정육각형":
            W = st.number_input("✍️ 대면 치수 W (mm)", value=28.0, step=0.5, key="t1_w")
            R = st.slider("🎚️ 모서리 R (mm)", 0.0, float(W/2.0), 2.9, 0.1, key="t1_r")
            H = W
            max_diag = (2 * W / np.sqrt(3)) - 2 * R * ((2 / np.sqrt(3)) - 1)
            A2 = (np.sqrt(3) / 2.0) * (W ** 2) - (2 * np.sqrt(3) - np.pi) * (R ** 2)

        elif shape_type == "사각형 (정/직사각)":
            W = st.number_input("✍️ 폭 W (mm)", value=25.0, step=0.5, key="t1_w_sq")
            H = st.number_input("✍️ 높이 H (mm)", value=25.0, step=0.5, key="t1_h_sq")
            R = st.slider("🎚️ 모서리 R (mm)", 0.0, float(min(W, H)/2.0), 1.0, 0.1, key="t1_r_sq")
            max_diag = np.sqrt(W**2 + H**2) - 2 * R * (np.sqrt(2) - 1)
            A2 = W * H - (4.0 - np.pi) * (R ** 2)

        else: # 트랙형
            W = st.number_input("✍️ 전체 폭 W (mm)", value=30.0, step=0.5, key="t1_w_tr")
            H = st.number_input("✍️ 높이 H (mm)", value=18.0, step=0.5, key="t1_h_tr")
            R = H / 2.0
            max_diag = W
            A2 = (W - H) * H + (np.pi / 4.0) * (H ** 2)

    A1 = (np.pi / 4.0) * (d_in ** 2)
    RA = (1.0 - A2 / A1) * 100.0 if A1 > 0 else 0.0
    elongation = A1 / A2 if A2 > 0 else 0.0
    d_eq = np.sqrt(4 * A2 / np.pi) if A2 > 0 else 0.0

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("소재 원형 단면적 (A₁)", f"{A1:.2f} mm²", f"원형 직경 Ø {d_in:.2f} mm", delta_color="off")
    c2.metric("성형 후 단면적 (A₂)", f"{A2:.2f} mm²", f"등가원경 Ø {d_eq:.2f} mm", delta_color="off")
    c3.metric("최대 대각/외경 치수 (D)", f"{max_diag:.2f} mm", f"대면 W {W:.2f} mm / R {R:.2f} mm", delta_color="off")
    c4.metric("감면율 (RA)", f"{RA:.2f} %", f"연신율 {elongation:.2f} 배", delta_color="normal")

    col_l, col_r = st.columns(2)
    n_pts = 120
    x_in = (d_in / 2.0) * np.cos(np.linspace(0, 2*np.pi, n_pts, endpoint=False))
    y_in = (d_in / 2.0) * np.sin(np.linspace(0, 2*np.pi, n_pts, endpoint=False))
    x_out, y_out = generate_shape_points(shape_type, W, H, R, n_points=n_pts)

    fig_2d = go.Figure()
    fig_2d.add_trace(go.Scatter(x=x_in, y=y_in, mode='lines', name=f'입력 원형 (Ø{d_in:.1f}mm)', line=dict(color='#64748b', dash='dash', width=2)))
    fig_2d.add_trace(go.Scatter(x=x_out, y=y_out, mode='lines', name=f'출력 {shape_type}', fill="toself", fillcolor='rgba(37, 99, 235, 0.2)', line=dict(color='#1d4ed8', width=3)))
    fig_2d.update_layout(title="<b>2D 단면 비교 (Cross-Section Overlay)</b>", xaxis=dict(scaleanchor="y", scaleratio=1), height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#ffffff', font=dict(color='#0f172a'))
    col_l.plotly_chart(fig_2d, use_container_width=True)

    z_levels = np.linspace(0, 100, 30)
    X_3d, Y_3d, Z_3d = [], [], []
    for z in z_levels:
        factor = 0.0 if z <= 30 else (1.0 if z >= 80 else (z - 30) / 50.0)
        x_curr = (1 - factor) * x_in + factor * x_out
        y_curr = (1 - factor) * y_in + factor * y_out
        X_3d.extend(x_curr); Y_3d.extend(y_curr); Z_3d.extend([z] * n_pts)

    I, J, K = [], [], []
    for i in range(len(z_levels) - 1):
        for j in range(n_pts):
            next_j = (j + 1) % n_pts
            p1, p2 = i * n_pts + j, i * n_pts + next_j
            p3, p4 = (i + 1) * n_pts + j, (i + 1) * n_pts + next_j
            I.extend([p1, p2]); J.extend([p2, p4]); K.extend([p3, p3])

    fig_3d = go.Figure(data=[go.Mesh3d(x=X_3d, y=Y_3d, z=Z_3d, i=I, j=J, k=K, intensity=Z_3d, colorscale='Blues', opacity=0.9)])
    fig_3d.update_layout(title="<b>3D 솔리드 인발 파이프라인</b>", scene=dict(aspectmode='data'), height=420, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#0f172a'))
    col_r.plotly_chart(fig_3d, use_container_width=True)

# ==========================================
# [TAB 2] 인발력 산출 (PDF 분류 기준 DB 적용)
# ==========================================
with tab2:
    st.subheader("2. 형상별 인발력 및 설비 부하(95% 한계) 검증")
    st.markdown("✍️ **[노란색/파란색 박스]**에 입력 치수 및 강종을 KEY-IN하여 **현장 엑셀 인발력**과 설비 가동 여부를 연산합니다.")

    machines_db = [
        {"name": "CD-0-2호기", "min_d": 3.8,  "max_d": 6.0,  "max_cap": 2.0},
        {"name": "CD-0-3호기", "min_d": 5.49, "max_d": 9.0,  "max_cap": 2.0},
        {"name": "CD-1호기",   "min_d": 8.0,  "max_d": 13.0, "max_cap": 5.0},
        {"name": "CD-5호기",   "min_d": 9.0,  "max_d": 15.0, "max_cap": 6.5},
        {"name": "CD-2-2호기", "min_d": 14.0, "max_d": 18.0, "max_cap": 8.0},
        {"name": "CD-3호기",   "min_d": 16.0, "max_d": 24.0, "max_cap": 15.0},
        {"name": "CD-4호기",   "min_d": 19.0, "max_d": 41.0, "max_cap": 25.0},
    ]

    steel_categories = {
        "1. 전자연철봉": {
            "SUYB1": 33.5
        },
        "2. 냉간압조용 탄소강 (SWRCH / Boron)": {
            "SWRCH6A": 33.1,
            "SWRCH8A": 34.2,
            "SWRCH10A (10A)": 35.5,
            "SWRCH12A (12A)": 38.8,
            "SWRCH15K": 41.4,
            "SWRCH18A": 46.4,
            "SWRCH20K": 44.4,
            "SWRCH22A": 47.1,
            "SWRCH25K(F)": 49.5,
            "SWRCH30K": 58.3,
            "SWRCH35K(F)": 60.5,
            "SWRCH38K(F)": 60.9,
            "SWRCH45K(F) (45K)": 64.7,
            "AISI/SAE 10B21": 51.1,
            "AISI/SAE 10B30": 57.6,
            "AISI/SAE 10B35": 60.8,
            "AISI/SAE 10B38": 64.1
        },
        "3. 기계구조용강 (S-C계열)": {
            "S20C": 48.5,
            "S25C": 51.6,
            "S35C": 69.9,
            "S45C (W/R)": 71.0,
            "S48C": 78.2
        },
        "4. 경화능 보증 구조용강 (H계열)": {
            "SCr415H": 52.2,
            "SCr420H": 58.2,
            "SCM415 (W/R)": 60.0,
            "SCM420 (W/R)": 79.0,
            "SCM435 (W/R)": 96.0,
            "SCM440 (W/R)": 104.0,
            "W/R-SNCM220H": 73.0,
            "LA-SNCM220H (SL04)": 58.0
        },
        "5. 베어링 / 스프링 / 고온합금강": {
            "SUJ2 (베어링강)": 115.0,
            "SUP9 (스프링강)": 95.0,
            "SNB16 (고온합금강볼트)": 118.2,
            "SA-100CRMNS7-4": 80.0
        },
        "6. 쾌삭강 (SUM)": {
            "SUM22 (W/R)": 40.0,
            "SUM24L (W/R)": 42.0,
            "SUM43 (W/R)": 69.0,
            "AISI/SAE 1151": 72.3
        },
        "7. 스테인리스강 (STS / SUS)": {
            "SUS303C": 52.8,
            "SUS303F": 59.9,
            "SUS304 (W/R)": 58.0,
            "SUS316L (W/R)": 54.0,
            "SUS410": 57.9,
            "SUS416": 56.8,
            "SUS420J2": 68.5,
            "SUS430F": 56.6,
            "W/R-SUS440C": 77.0,
            "XM7 (원재)": 75.0,
            "XM7 (12% 인발시)": 94.0
        },
        "8. 기타 열처리 & 합금/포스코강": {
            "W/R-SNCM439": 110.0,
            "SA-SNCM439": 71.0,
            "AISI/SAE 1050SH": 82.5,
            "AISI/SAE 1060S": 87.1,
            "AISI/SAE 1541": 81.0,
            "AISI/SAE 4140": 114.0,
            "AISI/SAE 4037": 65.1,
            "AISI/SAE 9254": 96.7,
            "POSMA45R": 82.6,
            "POSMA45RM": 76.0,
            "POSA1038B": 64.2,
            "POSA1021B": 52.6,
            "POSA5120BH": 53.7
        },
        "9. 사용자 직접 입력": {
            "직접 입력": 40.0
        }
    }

    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown("#### 📥 변형 유형 및 KEY-IN 치수 입력")
        draw_mode = st.selectbox("📌 인발 변형 유형 선택", ["1. 원형 - 원형", "2. 원형 - 사각", "3. 원형 - 육각"], key="t2_mode")
        
        d_in_t2 = st.number_input("✍️ 투입 원형 선경 W/ROD (MM)", value=32.0, min_value=1.0, step=0.5, key="t2_din_keyin")

        if draw_mode == "1. 원형 - 원형":
            d_out_t2 = st.number_input("✍️ 제품 원형 선경 (MM)", value=26.0, min_value=0.5, step=0.1, key="t2_dout_rd")
            a2_t2 = (np.pi / 4.0) * (d_out_t2 ** 2)
            diag_t2 = d_out_t2
            prod_size_for_m = d_out_t2

        elif draw_mode == "2. 원형 - 사각":
            w_out_t2 = st.number_input("✍️ 제품 사각 한변 치수 (MM)", value=18.0, min_value=0.5, step=0.1, key="t2_wout_sq")
            a2_t2 = w_out_t2 ** 2
            diag_t2 = w_out_t2 * np.sqrt(2.0)
            prod_size_for_m = w_out_t2

        else: # 원형 - 육각
            w_out_t2 = st.number_input("✍️ 제품 육각 대면 치수 W (MM)", value=26.0, min_value=0.5, step=0.1, key="t2_wout_hex")
            a2_t2 = (np.sqrt(3.0) / 2.0) * (w_out_t2 ** 2)
            diag_t2 = (2.0 * w_out_t2) / np.sqrt(3.0)
            prod_size_for_m = w_out_t2

        st.markdown("---")
        st.markdown("#### 🧬 강종 선택 (2단계 분류 체계)")
        cat_choice = st.selectbox("📌 1단계: 강종 분류 선택", list(steel_categories.keys()), key="t2_cat")
        sub_steels = steel_categories[cat_choice]
        steel_choice = st.selectbox("📌 2단계: 세부 강종 선택", list(sub_steels.keys()), key="t2_steel")

        if cat_choice == "9. 사용자 직접 입력" or steel_choice == "직접 입력":
            ts_kgf = st.number_input("✍️ T.S (W/ROD) (kgf/mm²)", value=40.0, step=1.0, key="t2_custom_ts")
        else:
            ts_kgf = sub_steels[steel_choice]

    a1_t2 = (np.pi / 4.0) * (d_in_t2 ** 2)
    ra_ratio = (a1_t2 - a2_t2) / a1_t2 if a1_t2 > 0 else 0.0
    ra_percent = ra_ratio * 100.0

    with col_f2:
        st.markdown("#### 📐 자동 계산된 단면 및 감면율")
        st.write(f"• **투입 면적 (A₁):** `{a1_t2:.2f} mm²`")
        st.write(f"• **제품 면적 (A₂):** `{a2_t2:.3f} mm²`")
        st.write(f"• **감면율 (RA):** `{ra_ratio:.6f}` (`{ra_percent:.2f}%`)")
        st.write(f"• **제품 대각 치수 (D):** `{diag_t2:.3f} mm`")
        st.write(f"• **적용 강종 분류:** `{cat_choice}` ➔ `{steel_choice}`")
        st.write(f"• **적용 T.S (W/ROD):** `{ts_kgf:.1f} kgf/mm²`")

    if a1_t2 > a2_t2 and a2_t2 > 0:
        force_ton = (1.25 / 0.35) * a2_t2 * ts_kgf * (0.03 + 0.55 * ra_ratio) / 1000.0

        st.markdown("---")
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("산출 인발력 (TON)", f"{force_ton:.3f} Ton", f"{force_ton * 9.80665:.2f} kN")
        m_c2.metric("제품 대각 치수", f"{diag_t2:.2f} mm", f"변형 유형: {draw_mode}")

        st.info("💡 **적용 엑셀 공식:** 인발력 (TON) = 1.25 / 0.35 × 제품면적(A₂) × T.S × (0.03 + 0.55 × 감면율비율) / 1000")

        st.markdown("---")
        st.markdown("### 🏭 설비별 작업 가능 여부 검증 (설비 능력 95% 제한 기준)")

        m_eval_data = []
        matched_machines = []

        for m in machines_db:
            size_ok = (m["min_d"] <= prod_size_for_m <= m["max_d"])
            usable_cap = m["max_cap"] * 0.95
            force_ok = (force_ton <= usable_cap)
            load_ratio = (force_ton / usable_cap) * 100.0 if usable_cap > 0 else 0.0

            if size_ok and force_ok:
                status = "🟢 작업 가능 (이상없음)"
                matched_machines.append(f"**{m['name']}** (부하율 {load_ratio:.1f}%)")
            elif size_ok and not force_ok:
                status = "🔴 인발력 초과 (작업불가)"
            else:
                status = "⚪ 선경 규격 미달/초과"

            m_eval_data.append({
                "작업 호기": m["name"],
                "작업 가능 제품선경": f"{m['min_d']} ~ {m['max_d']} mm",
                "설비 Max 톤수": f"{m['max_cap']:.1f} t",
                "95% 한계 인발력": f"{usable_cap:.2f} t",
                "소요 인발력": f"{force_ton:.3f} t",
                "설비 부하율": f"{load_ratio:.1f} %",
                "판정 결과": status
            })

        if matched_machines:
            st.success(f"✅ **현재 작업 조건({draw_mode} / {force_ton:.3f}t)에 이상이 없는 추천 설비:** " + ", ".join(matched_machines))
        else:
            st.error(f"⚠️ **경고:** 현재 소요 인발력({force_ton:.3f}t) 조건에 안전하게(95% 이내) 작업할 수 있는 설비가 없습니다.")

        df_m = pd.DataFrame(m_eval_data)
        st.dataframe(df_m, use_container_width=True)

    else:
        st.warning("투입 선경이 제품 단면보다 커야 인발력 연산이 가능합니다.")

# ==========================================
# [TAB 3] 중량 계산 (원형/사각/육각 지원)
# ==========================================
with tab3:
    st.subheader("3. 선재 / 봉재 규격별 중량 계산 (원형/사각/육각)")
    st.markdown("✍️ **[노란색/파란색 박스]**에 단면 치수, 길이, 비중을 KEY-IN하여 중량을 산출합니다.")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("#### 📐 제품 형상 및 치수 입력")
        bar_shape = st.selectbox("📌 제품 형상 선택", ["원형 (Round Bar)", "사각형 (Square / Rect Bar)", "정육각형 (Hexagon Bar)"], key="w_shape")

        if bar_shape == "원형 (Round Bar)":
            d_calc = st.number_input("✍️ 외경 직경 D (mm)", value=25.0, step=0.5, key="w_d")
            calc_area = (np.pi / 4.0) * (d_calc ** 2)
            shape_desc = f"원형 직경 Ø {d_calc:.2f} mm"

        elif bar_shape == "사각형 (Square / Rect Bar)":
            col_sq1, col_sq2 = st.columns(2)
            with col_sq1:
                w_sq = st.number_input("✍️ 폭 W (mm)", value=25.0, step=0.5, key="w_sq_w")
            with col_sq2:
                h_sq = st.number_input("✍️ 높이 H (mm)", value=25.0, step=0.5, key="w_sq_h")
            calc_area = w_sq * h_sq
            shape_desc = f"사각 W {w_sq:.2f} mm × H {h_sq:.2f} mm"

        else: # 정육각형 (Hexagon Bar)
            w_hex = st.number_input("✍️ 대면 치수 W (mm)", value=25.0, step=0.5, key="w_hex_w")
            calc_area = (np.sqrt(3.0) / 2.0) * (w_hex ** 2)
            shape_desc = f"육각 대면 W {w_hex:.2f} mm"

        length_mm = st.number_input("✍️ 제품 1본당 길이 L (mm)", value=3020.0, step=10.0, key="w_l")
        
        density_dict = {
            "Carbon Steel (7.85)": 7.85,
            "Stainless Steel 304 (7.93)": 7.93,
            "Stainless Steel 316 (7.98)": 7.98,
            "Stainless Steel 420 (7.70)": 7.70,
            "Stainless Steel 430 (7.70)": 7.70,
            "사용자 직접 입력": 7.85
        }
        mat_choice = st.selectbox("📌 재질 비중 (Specific Gravity Sg)", list(density_dict.keys()), key="w_mat")
        
        if mat_choice == "사용자 직접 입력":
            rho = st.number_input("✍️ 비중 직접 입력", value=7.85, step=0.01, key="w_rho")
        else:
            rho = density_dict[mat_choice]

    with col_w2:
        st.markdown("#### 📦 수량 입력")
        quantity = st.number_input("✍️ 총 수량 (EA)", value=1, step=1, key="w_qty")
        
        piece_weight_kg = calc_area * length_mm * rho * (10 ** -6)
        piece_weight_lb = piece_weight_kg * 2.20462
        
        total_weight_kg = piece_weight_kg * quantity
        total_weight_ton = total_weight_kg / 1000.0

    st.markdown("---")
    st.markdown("### 📊 중량 산출 연산 결과")
    wc1, wc2, wc3 = st.columns(3)
    
    wc1.metric("단품 1본 중량 (kg)", f"{piece_weight_kg:.3f} kg", f"단면적: {calc_area:.2f} mm² ({shape_desc})", delta_color="off")
    wc2.metric("단품 1본 중량 (lb)", f"{piece_weight_lb:.3f} lb", delta_color="off")
    wc3.metric(f"총 중량 (Total Weight, {quantity} EA)", f"{total_weight_kg:.2f} kg", f"{total_weight_ton:.4f} Ton")

    st.info("💡 **적용 공식:** 중량(kg) = 단면적(A, mm²) × 길이(L, mm) × 비중(Sg) × 10⁻⁶")

# ==========================================
# [TAB 4] 직진도 환산 (엑셀 수식 적용)
# ==========================================
with tab4:
    st.subheader("4. 환산 직진도 계산기")
    st.markdown("✍️ **[노란색/파란색 박스]**에 요구 길이/직진도 및 생산 제품 길이를 KEY-IN하여 **환산 직진도**를 연산합니다.")
    
    st.info("💡 **적용 공식:** 환산 직진도 = (직진도 × 제품길이²) / 수요가길이²")

    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown("#### 📏 수요가 기준 (Input)")
        req_length = st.number_input("✍️ 수요가길이 (mm)", value=4920.0, step=10.0, key="s_req_l")
        req_straightness = st.number_input("✍️ 직진도 (mm)", value=1.000, step=0.01, format="%.3f", key="s_req_s")

    with col_s2:
        st.markdown("#### 🏭 제품 기준 (Input)")
        prod_length = st.number_input("✍️ 제품길이 (mm)", value=1000.0, step=10.0, key="s_prod_l")

    if req_length > 0:
        conv_straightness = (req_straightness * (prod_length ** 2)) / (req_length ** 2)
    else:
        conv_straightness = 0.0

    with col_s3:
        st.markdown("#### ✅ 계산 결과 (Output)")
        st.metric("환산 직진도", f"{conv_straightness:.3f} mm")