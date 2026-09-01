from PIL import Image, ImageDraw
import numpy as np
import streamlit as st
from ultralytics import YOLO

# 1. 웹 페이지 기본 설정
st.set_page_config(page_title="CD-BAR 카운터", layout="centered")
st.title("🔩 CD-BAR 단면 자동 카운팅 시스템")
st.write("번들 특성에 맞춰 동일 선경 크기로 정규화하여 정밀 카운팅합니다.")


# 2. AI 모델(best.pt) 로드
@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:
    model = load_model()
except Exception as e:
    st.error(
        "❌ 'best.pt' 모델을 불러올 수 없습니다. GitHub 저장소에 best.pt 파일이 있는지 확인해주세요."
    )
    st.stop()

# 3. 이미지 업로드 UI
uploaded_file = st.file_uploader(
    "CD-BAR 촬영 사진을 업로드하세요", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # 4. YOLOv8 감지 실행
    with st.spinner("AI가 단면을 분석하고 동일 선경 규격화 중입니다..."):
        results = model(image, conf=0.35)
        boxes = results[0].boxes

        if len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            widths = xyxy[:, 2] - xyxy[:, 0]
            heights = xyxy[:, 3] - xyxy[:, 1]

            # [도메인 지식] 번들 대표 선경(중앙값) 산출
            median_w = np.median(widths)
            median_h = np.median(heights)
            target_radius = int((median_w + median_h) / 4)  # 동일 규격 반지름

            valid_centers = []

            for box, w, h in zip(xyxy, widths, heights):
                # 대표 크기 대비 ±25% 범위를 벗어나는 오탐지 자동 제거
                if (
                    0.75 * median_w <= w <= 1.25 * median_w
                    and 0.75 * median_h <= h <= 1.25 * median_h
                ):
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    valid_centers.append((cx, cy))

            # 5. PIL ImageDraw로 녹색 동일 원 및 중심점 그리기 (cv2 미사용)
            output_img = image.copy()
            draw = ImageDraw.Draw(output_img)

            for cx, cy in valid_centers:
                # 동일 규격 원 그리기 (lime 색상, 두께 3)
                x0, y0 = cx - target_radius, cy - target_radius
                x1, y1 = cx + target_radius, cy + target_radius
                draw.ellipse([x0, y0, x1, y1], outline="lime", width=3)

                # 중심점 표시 (빨간색 점)
                r_center = 3
                draw.ellipse(
                    [
                        cx - r_center,
                        cy - r_center,
                        cx + r_center,
                        cy + r_center,
                    ],
                    fill="red",
                )

            # 결과 출력
            st.image(
                output_img,
                caption="자동 정규화 및 카운팅 결과",
                use_container_width=True,
            )
            st.success(
                f"🎉 번들 내 감지된 총 CD-BAR 개수: **{len(valid_centers)}개**"
            )

        else:
            st.warning(
                "감지된 단면이 없습니다. 선명하고 초점이 맞은 사진을 업로드해 보세요."
            )
