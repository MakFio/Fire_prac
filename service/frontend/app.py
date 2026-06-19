import io, base64, tempfile, os
import streamlit as st
import requests
from PIL import Image

API_URL = "http://backend:8000"

icon = Image.open("service/frontend/favicon.png")

st.set_page_config(page_title="Fire Detection", layout="wide", page_icon=icon)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');
    html, body, .stMarkdown, .stText,
    .stButton > button > div, .stButton > button > div > p,
    .stTextInput > div > input,
    .stSlider, .stMetric, .stDataFrame, h1, h2, h3, h4, h5, h6,
    p, label, td, th {
        font-family: "Libre Baskerville", "Times New Roman", Times, serif !important;
    }
    span[data-testid="stIconMaterial"],
    [data-testid="collapsedControl"] span,
    [data-testid="collapsedControl"],
    .material-symbols-rounded {
        font-family: "Material Symbols Rounded" !important;
        font-feature-settings: "liga" !important;
        -webkit-font-feature-settings: "liga" !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Система детекции возгораний и задымления")

with st.sidebar:
    st.header("Настройки")
    api_url = st.text_input("API URL", value=API_URL)
    conf    = st.slider("Порог уверенности", 0.1, 0.9, 0.25, 0.05)
    st.divider()
    try:
        stats = requests.get(f"{api_url}/metrics", timeout=2).json()
        st.metric("Запросов",      stats["total_requests"])
        st.metric("Тревог (fire)", stats["total_alarms"])
        st.metric("Ср. latency",   f"{stats['avg_latency_ms']} мс")
    except Exception:
        st.warning("API недоступен")

tab1, tab2, tab3 = st.tabs(["Детекция изображений", "Детекция видео", "История"])

with tab1:
    uploaded = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Исходное")
            st.image(uploaded, use_container_width=True)

        if st.button("Запустить детекцию", type="primary"):
            with st.spinner("Обнаружение..."):
                resp = requests.post(
                    f"{api_url}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=30,
                )
            if resp.status_code == 200:
                data = resp.json()
                with col2:
                    st.subheader("Результат")
                    img_b = base64.b64decode(data["annotated_image_b64"])
                    st.image(Image.open(io.BytesIO(img_b)), use_container_width=True)

                if data["alarm"]:
                    st.error("Обнаружено возгорание!")
                else:
                    st.success("Возгораний не обнаружено")
                st.info(f"Инференс: {data['latency_ms']} мс")

                for d in data["detections"]:
                    icon_str = "🔥" if d["class"] == "fire" else "💨"
                    st.write(f"{icon_str} **{d['class']}** — {d['confidence']:.1%}")
            else:
                st.error(f"Ошибка API: {resp.status_code}")

with tab2:
    st.info("Поддерживается формат MP4. Видео обрабатывается покадрово; "
            "аннотированный результат доступен для скачивания.")
    uploaded_video = st.file_uploader("Загрузите видео", type=["mp4"])
    if uploaded_video:
        st.video(uploaded_video)

        if st.button("Запустить детекцию видео", type="primary"):
            with st.spinner("Обработка видео... Это может занять несколько минут."):
                resp = requests.post(
                    f"{api_url}/video_predict",
                    files={"file": (uploaded_video.name,
                                    uploaded_video.getvalue(),
                                    "video/mp4")},
                    timeout=600,
                )
            if resp.status_code == 200:
                data = resp.json()

                # Метрики
                col1, col2, col3 = st.columns(3)
                col1.metric("Кадров обработано", data["frame_count"])
                col2.metric("Детекций всего",    data["detections_total"])
                col3.metric("Скорость",          f"{data['fps_processed']} FPS")

                if data["alarm"]:
                    st.error("ТРЕВОГА: В видео обнаружено возгорание!")
                else:
                    st.success("Возгораний в видео не обнаружено")

                st.write(f"Общее время обработки: **{data['latency_ms']:.0f} мс**")

                if data["detections"]:
                    st.subheader("Сводка детекций")
                    for d in data["detections"]:
                        icon_str = "🔥" if d["class"] == "fire" else "💨"
                        st.write(
                            f"{icon_str} **{d['class']}** — "
                            f"обнаружено {d['count']} раз, "
                            f"макс. уверенность {d['max_confidence']:.1%}"
                        )

                # Скачивание аннотированного видео
                video_bytes = base64.b64decode(data["annotated_video_b64"])
                st.download_button(
                    label="⬇ Скачать аннотированное видео",
                    data=video_bytes,
                    file_name=f"annotated_{uploaded_video.name}",
                    mime="video/mp4",
                )
            else:
                st.error(f"Ошибка API: {resp.status_code} — {resp.text[:300]}")

with tab3:
    st.subheader("История запросов")
    if st.button("Обновить"):
        st.rerun()
    try:
        history = requests.get(f"{api_url}/alert_history?limit=100", timeout=3).json()
        if history:
            import pandas as pd
            df = pd.DataFrame([{
                "Время":      h["timestamp"][:19],
                "Файл":       h["filename"],
                "Детекций":   len(h["detections"]),
                "Тревога":    "Возгорание" if h["alarm"] else "Чисто",
                "Latency мс": h["latency_ms"],
            } for h in history])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("История пуста")
    except Exception:
        st.warning("Не удалось загрузить историю")
