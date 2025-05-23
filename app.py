import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

st.markdown(
    """
    <style>
    html, body, .main, .stApp, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
        background-color: #BC8648 !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: white !important;
    }
    html, body, [class*="css"], h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: black !important;
        text-align: center !important;
    }
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-16txtl3 {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception as e:
                st.warning(f"Intentando método alternativo de carga...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

st.title("🔍 DETECCIÓN DE OBJETOS EN IMÁGENES")
st.markdown("Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara. Ajusta los parámetros en la barra lateral para personalizar la detección.")

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if model:
    st.sidebar.title("PARÁMETROS")

    with st.sidebar:
        st.subheader('Configuración de detección')
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")

        st.subheader('Opciones avanzadas')
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("Algunas opciones avanzadas no están disponibles con esta configuración")

    main_container = st.container()

    with main_container:
        picture = st.camera_input("Capturar imagen", key="camera")

        if picture:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            with st.spinner("Detectando objetos..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()

            try:
                predictions = results.pred[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Objetos detectados")
                    label_names = model.names
                    category_count = {}
                    for category in categories:
                        category_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                        category_count[category_idx] = category_count.get(category_idx, 0) + 1

                    data = []
                    for category, count in category_count.items():
                        label = label_names[category]
                        confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                        data.append({
                            "Categoría": label,
                            "Cantidad": count,
                            "Confianza promedio": f"{confidence:.2f}"
                        })

                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index('Categoría')['Cantidad'])
                    else:
                        st.info("No se detectaron objetos con los parámetros actuales.")
                        st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")

                with col2:
                    st.subheader("Imagen con detecciones")
                    results.render()
                    st.image(cv2_img, channels='BGR', use_container_width=True)

            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
                st.stop()
else:
    st.error("No se pudo cargar el modelo. Por favor verifica las dependencias e inténtalo nuevamente.")
    st.stop()

st.markdown("---")
st.caption("**Acerca de la aplicación**: Esta aplicación utiliza YOLOv5 para detección de objetos en tiempo real. Desarrollada con Streamlit y PyTorch.")

try:
    st.image("wewewe.png", use_container_width=True)
except:
    st.warning("No se pudo cargar la imagen 'wewewe.png'.")
