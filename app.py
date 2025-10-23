import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image
# Importar ImagOps no es estrictamente necesario si no se usa, pero se mantiene del script original.
# from PIL import ImageOps as ImagOps 
from keras.models import load_model

import platform

# --- CSS GÓTICO (Paleta Arcano-Escarlata) ---
gothic_css_variant = """
<style>
/* Paleta base: Fondo #111111, Texto #E0E0E0 (Pergamino ligero), Acento #5A4832 (Bronce/Metal), Sangre #A50000 */
.stApp {
    background-color: #111111;
    color: #E0E0E0;
    font-family: 'Georgia', serif;
}

/* Título Principal (h1) */
h1 {
    color: #A50000; /* Rojo sangre */
    text-shadow: 3px 3px 8px #000000;
    font-size: 3.2em; 
    border-bottom: 5px solid #5A4832; /* Borde Bronce */
    padding-bottom: 10px;
    margin-bottom: 30px;
    text-align: center;
    letter-spacing: 2px;
}

/* Subtítulos (h2, h3): Énfasis en el bronce */
h2, h3 {
    color: #C0C0C0; /* Plata/gris claro */
    border-left: 5px solid #5A4832;
    padding-left: 10px;
    margin-top: 25px;
}

/* Input y Camera (El Papiro de Inscripción) */
div[data-testid="stTextInput"], div[data-testid="stTextarea"], .stFileUploader, .stCameraInput {
    background-color: #1A1A1A;
    border: 1px solid #5A4832;
    padding: 10px;
    border-radius: 5px;
    color: #F5F5DC;
}

/* Dataframe (No hay en este script, pero por consistencia) */
div[data-testid="stDataFrame"] table {
    background-color: #1A1A1A;
    border: 1px solid #5A4832;
    color: #E0E0E0;
}
div[data-testid="stDataFrame"] thead tr th {
    background-color: #2A2A2A !important;
    color: #A50000 !important;
}

/* Texto de Alertas (Revelaciones) */
.stSuccess { background-color: #20251B; color: #F5F5DC; border-left: 5px solid #5A4832; }
.stInfo { background-color: #1A1A25; color: #F5F5DC; border-left: 5px solid #5A4832; }
.stWarning { background-color: #352A1A; color: #F5F5DC; border-left: 5px solid #A50000; }

/* Streamlit Sidebar Background */
.css-1d3w5rq {
    background-color: #202020;
}
</style>
"""
st.markdown(gothic_css_variant, unsafe_allow_html=True)


# Configuración de página Streamlit
st.set_page_config(
    page_title="El Lector de Signos",
    page_icon="🔮",
    layout="wide"
)

# --- Carga del Modelo Keras ---
@st.cache_resource
def load_keras_model(path='keras_model.h5'):
    """Carga el modelo de clasificación Keras entrenado (ej. con Teachable Machine)."""
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"❌ Falló la Invocación del Espíritu del Modelo Keras ('{path}').")
        st.caption(f"Detalle: {e}")
        st.info("Asegura que el archivo 'keras_model.h5' esté presente y sea válido.")
        return None

# Muestra la versión de Python como un dato histórico
st.write(f"Versión del Scriptorium (Python): **{platform.python_version()}**")

# Intentar cargar el modelo
model = load_keras_model()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("🔮 El Lector de Signos: Clasificación de Esencias")

# --- Contenido Principal ---
try:
    # Nota: Este archivo ('OIG5.jpg') debe estar presente en el entorno de ejecución.
    image = Image.open('Bloodborne-5.webp')
    st.image(image, caption="El Glifo de Referencia", use_container_width=True)
except FileNotFoundError:
    st.warning("El Glifo de Referencia ('Bloodborne-5.webp') no fue hallado. Se omitirá la previsualización.")
except Exception as e:
    st.warning(f"Error al cargar el Glifo de Referencia: {e}")

with st.sidebar:
    st.subheader("Manifiesto del Códice")
    st.markdown("""
    Este artefacto utiliza un **Modelo Imbuido (Keras/Teachable Machine)** para clasificar la Esencia capturada.
    La visión se procesa a $224 \times 224$ píxeles y se normaliza para la inferencia.
    """)

# Si el modelo cargó, procede con la interfaz
if model:
    img_file_buffer = st.camera_input("📸 Captura la Esencia (Foto Oracular)")

    if img_file_buffer is not None:
        # Inicializar el array de datos (si no está ya inicializado)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Cargar el buffer de imagen como PIL Image
        img = Image.open(img_file_buffer).convert('RGB') # Asegurar RGB

        # Redimensionar a 224x224
        newsize = (224, 224)
        img = img.resize(newsize)

        # Convertir PIL Image a numpy array
        img_array = np.array(img)

        # Normalizar la imagen
        # La normalización de Teachable Machine va de [-1, 1]
        normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1 # Corregido a / 127.5 para rango [-1, 1]
        
        # Cargar la imagen en el array de inferencia
        data[0] = normalized_image_array

        # Ejecutar la inferencia
        with st.spinner("Descifrando la Predicción del Oráculo..."):
            prediction = model.predict(data)
            # st.code(f"Output Raw: {prediction}", language='python') # Útil para depuración

        st.subheader("📜 Revelación del Oráculo")
        
        # Interpretar las predicciones (asumiendo que los índices 0, 1, (2) corresponden a 'Izquierda', 'Arriba', ('Derecha'))
        
        # Asumiendo que las etiquetas de TM son: [0] = Izquierda, [1] = Arriba, [2] = Derecha (Comentado)
        
        results_found = False
        
        if prediction[0][0] > 0.5:
            st.success(f"**Sello de la Izquierda:** Revelado con Certeza: **{prediction[0][0]:.4f}**")
            results_found = True
        
        if prediction[0][1] > 0.5:
            st.success(f"**Sello Superior:** Revelado con Certeza: **{prediction[0][1]:.4f}**")
            results_found = True
        
        #if prediction[0][2] > 0.5:
        #    st.success(f"**Sello de la Derecha:** Revelado con Certeza: **{prediction[0][2]:.4f}**")
        #    results_found = True

        if not results_found:
             # Encontrar la predicción máxima si no hay nada > 0.5
            max_index = np.argmax(prediction[0])
            max_prob = prediction[0][max_index]
            
            label_map = {0: "Izquierda", 1: "Arriba", 2: "Derecha (Comentado)"}
            
            # Usar 'Izquierda' o 'Arriba' si es uno de los dos, sino usar el índice raw
            label = label_map.get(max_index, f"Índice {max_index}")
            
            st.info(f"El Oráculo duda, pero apunta a **{label}** con una Certeza Máxima de **{max_prob:.4f}**")
            st.caption("Ajusta tu umbral de confianza si esperas una clasificación más clara.")

else:
    # El modelo no cargó, el mensaje de error ya se mostró en load_keras_model
    pass

# Información adicional y pie de página
st.markdown("---")
st.caption("""
**Crónicas del Lector de Signos**: Este Códice opera asumiendo la presencia del archivo 'keras_model.h5' y las etiquetas correspondientes a sus índices de salida.
""")


