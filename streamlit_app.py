import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color
import requests
from io import BytesIO
import os

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Comparador Paño Lency Piccolo",
    page_icon="🧶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilos CSS Personalizados (Piccolo Style) ---
st.markdown("""
<style>
    /* Fuente y colores generales */
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Fredoka', sans-serif;
        color: #4A4A4A;
    }

    /* Fondo general */
    .stApp {
        background-color: #FAFAFA; /* Un gris muy claro, casi blanco */
    }

    /* Títulos */
    h1, h2, h3 {
        color: #3b3b4f;
        font-weight: 700;
        text-align: left;
    }

    /* Botones */
    .stButton > button {
        background-color: #FFB7C5 !important; /* Rosado suave */
        color: #FFFFFF !important;
        border: none;
        border-radius: 20px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: 500;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #FF8BA7 !important; /* Un poco más oscuro al hover */
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Enlaces tipo botón */
    a.piccolo-btn {
        display: inline-block;
        background-color: #FFB7C5;
        color: white;
        padding: 8px 16px;
        text-align: center;
        text-decoration: none;
        border-radius: 15px;
        font-weight: 500;
        margin-top: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: background-color 0.3s;
    }
    a.piccolo-btn:hover {
        background-color: #FF8BA7;
        color: white;
    }

    /* Tarjetas/Contenedores */
    .css-1r6slb0 {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Imágenes redondeadas */
    img {
        border-radius: 12px;
    }

    /* Ajuste del logo */
    .logo-container {
        display: flex;
        justify_content: center;
        margin-bottom: 20px;
    }
    
    /* Spinner/Status */
    .stStatusWidget {
        background-color: #FFF0F5; /* Lavanda suave */
        border: 1px solid #FFB7C5;
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)

# --- Funciones de Lógica ---

@st.cache_data
def load_data(csv_path):
    """Carga y procesa la base de datos de paños."""
    try:
        df = pd.read_csv(csv_path)
        # Parsear Color_RGB de string "(R, G, B)" a tupla/array
        # Asumiendo formato "(97, 64, 58)"
        df['rgb_tuple'] = df['Color_RGB'].apply(lambda x: tuple(map(int, x.strip('()').split(','))))
        
        # Pre-calcular LAB para comparar más rápido luego
        # Normalizar RGB a 0-1 para skimage
        rgb_values = np.array(df['rgb_tuple'].tolist()) / 255.0
        # Convertir a LAB
        lab_values = color.rgb2lab(rgb_values)
        df['lab_values'] = list(lab_values)
        
        return df
    except Exception as e:
        st.error(f"Error cargando la base de datos: {e}")
        return pd.DataFrame()

def load_image(image_file):
    """Carga una imagen y la devuelve como objeto PIL."""
    img = Image.open(image_file)
    return img

def extract_colors(image, k=10):
    """Extrae los k colores principales usando K-Means."""
    # Redimensionar para velocidad
    img_small = image.resize((150, 150))
    img_array = np.array(img_small)
    
    # Si tiene canal alfa, quitarlo
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    pixels = img_array.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_
    # Convertir a enteros
    colors = colors.astype(int)
    return colors

def find_closest_fabrics(target_rgb, df, n=2):
    """Encuentra los n paños más cercanos en Delta E CIELAB."""
    # Convertir target RGB a LAB
    target_rgb_norm = np.array(target_rgb).reshape(1, 1, 3) / 255.0
    target_lab = color.rgb2lab(target_rgb_norm).reshape(3,)
    
    # Calcular distancias
    distances = []
    for index, row in df.iterrows():
        fabric_lab = row['lab_values']
        # Delta E manual simple (Euclidiana en LAB es Delta E 76, suficiente para esto)
        d_e = np.sqrt(np.sum((target_lab - fabric_lab)**2))
        distances.append(d_e)
    
    df_copy = df.copy()
    df_copy['delta_e'] = distances
    
    # Ordenar y tomar los mejores n
    best_matches = df_copy.sort_values('delta_e').head(n)
    return best_matches

# --- Sidebar ---
with st.sidebar:
    logo_path = "logo-piccolo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    
    st.markdown("**🧵 Tu asistente de costura**")
    st.write("")
    
    st.info("""
**¿Cómo funciona?**

1. Sube una foto de tu proyecto.
2. Detectamos los colores clave.
3. Te sugerimos las mejores telas de nuestro inventario.
""")
    
    st.write("")
    st.write("")
    st.markdown("<small>Desarrollado para Piccolo Ind. SAS</small>", unsafe_allow_html=True)

# --- Interfaz Principal ---
st.title("Descubre tus Paños Ideales")
st.markdown("Sube la foto de tu proyecto y deja que nuestra IA encuentre la combinación perfecta de nuestra colección.")

# 2. Carga de Base de Datos
csv_file = "base_datos_panos.csv"
if os.path.exists(csv_file):
    df_panos = load_data(csv_file)
else:
    st.error("No se encontró el archivo 'base_datos_panos.csv'. Por favor verifica la carpeta del proyecto.")
    st.stop()

# 3. Entrada de Usuario
uploaded_file = st.file_uploader("Sube la foto de tu proyecto (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("👆 Sube una imagen para comenzar.")
    
    st.write("")
    st.divider()
    with st.expander("🛠️ Ver código para insertar en WordPress"):
        st.code('''<iframe src="TU_URL_DE_STREAMLIT" width="100%" height="800px" frameborder="0"></iframe>''', language="html")

if uploaded_file is not None:
    # Contenedor principal de resultados
    results_container = st.container()
    
    # 4. Progreso Amigable
    with st.status("Procesando tu imagen... 🎨", expanded=True) as status:
        st.write("Leyendo tu proyecto...")
        original_image = load_image(uploaded_file)
        
        st.write("Analizando los 10 colores principales... 🔍")
        extracted_colors = extract_colors(original_image, k=10)
        
        st.write("Comparando con nuestro inventario de Paño Lency... 🧵")
        # Preparamos los resultados pero no los mostramos aún
        status.update(label="¡Análisis Completado!", state="complete", expanded=False)

    # 5. Mostrar Resultados
    st.divider()
    
    # Mostrar Imagen Original
    col_orig, col_dummy = st.columns([1, 2])
    with col_orig:
        st.image(original_image, caption="Tu Proyecto Original", use_container_width=True)
    
    st.subheader("Tus Colores y Nuestras Sugerencias")
    
    # Iterar sobre los colores extraídos
    for i, color_rgb in enumerate(extracted_colors):
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color_rgb)
        
        # Buscar coincidencias
        matches = find_closest_fabrics(color_rgb, df_panos, n=2)
        
        # Crear un contenedor visual para este color
        with st.container():
            st.markdown(f"### Color Detectado #{i+1}")
            
            # Layout: Muestra de Color Detectado | Sugerencia 1 | Sugerencia 2
            cols = st.columns([1, 2, 2])
            
            # --- Columna 1: Color Detectado ---
            with cols[0]:
                st.markdown(f"""
                <div style="
                    background-color: {hex_color};
                    width: 100%;
                    height: 100px;
                    border-radius: 12px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 10px;
                "></div>
                <p style="text-align: center; font-weight: bold;">{hex_color}</p>
                """, unsafe_allow_html=True)
            
            # --- Columnas 2 y 3: Sugerencias ---
            match_idx = 0
            for _, match in matches.iterrows():
                # Determinar en qué columna va (1 o 2, que son indices 1 y 2 de 'cols')
                with cols[match_idx + 1]:
                    # Intentar cargar la imagen del paño
                    # La ruta en CSV es absoluta.
                    # Primero intentamos la ruta absoluta.
                    # Si falla, buscamos en subcarpetas relativas o usamos placeholder.
                    
                    fabric_img_path = match['Ruta_Foto']
                    display_img = None
                    
                    if os.path.exists(fabric_img_path):
                        display_img = Image.open(fabric_img_path)
                    else:
                        # Intento alternativo: buscar solo por nombre de archivo en Pano-PN relativo
                        possible_path = os.path.join("Pano-PN", os.path.basename(fabric_img_path))
                        # Si estuviéramos en la carpeta superior, ajustamos
                        possible_path_2 = os.path.join("..", "Pano-PN", os.path.basename(fabric_img_path))
                        
                        if os.path.exists(possible_path):
                            display_img = Image.open(possible_path)
                        elif os.path.exists(possible_path_2):
                             display_img = Image.open(possible_path_2)
                    
                    # Mostrar Tarjeta
                    st.markdown(f"**{match['Nombre']}**")
                    if display_img:
                        st.image(display_img, use_container_width=True)
                    else:
                        st.markdown(f"<div style='height: 100px; background-color: {match['Color_Hex']}; border-radius:10px; display:flex; align-items:center; justify-content:center; color:white;'>Imagen no disponible</div>", unsafe_allow_html=True)
                    
                    # Botón de Acción
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <a href="https://piccolo.com.co/producto/pano-lency-plano/" target="_blank" class="piccolo-btn">
                            🛒 Ver detalles
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
                
                match_idx += 1
            
            st.divider()

# Fin de la aplicación
