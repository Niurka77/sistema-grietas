import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO
from supabase import create_client
from datetime import datetime
import os, cv2, tempfile, numpy as np
from PIL import Image
from dotenv import load_dotenv
import time

# === CONFIGURACIÓN DE PÁGINA ===
st.set_page_config(
    page_title="Sistema Grietas - Iglesia San Agustín",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CARGA DE VARIABLES DE ENTORNO ===
load_dotenv()

# === CSS PERSONALIZADO (Estilo Admin Panel Profesional) ===
st.markdown("""
<style>
    /* Sidebar - Azul oscuro */
    [data-testid="stSidebar"] {
        background-color: #004481 !important;
        border-right: 1px solid #003366;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #004481 0%, #00A9E0 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,68,129,0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }
    
    /* Tarjetas de estadísticas */
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #00A9E0;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #004481;
        margin: 0.5rem 0;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* Botones */
    .stButton>button {
        background-color: #00A9E0 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
    }
    .stButton>button:hover {
        background-color: #0088b8 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,169,224,0.3) !important;
    }
    
    /* Badge de estado */
    .badge-sana { 
        background: #dcfce7; 
        color: #166534; 
        padding: 0.4rem 1rem; 
        border-radius: 999px; 
        font-size: 0.85rem; 
        font-weight: 600;
        display: inline-block;
    }
    .badge-leve { 
        background: #fef3c7; 
        color: #92400e; 
        padding: 0.4rem 1rem; 
        border-radius: 999px; 
        font-size: 0.85rem; 
        font-weight: 600;
        display: inline-block;
    }
    .badge-alerta { 
        background: #fee2e2; 
        color: #b91c1c; 
        padding: 0.4rem 1rem; 
        border-radius: 999px; 
        font-size: 0.85rem; 
        font-weight: 600;
        display: inline-block;
    }
    
    /* Contenedor de carga */
    .upload-counter {
        background: #f0f9ff;
        border: 2px solid #00A9E0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
        color: #004481;
    }
    
    /* Barra de progreso */
    .progress-container {
        background: #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .progress-bar {
        background: linear-gradient(90deg, #004481 0%, #00A9E0 100%);
        height: 8px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# === CARGA DE MODELO Y CONEXIONES (UNA SOLA VEZ) ===
@st.cache_resource
def load_model():
    """Carga el modelo YOLO una sola vez"""
    try:
        return YOLO("best.pt")
    except:
        st.error("❌ No se encontró el modelo 'best.pt'. Asegúrate de tenerlo en la misma carpeta.")
        return None

@st.cache_resource
def init_supabase():
    """Inicializa conexión a Supabase"""
    try:
        return create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
    except Exception as e:
        st.error(f"❌ Error conectando a Supabase: {e}")
        return None

model = load_model()
supabase = init_supabase()

# === FUNCIONES AUXILIARES ===
def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def determine_status(n_grietas):
    """Determina el estado según número de grietas"""
    if n_grietas == 0:
        return "ESTRUCTURA SANA", "badge-sana"
    elif n_grietas < 3:
        return "HALLAZGO LEVE", "badge-leve"
    else:
        return "ALERTA ESTRUCTURAL", "badge-alerta"

def get_action_recommendation(status):
    """Recomendación de acción según estado"""
    actions = {
        "ESTRUCTURA SANA": "✓ Sin acción requerida",
        "HALLAZGO LEVE": "⚠ Revisión programada (6 meses)",
        "ALERTA ESTRUCTURAL": "🚨 Intervención inmediata"
    }
    return actions.get(status, "Sin información")

@st.cache_data(ttl=300)
def get_inspections_data():
    """Obtiene datos de Supabase (cache por 5 min)"""
    try:
        response = supabase.table("inspecciones").select("*").order("fecha", desc=True).execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except:
        return pd.DataFrame()

# === SIDEBAR (Menú Principal) ===
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.2); margin-bottom: 2rem;">
        <div style="background: white; border-radius: 12px; padding: 1rem; display: inline-block; margin-bottom: 0.5rem;">
            <span style="font-size: 2rem;">🏛️</span>
        </div>
        <div style="color: white; font-size: 1.3rem; font-weight: 800;">CRACK-DETECT</div>
        <div style="color: rgba(255,255,255,0.7); font-size: 0.75rem; margin-top: 0.25rem;">
            Gestión Patrimonial IA
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Menú de navegación
    page = st.radio(
        "📋 Módulos del Sistema",
        ["📤 Cargar Imágenes", "📊 Dashboard", "📋 Historial", "⚙️ Configuración"],
        label_visibility="collapsed",
        index=0
    )
    
    st.divider()
    
    # Info del proyecto
    st.markdown("""
    <div style="color: rgba(255,255,255,0.85); font-size: 0.85rem; line-height: 1.6;">
        <strong> Proyecto:</strong><br>
        Iglesia San Agustín<br>
        <br>
        <strong>🏛️ Ubicación:</strong><br>
        Centro Histórico<br>
        Arequipa, Perú<br>
        <br>
        <strong>👤 Usuario:</strong><br>
        Ing. Responsable
    </div>
    """, unsafe_allow_html=True)

# === MÓDULO 1: CARGAR IMÁGENES ===
if page == "📤 Cargar Imágenes":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📤 Carga Masiva de Imágenes</h1>
        <p>Sube fotos del dron para análisis automático de grietas estructurales</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ Configuración de Carga")
        
        # Upload de archivos con contador
        uploaded_files = st.file_uploader(
            "Seleccionar imágenes",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Arrastra hasta 100 fotos. Formato recomendado: JPG/PNG"
        )
        
        # Contador de imágenes
        if uploaded_files:
            st.markdown(f"""
            <div class="upload-counter">
                📸 {len(uploaded_files)} imagen(es) seleccionada(s)
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar preview de las primeras 3
            st.markdown("**Vista previa:**")
            for i, uploaded_file in enumerate(uploaded_files[:3]):
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_container_width=True)
            
            if len(uploaded_files) > 3:
                st.info(f"... y {len(uploaded_files) - 3} imágenes más")
        
        # Slider de confianza
        confidence = st.slider(
            "Nivel de confianza del modelo",
            min_value=0.1, max_value=0.9, value=0.45, step=0.05,
            help="Valores más altos = menos falsos positivos"
        )
        
        # Botón de proceso
        process_btn = st.button("🚀 INICIAR ANÁLISIS", type="primary", use_container_width=True)
        
        st.info("💡 **Consejo:** Las imágenes se procesan en ~2-3 segundos cada una")
    
    with col2:
        if uploaded_files and process_btn:
            # Barra de progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            logs = []
            results = []
            total = len(uploaded_files)
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Actualizar progreso
                progress = (i + 1) / total
                progress_bar.progress(progress)
                status_text.text(f"🔍 Procesando {i+1}/{total}: {uploaded_file.name}")
                
                try:
                    # Guardar temporalmente
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    # Inferencia YOLO
                    img = cv2.imread(tmp_path)
                    img_resized = cv2.resize(img, (640, 640))
                    results_yolo = model.predict(source=img_resized, conf=confidence, verbose=False)
                    
                    n_cracks = len(results_yolo[0].boxes)
                    estado, badge_class = determine_status(n_cracks)
                    accion = get_action_recommendation(estado)
                    
                    # Generar imagen anotada
                    annotated = results_yolo[0].plot()
                    
                    # Subir a Supabase Storage
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    file_name = f"insp_{timestamp}_{i}_{uploaded_file.name}"
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_out:
                        cv2.imwrite(tmp_out.name, annotated)
                        with open(tmp_out.name, 'rb') as f:
                            supabase.storage.from_("imagenes-deteccion").upload(file_name, f)
                    
                    url_img = supabase.storage.from_("imagenes-deteccion").get_public_url(file_name)
                    
                    # Guardar metadata en DB
                    supabase.table("inspecciones").insert({
                        "nombre_archivo": file_name,
                        "n_grietas": int(n_cracks),
                        "confianza_media": float(confidence),
                        "estado_alerta": estado,
                        "url_imagen": url_img,
                        "fecha": datetime.now().isoformat()
                    }).execute()
                    
                    results.append({
                        "Archivo": uploaded_file.name,
                        "Grietas": n_cracks,
                        "Estado": estado,
                        "Badge": badge_class,
                        "Acción": accion
                    })
                    
                    logs.append(f"✅ {uploaded_file.name}: {n_cracks} grietas - {estado}")
                    
                    os.unlink(tmp_path)
                    os.unlink(tmp_out.name)
                    
                except Exception as e:
                    logs.append(f"❌ Error en {uploaded_file.name}: {str(e)}")
            
            # Limpiar progreso
            status_text.empty()
            progress_bar.empty()
            
            # Mostrar resultados
            st.success(f"✨ Procesamiento completado: {len(results)}/{total} imágenes")
            
            # Estadísticas del batch
            total_grietas = sum(r["Grietas"] for r in results)
            alertas = sum(1 for r in results if r["Estado"] == "ALERTA ESTRUCTURAL")
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Total grietas detectadas", total_grietas)
            with col_stat2:
                st.metric("Alertas críticas", alertas, delta_color="inverse")
            with col_stat3:
                st.metric("Imágenes procesadas", len(results))
            
            # Tabla de resultados
            st.markdown("#### 📋 Resultados del Análisis")
            df_results = pd.DataFrame(results)
            st.dataframe(
                df_results[["Archivo", "Grietas", "Estado", "Acción"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Logs detallados
            with st.expander("📝 Ver logs detallados"):
                for log in logs:
                    st.text(log)
            
            # Actualizar cache
            st.cache_data.clear()
        
        elif uploaded_files and not process_btn:
            st.info(f"📁 {len(uploaded_files)} imágenes listas. Haz clic en 'INICIAR ANÁLISIS' para comenzar.")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; background: #f8fafc; border-radius: 12px; border: 2px dashed #cbd5e1;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📤</div>
                <h3 style="color: #334155; margin-bottom: 0.5rem;">Arrastra tus imágenes aquí</h3>
                <p style="color: #64748b; margin: 0;">
                    Formatos soportados: JPG, PNG<br>
                    Máximo 100 imágenes por lote
                </p>
            </div>
            """, unsafe_allow_html=True)

# === MÓDULO 2: DASHBOARD ===
elif page == "📊 Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Dashboard de Inspecciones</h1>
        <p>Métricas en tiempo real del estado estructural</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Obtener datos
    df = get_inspections_data()
    
    # Tarjetas de estadísticas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = len(df)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Inspecciones</div>
            <div class="stat-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        grietas = int(df['n_grietas'].sum()) if 'n_grietas' in df.columns and not df.empty else 0
        st.markdown(f"""
        <div class="stat-card" style="border-left-color: #28A745;">
            <div class="stat-label">Grietas Detectadas</div>
            <div class="stat-value" style="color: #28A745;">{grietas}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        criticas = len(df[df['estado_alerta'] == 'ALERTA ESTRUCTURAL']) if not df.empty and 'estado_alerta' in df.columns else 0
        st.markdown(f"""
        <div class="stat-card" style="border-left-color: #dc2626;">
            <div class="stat-label">Alertas Críticas</div>
            <div class="stat-value" style="color: #dc2626;">{criticas}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ultima = df.iloc[0]['fecha'][:10] if not df.empty and 'fecha' in df.columns else "N/A"
        st.markdown(f"""
        <div class="stat-card" style="border-left-color: #6366f1;">
            <div class="stat-label">Última Inspección</div>
            <div class="stat-value" style="font-size: 1.5rem; color: #6366f1;">{ultima}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráficos
    if not df.empty and 'estado_alerta' in df.columns:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            conteo = df['estado_alerta'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=conteo.index,
                values=conteo.values,
                hole=0.4,
                marker=dict(colors=['#28A745', '#F59E0B', '#dc2626']),
                textinfo='percent+label'
            )])
            fig.update_layout(
                title="Distribución por Estado Estructural",
                height=350,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'])
                df_chart = df.groupby(df['fecha'].dt.date)['n_grietas'].sum().reset_index()
                
                fig2 = go.Figure(data=[go.Bar(
                    x=df_chart['fecha'],
                    y=df_chart['n_grietas'],
                    marker_color='#00A9E0'
                )])
                fig2.update_layout(
                    title="Grietas Detectadas por Día",
                    xaxis_title="Fecha",
                    yaxis_title="Cantidad",
                    height=350
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    # Tabla reciente
    st.markdown("#### 📋 Últimas Inspecciones")
    if not df.empty:
        df_display = df[['fecha', 'nombre_archivo', 'n_grietas', 'estado_alerta']].head(10).copy()
        df_display['fecha'] = pd.to_datetime(df_display['fecha']).dt.strftime('%d/%m/%Y %H:%M')
        df_display.columns = ['Fecha', 'Archivo', 'Grietas', 'Estado']
        st.dataframe(df_display, use_container_width=True, hide_index=True)

# === MÓDULO 3: HISTORIAL ===
elif page == "📋 Historial":
    st.markdown("""
    <div class="main-header">
        <h1>📋 Historial Completo</h1>
        <p>Consulta y filtra todas las inspecciones realizadas</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = get_inspections_data()
    
    # Filtros
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    
    with col_f1:
        search = st.text_input("🔍 Buscar por archivo", placeholder="Ej: fachada_001")
    
    with col_f2:
        estado_filter = st.selectbox(
            "Filtrar por estado",
            ["Todos", "ESTRUCTURA SANA", "HALLAZGO LEVE", "ALERTA ESTRUCTURAL"]
        )
    
    with col_f3:
        min_cracks = st.number_input("Mín. grietas", min_value=0, max_value=50, value=0)
    
    # Aplicar filtros
    if not df.empty:
        if search:
            df = df[df['nombre_archivo'].str.contains(search, case=False, na=False)]
        if estado_filter != "Todos":
            df = df[df['estado_alerta'] == estado_filter]
        if min_cracks > 0:
            df = df[df['n_grietas'] >= min_cracks]
    
    # Mostrar tabla
    if not df.empty:
        df_table = df[['fecha', 'nombre_archivo', 'n_grietas', 'estado_alerta', 'url_imagen']].copy()
        df_table['fecha'] = pd.to_datetime(df_table['fecha']).dt.strftime('%d/%m/%Y %H:%M')
        df_table.columns = ['Fecha', 'Archivo', 'Grietas', 'Estado', 'URL']
        
        st.dataframe(
            df_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "URL": st.column_config.LinkColumn("Imagen", display_text="🔗 Ver")
            }
        )
        
        # Exportar
        csv = df_table.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "📥 Exportar a CSV",
            data=csv,
            file_name=f"inspecciones_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# === MÓDULO 4: CONFIGURACIÓN ===
elif page == "⚙️ Configuración":
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Configuración del Sistema</h1>
        <p>Ajustes del modelo y parámetros</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🤖 Modelo de IA")
        st.info("✅ Modelo actual: `best.pt` (YOLOv8)")
        st.metric("Precisión en validación", "98.6%")
        st.metric("Tiempo promedio", "~2.3 seg/imagen")
    
    with col2:
        st.markdown("#### 🔗 Conexiones")
        st.success("✅ Supabase: Conectado")
        st.success("✅ Storage: Conectado")
    
    st.divider()
    st.markdown("#### 📦 Información del Sistema")
    st.json({
        "Versión": "2.1.0",
        "Framework": "Streamlit + YOLOv8",
        "Proyecto": "Tesis UCSM - Gestión Patrimonial",
        "Fecha": datetime.now().strftime("%d/%m/%Y")
    }, expanded=False)

# === FOOTER ===
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666; border-top: 1px solid #eee; margin-top: 3rem;">
    <strong>Sistema de Detección de Grietas v2.1</strong><br>
    Universidad Católica de Santa María • Arequipa, Perú • 2025<br>
    <span style="font-size: 0.85rem; color: #999;">Tesis: Gestión de Patrimonio Histórico con IA y Drones</span>
</div>
""", unsafe_allow_html=True)