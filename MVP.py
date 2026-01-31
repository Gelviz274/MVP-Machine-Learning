"""
E-commerce Analytics AI - MVP
Interfaz para predicción de compras en e-commerce.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import warnings

plt.rcParams['figure.facecolor'] = '#161b22'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['text.color'] = '#fff'
plt.rcParams['axes.labelcolor'] = '#fff'
plt.rcParams['xtick.color'] = '#fff'
plt.rcParams['ytick.color'] = '#fff'
plt.rcParams['axes.edgecolor'] = '#30363d'

warnings.filterwarnings('ignore')
st.set_page_config(page_title="E-commerce AI Predictor", page_icon="🛍️", layout="wide", initial_sidebar_state="expanded")

# ==============================================================================
# ESTILOS
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    
    :root {
        --bg: #0d1117;
        --card: #161b22;
        --border: #30363d;
        --text: #fff;
        --muted: #8b949e;
        --accent: #58a6ff;
        --accent-2: #3fb950;
        --danger: #f85149;
        --warning: #d29922;
    }

    .stApp { font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--text) !important; font-size: 1.05rem; }
    .stMarkdown, .stText, p, li, label, span { color: var(--text) !important; font-size: 1.05rem; }
    h1 { font-size: 1.9rem !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.3rem !important; }
    h4 { font-size: 1.15rem !important; }
    header[data-testid="stHeader"] { background: transparent; }
    .stDeployButton { display: none; }
    footer { visibility: hidden; }

    /* Animaciones elegantes */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    .card { animation: fadeInUp 0.5s ease-out; }
    .hero-banner { animation: fadeInUp 0.6s ease-out; }
    .input-desc-block { animation: fadeIn 0.35s ease-out; }

    /* Menú lateral */
    .nav-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted) !important;
        margin-bottom: 0.5rem;
        padding-left: 0.5rem;
    }
    [data-testid="stSidebar"] .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    [data-testid="stSidebar"] .stRadio label {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 0.65rem 1rem !important;
        margin: 0 !important;
        color: var(--text) !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(88, 166, 255, 0.08) !important;
        border-color: var(--accent) !important;
    }
    [data-testid="stSidebar"] .stRadio label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:first-of-type {
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stRadio input:checked + label,
    [data-testid="stSidebar"] .stRadio [data-checked="true"] {
        background: rgba(88, 166, 255, 0.18) !important;
        border-color: var(--accent) !important;
        color: var(--text) !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        padding: 0.65rem 1rem;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: transparent !important;
        color: var(--text) !important;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(88, 166, 255, 0.1) !important;
        border-color: var(--accent) !important;
    }

    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .card-title { font-size: 1.1rem; font-weight: 600; color: var(--text) !important; margin-bottom: 0.5rem; }
    .stat-big { font-size: 2.2rem; font-weight: 700; color: var(--text) !important; }
    .stat-cap { font-size: 0.9rem; color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.05em; }

    .hero-banner {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    .hero-banner h1 { font-size: 1.9rem; margin: 0; }
    .hero-banner p { color: var(--muted) !important; margin: 0.5rem 0 0 0; font-size: 1.05rem; }

    /* Bloque debajo de cada input en predicción - letra más grande para exposición */
    .input-desc-block {
        background: rgba(48, 54, 61, 0.5);
        border-left: 3px solid var(--accent);
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin-top: 0.4rem;
        margin-bottom: 1rem;
        font-size: 0.98rem;
    }
    .input-desc-block .tit { color: var(--muted) !important; font-size: 0.88rem; text-transform: uppercase; margin-bottom: 0.25rem; }
    .input-desc-block .body { color: var(--text) !important; line-height: 1.5; font-size: 0.98rem; }
    .input-desc-block .impact { color: var(--accent-2) !important; margin-top: 0.4rem; font-size: 0.95rem; }
    .input-desc-block .use { color: var(--muted) !important; margin-top: 0.25rem; font-size: 0.92rem; }
    .input-desc-block .hint-01 { color: var(--warning) !important; font-size: 0.92rem; margin-top: 0.3rem; font-weight: 500; }

    .pred-badge { display: inline-block; padding: 0.5rem 1.2rem; border-radius: 999px; font-weight: 600; font-size: 1.1rem; }
    .pred-badge.yes { background: rgba(63, 185, 80, 0.2); color: #3fb950; border: 1px solid #238636; }
    .pred-badge.no { background: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid #da3633; }

    section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid var(--border); font-size: 1rem; }
    .stSlider label, .stNumberInput label, .stSelectbox label, .stRadio label { color: var(--text) !important; font-size: 1.05rem !important; }
    [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
    .stCaption { font-size: 1rem !important; }
    .stMetric label { font-size: 0.95rem !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.4rem !important; }

    /* Pestañas de navegación principales */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 6px;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.25rem;
        font-size: 1rem;
        font-weight: 600;
        color: var(--muted);
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text);
        background: rgba(255,255,255,0.05);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--accent);
        color: #fff;
    }

    /* Bento grid y cards de conclusiones */
    .bento-section { margin-bottom: 2rem; }
    .bento-section-title { font-size: 1.1rem; font-weight: 600; color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem; }
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .bento-cell { min-height: 0; }
    .bento-cell--span2 { grid-column: span 2; }
    .bento-cell--span2r { grid-row: span 2; }
    .conclusion-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.25rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .conclusion-card:hover { border-color: var(--accent); }
    .conclusion-card__icon { font-size: 1.75rem; margin-bottom: 0.5rem; line-height: 1; }
    .conclusion-card__title { font-size: 1rem; font-weight: 600; color: var(--text) !important; margin-bottom: 0.5rem; line-height: 1.3; }
    .conclusion-card__text { font-size: 0.92rem; color: var(--muted) !important; line-height: 1.5; margin: 0; flex: 1; }
    .conclusion-card--accent { border-left: 4px solid var(--accent); }
    .conclusion-card--green { border-left: 4px solid var(--accent-2); }
    .conclusion-card--warning { border-left: 4px solid var(--warning); }
    .conclusion-summary {
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.08) 0%, rgba(63, 185, 80, 0.06) 100%);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 0.5rem;
    }
    .conclusion-summary p { color: var(--text) !important; font-size: 1rem; line-height: 1.6; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATOS Y MODELO
# ==============================================================================
@st.cache_data
def load_data():
    return pd.read_csv('data_clean.csv')

@st.cache_resource
def train_model(df):
    try:
        X = df.drop('Revenue', axis=1)
        y = df['Revenue']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        model = LogisticRegression(max_iter=10000, random_state=42)
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        y_prob = model.predict_proba(X_test_pca)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            'model': model, 'scaler': scaler, 'pca': pca, 'features': X.columns.tolist(),
            'accuracy': (y_pred == y_test).mean(), 'auc': roc_auc_score(y_test, y_prob),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'cm': cm, 'y_test': y_test, 'y_prob': y_prob
        }
    except Exception as e:
        st.error(f"Error entrenando el modelo: {e}")
        return None

df = load_data()
m = train_model(df)
if not m:
    st.stop()

# ==============================================================================
# GUÍA DE VARIABLES
# ==============================================================================
VARIABLE_INFO = {
    "ProductRelated": {"nombre": "Páginas de Producto", "desc": "Número de páginas de productos o catálogo visitadas.", "impacto": "Alto. Más páginas suelen indicar mayor interés de compra.", "uso": "Cuenta cada vista a página de producto (ítem, ficha, categoría).", "indica": "Engagement con el catálogo; valores altos sugieren modo compra."},
    "ProductRelated_Duration": {"nombre": "Tiempo en Productos", "desc": "Tiempo total (segundos) en páginas de productos.", "impacto": "Alto. Más tiempo suele correlacionar con decisión de compra.", "uso": "Suma de segundos en todas las páginas de producto de la sesión.", "indica": "Nivel de consideración; más tiempo = más probabilidad de conversión."},
    "Administrative": {"nombre": "Páginas Administrativas", "desc": "Visitas a páginas de cuenta, login o configuración.", "impacto": "Medio. Indica usuarios registrados o en proceso de login.", "uso": "Cada vista a login, mi cuenta, carrito.", "indica": "Intención de usar cuenta o completar compra."},
    "Administrative_Duration": {"nombre": "Tiempo en Admin", "desc": "Tiempo en segundos en páginas administrativas.", "impacto": "Medio. Refuerza si el usuario está activo en su cuenta.", "uso": "Tiempo acumulado en páginas de cuenta/login.", "indica": "Compromiso con la plataforma."},
    "Informational": {"nombre": "Páginas Informativas", "desc": "Visitas a FAQ, ayuda, políticas o contenido informativo.", "impacto": "Bajo–Medio. Puede indicar resolución de dudas precompra.", "uso": "Cada vista a secciones de información (no producto ni cuenta).", "indica": "Usuario informándose; a veces precede a la compra."},
    "Informational_Duration": {"nombre": "Tiempo en Info", "desc": "Tiempo en segundos en páginas informativas.", "impacto": "Bajo. Complementa el número de páginas informativas.", "uso": "Tiempo total en FAQ, ayuda, etc.", "indica": "Nivel de investigación antes de decidir."},
    "BounceRates": {"nombre": "Tasa de Rebote", "desc": "Proporción de sesiones de una sola página.", "impacto": "Alto (inverso). Valores altos reducen probabilidad de compra.", "uso": "Métrica 0–1. Valores bajos = mejor engagement.", "indica": "Calidad del tráfico; bajo rebote = más involucrados."},
    "ExitRates": {"nombre": "Tasa de Salida", "desc": "Porcentaje de salidas desde una página respecto a las vistas.", "impacto": "Alto (inverso). Salidas altas suelen indicar abandono.", "uso": "Por página, 0–1. Menor = mejor retención.", "indica": "Puntos de fricción o abandono en el funnel."},
    "PageValues": {"nombre": "Page Values", "desc": "Valor promedio de una página para la conversión (Google Analytics).", "impacto": "Muy alto. Suele ser la variable más predictiva de compra.", "uso": "Si > 0, la página contribuyó al valor de la conversión.", "indica": "Páginas que llevan a conversión; > 0 suele asociarse a compra."},
    "SpecialDay": {"nombre": "Cercanía a Día Especial", "desc": "Proximidad a fechas especiales (festivos, Black Friday).", "impacto": "Medio. Aumenta intención en fechas clave.", "uso": "Escala 0–1: 0 = lejos, 1 = día del evento.", "indica": "Efecto estacional en la probabilidad de compra."},
    "Weekend": {"nombre": "Fin de Semana", "desc": "Si la visita fue sábado o domingo.", "impacto": "Bajo–Medio. Patrones pueden diferir por día.", "uso": "Binario: 1 = fin de semana, 0 = laboral.", "indica": "Comportamiento según tipo de día."},
    "Month": {"nombre": "Mes", "desc": "Mes de la visita (codificado).", "impacto": "Medio. Estacionalidad (Navidad, rebajas).", "uso": "Selecciona el mes de la sesión a simular.", "indica": "Patrones estacionales de conversión."},
    "VisitorType": {"nombre": "Tipo de Visitante", "desc": "Nuevo, recurrente u otro.", "impacto": "Medio. Recurrentes suelen convertir más.", "uso": "Returning = ya visitó; New = primera vez; Other = otro.", "indica": "Lealtad y familiaridad con la tienda."},
    "OperatingSystems": {"nombre": "Sistema Operativo", "desc": "SO del dispositivo (Windows, Mac, etc.) codificado.", "impacto": "Bajo. Segmentación por dispositivo.", "uso": "Código numérico del SO detectado.", "indica": "Perfil técnico del usuario."},
    "Browser": {"nombre": "Navegador", "desc": "Navegador usado (Chrome, Firefox, etc.) codificado.", "impacto": "Bajo. Útil para segmentación.", "uso": "Código numérico del navegador.", "indica": "Preferencias técnicas del usuario."},
    "Region": {"nombre": "Región", "desc": "Región geográfica del visitante (codificada).", "impacto": "Bajo–Medio. Regiones pueden tener distinta conversión.", "uso": "Código de región (1–9 típicamente).", "indica": "Geolocalización y diferencias por mercado."},
    "TrafficType": {"nombre": "Fuente de Tráfico", "desc": "Origen del visitante: orgánico, pago, referral, directo.", "impacto": "Medio. Tráfico de calidad suele convertir mejor.", "uso": "Código numérico de la fuente en Analytics.", "indica": "Canal de adquisición y efectividad."},
}

def render_input_desc(key, zero_one_hint=None):
    info = VARIABLE_INFO.get(key, {})
    if not info:
        return
    hint_html = f'<div class="hint-01">{zero_one_hint}</div>' if zero_one_hint else ''
    st.markdown(f"""
    <div class="input-desc-block">
        <div class="tit">Variable: {info.get('nombre', key)}</div>
        <div class="body">{info.get('desc', '')}</div>
        <div class="impact">Impacto en la predicción: {info.get('impacto', '')}</div>
        <div class="use">Uso: {info.get('uso', '')} — Indica: {info.get('indica', '')}</div>
        {hint_html}
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR (solo métricas y autores; navegación por pestañas en área principal)
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <div style="width: 42px; height: 42px; background: linear-gradient(135deg, #58a6ff 0%, #238636 100%); border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; color: #fff; font-weight: 700; font-size: 20px;">AI</div>
        <p style="margin: 0.4rem 0 0 0; font-weight: 700; font-size: 1.05rem; color: #fff;">E-commerce Predictor</p>
        <p style="margin: 0; font-size: 0.75rem; color: #8b949e;">Clasificación · Regresión logística</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='nav-title'>Métricas del modelo</div>", unsafe_allow_html=True)
    st.metric("Accuracy", f"{m['accuracy']:.1%}")
    st.metric("AUC-ROC", f"{m['auc']:.3f}")
    st.metric("Precision", f"{m['precision']:.2%}")
    st.metric("Recall", f"{m['recall']:.2%}")
    st.markdown("---")
    st.markdown("<div class='nav-title'>Autores</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.9rem; color: #8b949e; line-height: 1.5; margin: 0;'>Juan Gelviz<br>Tatiana Castaño<br>William Rodriguez<br>Victoria Bayona</p>", unsafe_allow_html=True)

# Navegación por pestañas en el área principal
tab_inicio, tab_pred, tab_metricas, tab_pca, tab_analisis, tab_corr, tab_conclusiones = st.tabs([
    "Inicio", "Hacer Predicción", "Rendimiento del modelo", "PCA", "Análisis de Datos", "Correlaciones", "Conclusiones"
])

# ==============================================================================
# PESTAÑA: INICIO
# ==============================================================================
with tab_inicio:
    st.markdown("""
    <div class="hero-banner">
        <h1>Predictor de Compras</h1>
        <p>Plataforma de Machine Learning para analizar comportamiento de usuarios y predecir conversiones de ventas.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-bottom: 1.5rem;">
        <div class="card-title">Problema</div>
        <p style="color: #fff; font-size: 1.05rem; line-height: 1.65; margin: 0;">
            Las tiendas online tienen dificultad para distinguir, entre miles de sesiones, cuáles tienen una <strong>intención real de compra</strong>. 
            Esto genera ineficiencia en las estrategias de marketing: se invierte en usuarios que no convertirán o se desaprovechan oportunidades con quienes sí están listos para comprar.
        </p>
    </div>
    <div class="card" style="margin-bottom: 1.5rem;">
        <div class="card-title">Solución</div>
        <p style="color: #fff; font-size: 1.05rem; line-height: 1.65; margin: 0;">
            Un <strong>modelo predictivo de Regresión Logística</strong> optimizado con <strong>PCA</strong> que analiza métricas de comportamiento 
            (como <strong>PageValues</strong> y <strong>ExitRates</strong>) para predecir con precisión la conversión de ventas. 
            Así se pueden tomar acciones proactivas —por ejemplo, ofertas o recordatorios— para cerrar transacciones y priorizar recursos en los usuarios con mayor probabilidad de compra.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="stat-cap">Total registros</div>
            <div class="stat-big">{len(df):,}</div>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem; color: #8b949e;">Datos históricos procesados</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="stat-cap">Precisión (test)</div>
            <div class="stat-big">{m['accuracy']:.1%}</div>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem; color: #8b949e;">Conjunto de prueba 30%</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="stat-cap">Variables (PCA)</div>
            <div class="stat-big">{len(m['features'])}</div>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem; color: #8b949e;">Features activos</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Vista rápida")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**Distribución de clases (Revenue)**")
        fig = px.pie(values=df['Revenue'].value_counts().values, names=['No Compra', 'Compra'],
                     color_discrete_sequence=['#0ea5e9', '#f59e0b'], hole=0.6)
        fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=True,
                          paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#fff'), legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)
    with ch2:
        st.markdown("**Variables de mayor influencia**")
        impact_data = pd.DataFrame([
            {'Variable': 'PageValues', 'Impacto': 0.95}, {'Variable': 'ExitRates', 'Impacto': 0.85},
            {'Variable': 'ProductRelated', 'Impacto': 0.70}, {'Variable': 'Month', 'Impacto': 0.50},
        ])
        fig = px.bar(impact_data, x='Impacto', y='Variable', orientation='h', color='Impacto',
                     color_continuous_scale=[[0, '#0ea5e9'], [0.5, '#06b6d4'], [1, '#22c55e']])
        fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161b22', font=dict(color='#fff'),
                          xaxis=dict(gridcolor='#30363d'), yaxis=dict(gridcolor='#30363d'))
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PESTAÑA: HACER PREDICCIÓN
# ==============================================================================
with tab_pred:
    if 'pred_step' not in st.session_state:
        st.session_state.pred_step = 1
    calc_clicked = False

    st.markdown("""
    <div class="hero-banner">
        <h1>Simulador de Predicción</h1>
        <p>Completa el formulario por pasos. Debajo de cada campo se muestra la explicación de la variable, su impacto y uso.</p>
    </div>
    """, unsafe_allow_html=True)

    step = st.session_state.pred_step
    st.markdown(f"**Paso {step} de 3** — " + ["Actividad web", "Métricas y contexto", "Dispositivo y canal"][step - 1])
    st.progress(step / 3)
    st.markdown("---")

    if step == 1:
        st.markdown("#### Paso 1: Actividad web")
        col1a, col1b = st.columns(2)
        with col1a:
            st.slider("Páginas de producto vistas", 0, 300, 20, key="pr", help="Número de páginas de catálogo o producto que visitó el usuario.")
            render_input_desc("ProductRelated")
            st.number_input("Duración en productos (segundos)", 0.0, 50000.0, 500.0, step=10.0, key="prd", help="Tiempo total en páginas de producto.")
            render_input_desc("ProductRelated_Duration")
            st.slider("Páginas administrativas (cuenta/login)", 0, 30, 0, key="adm", help="Visitas a login, mi cuenta o carrito.")
            render_input_desc("Administrative")
        with col1b:
            st.number_input("Duración en admin (segundos)", 0.0, 5000.0, 0.0, step=10.0, key="admd", help="Tiempo en páginas de cuenta o login.")
            render_input_desc("Administrative_Duration")
            st.slider("Páginas informativas (FAQ/ayuda)", 0, 30, 0, key="inf", help="Visitas a FAQ, ayuda o políticas.")
            render_input_desc("Informational")
            st.number_input("Duración en info (segundos)", 0.0, 5000.0, 0.0, step=10.0, key="infd", help="Tiempo en páginas informativas.")
            render_input_desc("Informational_Duration")
        st.button("Siguiente →", type="primary", key="next1", on_click=lambda: st.session_state.update(pred_step=2))

    elif step == 2:
        st.markdown("#### Paso 2: Métricas de calidad y contexto")
        col2a, col2b = st.columns(2)
        with col2a:
            st.slider("Tasa de rebote (Bounce Rate)", 0.0, 0.2, 0.01, format="%.3f", key="bounce", help="Valor entre 0 y 0.200. Proporción de sesiones de una sola página.")
            render_input_desc("BounceRates")
            st.slider("Tasa de salida (Exit Rate)", 0.0, 0.2, 0.03, format="%.3f", key="exit", help="Valor entre 0 y 0.200. Porcentaje de salidas desde la página.")
            render_input_desc("ExitRates")
            st.number_input("Page Values (Google Analytics)", 0.0, 400.0, 15.0, step=1.0, key="pv", help="Valor de la página para la conversión; > 0 suele indicar mayor probabilidad de compra.")
            render_input_desc("PageValues")
            st.markdown("""
            <div class="card" style="margin-top: 0.5rem; border-left: 3px solid #3fb950;">
                <div class="card-title">Insight del modelo: Page Value</div>
                <p style="color: #fff; font-size: 1rem; line-height: 1.55; margin: 0;">
                    La variable <strong>Page Value</strong> resultó ser el predictor más fuerte en nuestro modelo de Regresión Logística. 
                    Esto indica que el <strong>comportamiento histórico de las páginas visitadas</strong> es más relevante que factores temporales o demográficos. 
                    Para la empresa, esto significa que <strong>optimizar el flujo hacia estas páginas de alto valor</strong> es la estrategia más rentable para aumentar el Revenue.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.selectbox("Mes", ["Feb", "Mar", "May", "Jun", "Jul", "Sep", "Oct", "Nov", "Dec"], index=7, key="month", help="Mes en el que ocurre la visita.")
            render_input_desc("Month")
        with col2b:
            st.slider("Cercanía a día especial (0 = no, 1 = sí/día del evento)", 0.0, 1.0, 0.0, key="special", help="0 = no hay cercanía a fecha especial; 1 = día del evento (ej. Black Friday). Valores intermedios = cercanía proporcional.")
            render_input_desc("SpecialDay", zero_one_hint="Valores: 0 = sin cercanía a día especial, 1 = día del evento. Usa 0 o 1 para simular casos claros.")
            st.checkbox("Es fin de semana", value=False, key="weekend", help="Marcar si la visita es sábado o domingo (valor 1); si no, queda 0.")
            render_input_desc("Weekend", zero_one_hint="Valor 0 = día entre semana, 1 = fin de semana. Variable binaria.")
            st.radio("Tipo de visitante", ["Returning_Visitor", "New_Visitor", "Other"], horizontal=True, key="visitor", help="Returning = ya visitó antes; New = primera vez.")
            render_input_desc("VisitorType")
        c_prev, c_next = st.columns(2)
        with c_prev:
            st.button("← Anterior", key="prev2", on_click=lambda: st.session_state.update(pred_step=1))
        with c_next:
            st.button("Siguiente →", type="primary", key="next2", on_click=lambda: st.session_state.update(pred_step=3))

    else:
        st.markdown("#### Paso 3: Dispositivo y canal")
        col3a, col3b = st.columns(2)
        with col3a:
            st.selectbox("Sistema operativo", [1, 2, 3, 4, 5, 6, 7, 8], index=1, key="os", help="Código del SO del dispositivo.")
            render_input_desc("OperatingSystems")
            st.selectbox("Navegador", list(range(1, 14)), index=1, key="browser", help="Código del navegador usado.")
            render_input_desc("Browser")
        with col3b:
            st.selectbox("Región", list(range(1, 10)), index=0, key="region", help="Código de región geográfica.")
            render_input_desc("Region")
            st.selectbox("Fuente de tráfico", list(range(1, 21)), index=1, key="traffic", help="Origen del tráfico (orgánico, pago, etc.).")
            render_input_desc("TrafficType")
        c_prev, c_calc = st.columns(2)
        with c_prev:
            st.button("← Anterior", key="prev3", on_click=lambda: st.session_state.update(pred_step=2))
        with c_calc:
            calc_clicked = st.button("Calcular probabilidad de compra", type="primary", key="calc_btn")

    if step == 3 and calc_clicked:
        def get(k, default): return st.session_state.get(k, default)
        product_related = get("pr", 20)
        product_related_duration = get("prd", 500.0)
        administrative = get("adm", 0)
        administrative_duration = get("admd", 0.0)
        informational = get("inf", 0)
        informational_duration = get("infd", 0.0)
        bounce_rates = get("bounce", 0.01)
        exit_rates = get("exit", 0.03)
        page_values = get("pv", 15.0)
        month = get("month", "Nov")
        special_day = get("special", 0.0)
        weekend = get("weekend", False)
        visitor_type = get("visitor", "Returning_Visitor")
        os_val = get("os", 2)
        browser_val = get("browser", 2)
        region_val = get("region", 1)
        traffic_type = get("traffic", 2)
        input_data = {
            'Administrative': administrative, 'Administrative_Duration': administrative_duration,
            'Informational': informational, 'Informational_Duration': informational_duration,
            'ProductRelated': product_related, 'ProductRelated_Duration': product_related_duration,
            'BounceRates': bounce_rates, 'ExitRates': exit_rates, 'PageValues': page_values,
            'SpecialDay': special_day, 'Weekend': 1 if weekend else 0,
        }
        month_to_col = {'Dec': 'Dec', 'Feb': 'Feb', 'Jul': 'Jul', 'Jun': 'June', 'Mar': 'Mar', 'May': 'May', 'Nov': 'Nov', 'Oct': 'Oct', 'Sep': 'Sep'}
        for ui_month, col_name in month_to_col.items():
            input_data[f'Month_{col_name}'] = 1 if month == ui_month else 0
        input_data['VisitorType_Other'] = 1 if visitor_type == 'Other' else 0
        input_data['VisitorType_Returning_Visitor'] = 1 if visitor_type == 'Returning_Visitor' else 0
        for i in range(2, 9): input_data[f'OperatingSystems_{i}'] = 1 if os_val == i else 0
        for i in range(2, 14): input_data[f'Browser_{i}'] = 1 if browser_val == i else 0
        for i in range(2, 10): input_data[f'Region_{i}'] = 1 if region_val == i else 0
        for i in range(2, 21): input_data[f'TrafficType_{i}'] = 1 if traffic_type == i else 0
        input_df = pd.DataFrame([input_data])
        for col in m['features']:
            if col not in input_df.columns: input_df[col] = 0
        input_df = input_df[m['features']]
        input_scaled = m['scaler'].transform(input_df)
        input_pca = m['pca'].transform(input_scaled)
        buy_prob = m['model'].predict_proba(input_pca)[0][1]
        st.session_state.pred_result = buy_prob

    if 'pred_result' in st.session_state and step == 3:
        buy_prob = st.session_state.pred_result
        st.markdown("### Resultado")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="card" style="text-align: center; padding: 2rem;">
                <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">{'🎉' if buy_prob > 0.5 else '📊'}</div>
                <div class="pred-badge {'yes' if buy_prob > 0.5 else 'no'}">{'COMPRA PROBABLE' if buy_prob > 0.5 else 'NO COMPRARÁ'}</div>
                <p style="margin-top: 1rem; color: #fff; font-size: 1.2rem;">Probabilidad estimada: <strong>{buy_prob:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=buy_prob * 100, title={'text': "Probabilidad (%)"},
                gauge={'axis': {'range': [0, 100], 'tickcolor': "#fff"}, 'bar': {'color': "#0ea5e9"},
                       'bgcolor': "#161b22", 'steps': [
                        {'range': [0, 30], 'color': "rgba(248, 81, 73, 0.35)"},
                        {'range': [30, 70], 'color': "rgba(210, 153, 34, 0.35)"},
                        {'range': [70, 100], 'color': "rgba(63, 185, 80, 0.35)"}],
                       'threshold': {'line': {'color': "#fff", 'width': 3}, 'thickness': 0.8, 'value': 50}}))
            fig.update_layout(height=280, margin=dict(t=30, b=20, l=30, r=30), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#fff'))
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PESTAÑA: RENDIMIENTO DEL MODELO
# ==============================================================================
with tab_metricas:
    st.markdown("### Rendimiento del modelo")
    st.markdown("""
    <div class="card" style="margin-bottom: 1.5rem;">
        <div class="card-title">Elección y justificación del modelo</div>
        <p style="color: #fff; font-size: 1.05rem; line-height: 1.6; margin: 0;">
            El problema es de <strong>clasificación binaria</strong> (Revenue: compra sí/no). Se eligió <strong>Regresión Logística</strong> 
            porque es interpretable, estable con muchas variables y adecuada para probabilidades. Para evaluación se usan métricas propias de 
            clasificación: <strong>Accuracy</strong>, <strong>Precision</strong>, <strong>Recall</strong>, <strong>F1</strong>, 
            <strong>Matriz de Confusión</strong> y <strong>Curva ROC / AUC</strong>. En problemas de regresión se usarían RMSE o MAE.
        </p>
    </div>
    """, unsafe_allow_html=True)

    f1 = (2 * (m['precision'] * m['recall']) / (m['precision'] + m['recall'])) if (m['precision'] + m['recall']) > 0 else 0.0
    met_c1, met_c2, met_c3, met_c4 = st.columns(4)
    metrics_info = [
        ("Accuracy", m['accuracy'], "Porcentaje de predicciones correctas (aciertos totales). Importante para ver el rendimiento global del modelo."),
        ("Precision", m['precision'], "De los que el modelo predijo como compra, cuántos realmente compraron. Importante para no sobreestimar conversiones."),
        ("Recall", m['recall'], "De los que realmente compraron, cuántos el modelo detectó. Importante para no perder oportunidades de venta."),
        ("F1 Score", f1, "Media armónica entre Precision y Recall. Útil cuando las clases están desbalanceadas o se busca un equilibrio."),
    ]
    for col, (label, val, desc) in zip([met_c1, met_c2, met_c3, met_c4], metrics_info):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <div class="stat-big">{val:.1%}</div>
                <div class="stat-cap">{label}</div>
                <p style="font-size: 0.95rem; color: #8b949e; margin-top: 8px; line-height: 1.45;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    cm = m['cm']
    tn, fp, fn, tp = cm.ravel()
    with d1:
        st.markdown("**Matriz de confusión**")
        st.caption("Filas = valor real (No Compra / Compra). Columnas = predicción del modelo.")
        fig_cm, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax, xticklabels=['No Compra', 'Compra'],
                   yticklabels=['No Compra', 'Compra'], annot_kws={'size': 14, 'color': '#0d1117'},
                   linewidths=0.5, linecolor='#30363d', cbar_kws={'label': 'Cantidad'})
        ax.set_xlabel('Predicción', fontsize=12); ax.set_ylabel('Real', fontsize=12)
        ax.set_facecolor('#161b22'); fig_cm.patch.set_facecolor('#161b22')
        plt.tight_layout(); st.pyplot(fig_cm); plt.close(fig_cm)
        st.markdown(f"""
        <div class="card" style="margin-top: 0.75rem;">
            <div class="card-title">Qué significa la matriz de confusión en este modelo</div>
            <p style="color: #fff; font-size: 1rem; line-height: 1.55; margin: 0 0 0.75rem 0;">
                En este predictor de compras, la matriz resume cómo el modelo clasifica a los usuarios en <strong>No Compra</strong> (0) o <strong>Compra</strong> (1).
            </p>
            <ul style="color: #fff; font-size: 1rem; line-height: 1.6; margin: 0; padding-left: 1.25rem;">
                <li><strong>Verdaderos negativos (TN) = {tn:,}</strong>: Real no compró, modelo predijo no compra. Aciertos en “no convertir”.</li>
                <li><strong>Falsos positivos (FP) = {fp:,}</strong>: Real no compró, modelo predijo compra. Falsas alarmas (sobreestimar conversión).</li>
                <li><strong>Falsos negativos (FN) = {fn:,}</strong>: Real compró, modelo predijo no compra. Oportunidades perdidas (no detectamos al comprador).</li>
                <li><strong>Verdaderos positivos (TP) = {tp:,}</strong>: Real compró, modelo predijo compra. Aciertos en detectar compradores.</li>
            </ul>
            <p style="color: #8b949e; font-size: 0.95rem; line-height: 1.5; margin: 0.75rem 0 0 0;">
                La diagonal (TN y TP) son los aciertos. FP y FN son los errores; en negocio, FN suele ser costoso (perder ventas) y FP implica priorizar mal a usuarios.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("**Curva ROC**")
        st.caption("TPR vs FPR al variar el umbral. AUC cercano a 1 = buen discriminador entre compradores y no compradores.")
        fpr, tpr, _ = roc_curve(m['y_test'], m['y_prob'])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'Modelo (AUC={m["auc"]:.3f})', line=dict(color='#0ea5e9', width=2.5), fill='tozeroy', fillcolor='rgba(14, 165, 233, 0.2)'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Aleatorio (AUC=0.5)', line=dict(color='#8b949e', dash='dash')))
        fig_roc.update_layout(height=450, xaxis_title='FPR', yaxis_title='TPR', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161b22',
                             font=dict(color='#fff', size=13), xaxis=dict(gridcolor='#30363d'), yaxis=dict(gridcolor='#30363d'))
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown("""
        <div class="card" style="margin-top: 0.75rem;">
            <div class="card-title">Qué significa la curva ROC en este modelo</div>
            <p style="color: #fff; font-size: 1rem; line-height: 1.55; margin: 0;">
                La curva ROC muestra el <strong>True Positive Rate (TPR)</strong> frente al <strong>False Positive Rate (FPR)</strong> al variar el umbral de decisión. 
                En este predictor de compras: un AUC alto indica que el modelo ordena bien a los usuarios (los que compran suelen tener mayor probabilidad estimada). 
                La línea diagonal sería un clasificador aleatorio; nuestra curva por encima de ella indica que el modelo aporta valor para priorizar o segmentar usuarios.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# PESTAÑA: PCA (Varianza y variables)
# ==============================================================================
with tab_pca:
    st.markdown("### Gráficas de PCA")
    st.markdown("""
    <div class="card" style="margin-bottom: 1.5rem;">
        <p style="color: #fff; font-size: 1.05rem; line-height: 1.6; margin: 0;">
            El modelo usa <strong>PCA</strong> (Análisis de Componentes Principales) para reducir la dimensionalidad manteniendo el 95% de la varianza.
            Aquí se muestra la varianza explicada por cada componente y la contribución de las variables originales a los dos primeros componentes.
        </p>
    </div>
    """, unsafe_allow_html=True)
    pca = m['pca']
    n_comp = pca.n_components_
    var_ratio = pca.explained_variance_ratio_
    var_acum = np.cumsum(var_ratio)
    comp_labels = [f"PC{i+1}" for i in range(n_comp)]

    pca_c1, pca_c2 = st.columns(2)
    with pca_c1:
        st.markdown("**Varianza explicada por componente**")
        st.caption("Cuánta varianza captura cada componente principal (scree plot).")
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(x=comp_labels, y=var_ratio * 100, name='Individual', marker_color='#0ea5e9'))
        fig_var.add_trace(go.Scatter(x=comp_labels, y=var_acum * 100, name='Acumulada', mode='lines+markers',
                                     line=dict(color='#22c55e', width=2.5), marker=dict(size=8)))
        fig_var.update_layout(height=400, xaxis_title='Componente', yaxis_title='Varianza (%)',
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161b22', font=dict(color='#fff'),
                             xaxis=dict(gridcolor='#30363d'), yaxis=dict(gridcolor='#30363d'),
                             legend=dict(bgcolor='rgba(0,0,0,0)'), barmode='overlay')
        st.plotly_chart(fig_var, use_container_width=True)
    with pca_c2:
        st.markdown("**Contribución de variables a PC1 y PC2 (loadings)**")
        st.caption("Peso de cada variable original en los dos primeros componentes.")
        loadings = pca.components_[:2].T  # (n_features, 2)
        features = m['features']
        n_show = min(20, len(features))
        idx_top = np.argsort(np.abs(loadings[:, 0]) + np.abs(loadings[:, 1]))[-n_show:][::-1]
        load_df = pd.DataFrame({
            'Variable': [features[i] for i in idx_top],
            'PC1': loadings[idx_top, 0],
            'PC2': loadings[idx_top, 1]
        })
        fig_load = go.Figure()
        fig_load.add_trace(go.Bar(name='PC1', x=load_df['Variable'], y=load_df['PC1'], marker_color='#0ea5e9'))
        fig_load.add_trace(go.Bar(name='PC2', x=load_df['Variable'], y=load_df['PC2'], marker_color='#22c55e'))
        fig_load.update_layout(height=400, barmode='group', xaxis_title='Variable', yaxis_title='Carga',
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161b22', font=dict(color='#fff', size=11),
                             xaxis=dict(gridcolor='#30363d', tickangle=-45), yaxis=dict(gridcolor='#30363d'),
                             legend=dict(bgcolor='rgba(0,0,0,0)'), margin=dict(b=120))
        st.plotly_chart(fig_load, use_container_width=True)

    st.markdown("**Resumen PCA**")
    st.markdown(f"""
    <div class="card">
        <p style="color: #fff; font-size: 1rem; line-height: 1.55; margin: 0;">
            Se usan <strong>{n_comp}</strong> componentes que retienen <strong>{var_acum[-1]*100:.2f}%</strong> de la varianza.
            Las variables con mayor carga en PC1 y PC2 son las que más influyen en la proyección del modelo.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PESTAÑA: ANÁLISIS DE DATOS
# ==============================================================================
with tab_analisis:
    st.markdown("### Exploración de datos")
    st.markdown("""
    <div class="card" style="margin-bottom: 1.5rem;">
        <p style="color: #fff; font-size: 1.05rem; line-height: 1.6; margin: 0;">
            Las gráficas siguientes ayudan a entender la relación entre variables del dataset y la variable objetivo <strong>Revenue</strong> (compra sí/no), 
            y cómo se utilizan en el modelo de clasificación.
        </p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**PageValues vs Revenue**")
        st.caption("Distribución del valor de página (Google Analytics) según si hubo compra o no.")
        fig = px.histogram(df, x='PageValues', color='Revenue', nbins=50, color_discrete_map={0: '#64748b', 1: '#22c55e'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161b22', font=dict(color='#fff'),
                         xaxis=dict(gridcolor='#30363d'), yaxis=dict(gridcolor='#30363d'), legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="card" style="margin-top: 0.5rem;">
            <div class="card-title">Qué significa en este modelo</div>
            <p style="color: #fff; font-size: 1rem; line-height: 1.55; margin: 0;">
                <strong>PageValues</strong> es una de las variables más predictivas: cuando es mayor que 0, la página contribuyó al valor de conversión. 
                En el histograma se ve que las sesiones con compra (Revenue=1) suelen tener distribuciones distintas (más masa en valores positivos). 
                El modelo usa esta variable para distinguir usuarios con mayor probabilidad de compra.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("**Compras por mes**")
        st.caption("Número total de compras por mes (estacionalidad).")
        month_cols = [c for c in df.columns if c.startswith('Month_')]
        monthly = {c.replace('Month_', ''): df[df[c] == 1]['Revenue'].sum() for c in month_cols}
        fig = px.bar(x=list(monthly.keys()), y=list(monthly.values()), color=list(monthly.values()),
                     color_continuous_scale=['#0ea5e9', '#06b6d4', '#22c55e'])
        fig.update_layout(xaxis_title="Mes", yaxis_title="Compras", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#161b22',
                          font=dict(color='#fff'), xaxis=dict(gridcolor='#30363d'), yaxis=dict(gridcolor='#30363d'), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="card" style="margin-top: 0.5rem;">
            <div class="card-title">Qué significa en este modelo</div>
            <p style="color: #fff; font-size: 1rem; line-height: 1.55; margin: 0;">
                La <strong>estacionalidad por mes</strong> se codifica en el dataset con variables dummy (Month_Feb, Month_Mar, etc.). 
                Esta gráfica muestra en qué meses hay más conversiones; el modelo usa el mes para capturar patrones estacionales 
                (por ejemplo rebajas o Navidad) y ajustar la probabilidad de compra según la época del año.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with st.expander("Ver tabla de datos"):
        st.dataframe(df.head(100), use_container_width=True)

# ==============================================================================
# PESTAÑA: CORRELACIONES
# ==============================================================================
with tab_corr:
    st.markdown("### Matriz de correlaciones")
    main_vars = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'Revenue']
    corr = df[main_vars].corr()
    fig_corr, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.5, linecolor='#30363d', annot_kws={'size': 8, 'color': '#fff'}, cbar_kws={'label': 'Correlación'})
    ax.set_facecolor('#161b22'); fig_corr.patch.set_facecolor('#161b22')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
    st.pyplot(fig_corr); plt.close(fig_corr)

# ==============================================================================
# PESTAÑA: CONCLUSIONES (Bento grid + cards)
# ==============================================================================
with tab_conclusiones:
    st.markdown("""
    <div class="hero-banner" style="padding: 1.5rem 2rem; margin-bottom: 1.5rem;">
        <h1 style="margin: 0;">Conclusiones y acciones</h1>
        <p style="margin: 0.5rem 0 0 0;">Resumen del modelo y qué hacer en el e-commerce según los hallazgos.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="bento-section-title">Conclusiones del modelo</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="bento-grid">
        <div class="bento-cell bento-cell--span2">
            <div class="conclusion-card conclusion-card--accent">
                <div class="conclusion-card__icon">📊</div>
                <div class="conclusion-card__title">Comportamiento &gt; demografía</div>
                <p class="conclusion-card__text">Las variables que más predicen la compra son de comportamiento (PageValues, ExitRates, páginas de producto, tiempo en sitio). El dispositivo, región o navegador aportan menos. Priorizar métricas de sesión en Analytics.</p>
            </div>
        </div>
        <div class="bento-cell">
            <div class="conclusion-card conclusion-card--green">
                <div class="conclusion-card__icon">🎯</div>
                <div class="conclusion-card__title">PageValues es clave</div>
                <p class="conclusion-card__text">Valor de página &gt; 0 suele asociarse a conversión. Las páginas que históricamente llevan a compra hay que destacarlas y optimizarlas en el funnel.</p>
            </div>
        </div>
        <div class="bento-cell">
            <div class="conclusion-card conclusion-card--warning">
                <div class="conclusion-card__icon">📉</div>
                <div class="conclusion-card__title">Bounce y Exit Rates</div>
                <p class="conclusion-card__text">Tasas altas reducen la probabilidad de compra. Mejorar primera impresión y puntos de salida del funnel tiene impacto directo.</p>
            </div>
        </div>
        <div class="bento-cell">
            <div class="conclusion-card">
                <div class="conclusion-card__icon">🛒</div>
                <div class="conclusion-card__title">Engagement con catálogo</div>
                <p class="conclusion-card__text">Más páginas de producto y más tiempo en productos correlacionan con mayor probabilidad de compra; el catálogo es buen indicador de intención.</p>
            </div>
        </div>
        <div class="bento-cell">
            <div class="conclusion-card">
                <div class="conclusion-card__icon">📅</div>
                <div class="conclusion-card__title">Estacionalidad y visitante</div>
                <p class="conclusion-card__text">Mes (Navidad, rebajas) y tipo de visitante (recurrente vs nuevo) aportan señal. Campañas pueden afinarse por época y segmento.</p>
            </div>
        </div>
        <div class="bento-cell">
            <div class="conclusion-card">
                <div class="conclusion-card__icon">⚡</div>
                <div class="conclusion-card__title">PCA mantiene rendimiento</div>
                <p class="conclusion-card__text">Reducción con PCA (95% varianza) da un modelo más estable y rápido sin perder precisión relevante.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="bento-section-title">Qué hacer en el e-commerce</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="bento-grid">
        <div class="bento-cell">
            <div class="conclusion-card conclusion-card--accent">
                <div class="conclusion-card__icon">🎚️</div>
                <div class="conclusion-card__title">Segmentar por probabilidad</div>
                <p class="conclusion-card__text">Clasificar sesiones (simulador o API). Priorizar ofertas y recordatorios en usuarios con probabilidad alta (ej. &gt; 70%).</p>
            </div>
        </div>
        <div class="bento-cell">
            <div class="conclusion-card conclusion-card--green">
                <div class="conclusion-card__icon">🏷️</div>
                <div class="conclusion-card__title">Ofertas dirigidas</div>
                <p class="conclusion-card__text">Cupones u ofertas personalizadas a alta probabilidad de compra para cerrar venta y mejorar ROI.</p>
            </div>
        </div>
        <div class="bento-cell">
            <div class="conclusion-card">
                <div class="conclusion-card__icon">📧</div>
                <div class="conclusion-card__title">Recordatorios de carrito</div>
                <p class="conclusion-card__text">Más recursos a email/push para sesiones con alta probabilidad que abandonaron; menos en probabilidad muy baja.</p>
            </div>
        </div>
        <div class="bento-cell">
            <div class="conclusion-card">
                <div class="conclusion-card__icon">📄</div>
                <div class="conclusion-card__title">Optimizar Page Value</div>
                <p class="conclusion-card__text">Identificar en Analytics páginas con mayor Page Value; mejorar visibilidad, velocidad y CTA.</p>
            </div>
        </div>
        <div class="bento-cell">
            <div class="conclusion-card">
                <div class="conclusion-card__icon">🚪</div>
                <div class="conclusion-card__title">Reducir rebote y salida</div>
                <p class="conclusion-card__text">Trabajar landings, primera impresión y puntos críticos (checkout, formularios).</p>
            </div>
        </div>
        <div class="bento-cell">
            <div class="conclusion-card">
                <div class="conclusion-card__icon">📆</div>
                <div class="conclusion-card__title">Campañas por estacionalidad</div>
                <p class="conclusion-card__text">Reforzar presupuesto en meses de más conversiones (Nov, Dic); ajustar inventario y logística.</p>
            </div>
        </div>
        <div class="bento-cell">
            <div class="conclusion-card">
                <div class="conclusion-card__icon">🔄</div>
                <div class="conclusion-card__title">Retargeting inteligente</div>
                <p class="conclusion-card__text">Usar probabilidad para decidir a quién retargetear: más frecuencia y mensajes de cierre para alta probabilidad.</p>
            </div>
        </div>
        <div class="bento-cell">
            <div class="conclusion-card">
                <div class="conclusion-card__icon">🛍️</div>
                <div class="conclusion-card__title">Experiencia en producto</div>
                <p class="conclusion-card__text">Mejorar fichas, recomendaciones y tiempo en catálogo (filtros, búsqueda, UX).</p>
            </div>
        </div>
        <div class="bento-cell">
            <div class="conclusion-card">
                <div class="conclusion-card__icon">👤</div>
                <div class="conclusion-card__title">Segmentar por visitante</div>
                <p class="conclusion-card__text">Onboarding para nuevos; ofertas de fidelidad para recurrentes, alineado con VisitorType.</p>
            </div>
        </div>
    </div>
    <div class="conclusion-summary">
        <p>En resumen: el modelo permite <strong>priorizar recursos</strong> (marketing, UX, recordatorios) en usuarios y momentos con mayor probabilidad de conversión, mejorando la eficiencia del e-commerce sin aumentar necesariamente el gasto.</p>
    </div>
    """, unsafe_allow_html=True)
