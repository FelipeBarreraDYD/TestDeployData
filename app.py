import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# Configuración de página DEBE SER LA PRIMERA LÍNEA
st.set_page_config(
    page_title="Analizador de Datos",
    page_icon="📊",
    layout="wide"
)

# Cargar modelo de IA local compatible
@st.cache_resource
def load_ai_model():
    try:
        return pipeline(
            task="text-generation",
            model="PlanTL-GOB-ES/gpt2-base-bne",  # Modelo compatible en Español
            device=-1
        )
    except Exception as e:
        st.error(f"Error cargando el modelo: {str(e)}")
        st.stop()

generator = load_ai_model()

# Función de análisis optimizada
def generar_analisis_ia(df):
    try:
        sample_data = df.head(3).to_string(max_colwidth=20, max_rows=3)
        
        prompt = f"""
        Analiza este dataset en español:
        Columnas: {', '.join(df.columns)}
        Muestra: {sample_data}

        Genera un informe con:
        1. Descripción general
        2. Dos hallazgos clave
        3. Una recomendación de análisis
        """
        
        response = generator(
            prompt,
            max_length=500,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=1
        )
        return response[0]['generated_text']
        
    except Exception as e:
        return f"Error: {str(e)[:200]}"

# Cache mejorado
@st.cache_data(show_spinner=False)
def cached_ia_analysis(df):
    return generar_analisis_ia(df)

# Cargar datos de ejemplo
@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv('sample_data.csv')
    except FileNotFoundError:
        st.error("Archivo de datos de ejemplo no encontrado.")
        return None

# Sidebar configuración y carga de datos
st.sidebar.header("Configuración de Datos")
uploaded_file = st.sidebar.file_uploader(
    "Sube tu dataset (CSV)",
    type=["csv"]
)
clean_method = st.sidebar.radio(
    "Manejar valores faltantes:",
    ["Rellenar con 0", "Eliminar filas con NA"]
)

# Función de carga modificada para resetear el análisis previo
def load_and_clean(uploader):
    try:
        if uploader.name.endswith('.csv'):
            df = pd.read_csv(uploader)
        else:
            df = pd.read_excel(uploader)
        
        # Resetear análisis anterior al cargar nuevo dataset
        if 'ia_report' in st.session_state:
            del st.session_state.ia_report
            
        if clean_method == "Rellenar con 0":
            df.fillna(0, inplace=True)
        else:
            df.dropna(inplace=True)
        return df
    except Exception as e:
        st.sidebar.error(f"Error al cargar archivo: {str(e)}")
        return None

# Cargar datos (sin ejecución automática de IA)
current_df = load_and_clean(uploaded_file) if uploaded_file else load_sample_data()

# Navbar actualizada
page = st.sidebar.radio(
    "Navegación",
    ["Inicio", "Análisis Exploratorio", "Análisis Descriptivo", "Acerca de"]
)

# Título de la aplicación
st.title("📊 Analizador Interactivo de Datasets")
st.markdown("Explora y visualiza cualquier conjunto de datos de forma interactiva")

# Página Inicio
if page == "Inicio":
    st.header("Bienvenido al Analizador de Datasets")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🚀 ¿Qué puedes hacer?
        - **Visualizar datos** en tablas interactivas
        - **Analizar relaciones** entre variables
        - **Generar gráficos** profesionales
        - **Explorar distribuciones** estadísticas
        """
        )
        if current_df is not None:
            st.subheader("Vista previa de los datos")
            st.dataframe(current_df.head())
    with col2:
        if current_df is not None:
            st.markdown("### 📌 Resumen Rápido")
            st.write(f"- **Filas:** {current_df.shape[0]}")
            st.write(f"- **Columnas:** {current_df.shape[1]}")
            st.write(f"- **Variables numéricas:** {len(current_df.select_dtypes(include=np.number).columns)}")
            st.write(f"- **Variables categóricas:** {len(current_df.select_dtypes(include=['object','category']).columns)}")

# Página Análisis Exploratorio
elif page == "Análisis Exploratorio":
    st.header("Análisis Exploratorio de Datos")
    if current_df is not None:
        st.markdown("Explora tus datos mediante visualizaciones interactivas y análisis estadísticos")
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(current_df.describe())
        st.subheader("Matriz de Correlación")
        num_cols = current_df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) > 1:
            corr = current_df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot(fig)
        else:
            st.warning("Se necesitan al menos 2 variables numéricas para la matriz de correlación")
        st.subheader("Generador de Gráficos")
        c1, c2 = st.columns(2)
        with c1:
            plot_type = st.selectbox("Tipo de gráfico", ["Histograma","Dispersión","Barras"])
        with c2:
            x_var = st.selectbox("Variable X", current_df.columns)
            y_var = None
            if plot_type != "Histograma":
                y_var = st.selectbox("Variable Y", current_df.columns)
        fig, ax = plt.subplots()
        try:
            if plot_type == "Histograma":
                sns.histplot(current_df[x_var], kde=True, ax=ax)
                ax.set_title(f'Distribución de {x_var}')
            elif plot_type == "Dispersión":
                sns.scatterplot(x=x_var, y=y_var, data=current_df, ax=ax)
                ax.set_title(f'{x_var} vs {y_var}')
            else:
                sns.barplot(x=x_var, y=y_var, data=current_df, ax=ax)
                ax.set_title(f'{x_var} vs {y_var}')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error al generar gráfico: {str(e)}")

# Página Análisis Descriptivo
elif page == "Análisis Descriptivo":
    st.header("Análisis Descriptivo con IA")
    
    if current_df is None:
        st.warning("Primero carga un dataset")
    else:
        # Sección de controles
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### Generar nuevo análisis")
            
            # Botón para análisis completo
            if st.button("🧠 Ejecutar Análisis Completo con IA"):
                with st.spinner("Analizando dataset. Esto puede tomar 1-2 minutos..."):
                    try:
                        st.session_state.ia_report = cached_ia_analysis(current_df)
                        st.success("¡Análisis completado!")
                    except Exception as e:
                        st.error(f"Error en el análisis: {str(e)}")
        
        # Sección de resultados
        if 'ia_report' in st.session_state:
            st.markdown("---")
            st.markdown("## 📄 Informe Generado")
            st.markdown(st.session_state.ia_report)
            
            # Metadata del análisis
            with st.expander("Detalles técnicos"):
                st.write(f"Filas analizadas: {len(current_df)}")
                st.write(f"Columnas analizadas: {len(current_df.columns)}")
                st.write(f"Tamaño del informe: {len(st.session_state.ia_report)//4} tokens aproximados")
        else:
            st.info("Presiona el botón para generar un análisis con IA")

# Página Acerca de
elif page == "Acerca de":
    st.header("Acerca de la Aplicación")
    st.markdown("""
    ### Características Principales:
    - **Carga múltiples formatos:** CSV y Excel
    - **Limpieza automática:** Manejo de valores faltantes
    - **Visualización interactiva:** Gráficos personalizables
    - **Análisis estadístico:** Informes descriptivos completos
    - **Análisis IA automatizado:** Se ejecuta al cargar datos

    Desarrollado con Streamlit y Python 🐍
    """
    )

if __name__ == "__main__":
    pass
