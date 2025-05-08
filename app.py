"""
ANALIZADOR INTERACTIVO DE DATASETS
Aplicación para explorar y visualizar cualquier conjunto de datos
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# Configurar la página
st.set_page_config(
    page_title="Analizador de Datos",
    page_icon="📊",
    layout="wide"
)

# Cargar datos de ejemplo
@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv('sample_data.csv')
    except FileNotFoundError:
        st.error("Archivo de datos de ejemplo no encontrado.")
        return None

# Sidebar para carga de datos y configuración
st.sidebar.header("Configuración de Datos")

# Cargar datos de usuario
uploaded_file = st.sidebar.file_uploader(
    "Sube tu dataset (CSV o Excel)",
    type=["csv", "xlsx"]
)

# Opciones de limpieza
clean_method = st.sidebar.radio(
    "Manejar valores faltantes:",
    ["Rellenar con 0", "Eliminar filas con NA"]
)

# Procesar datos cargados
current_df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            current_df = pd.read_csv(uploaded_file)
        else:
            current_df = pd.read_excel(uploaded_file)
            
        # Aplicar limpieza
        if clean_method == "Rellenar con 0":
            current_df.fillna(0, inplace=True)
        else:
            current_df.dropna(inplace=True)
            
    except Exception as e:
        st.sidebar.error(f"Error al cargar archivo: {str(e)}")
else:
    current_df = load_sample_data()

# Título de la aplicación
st.title("📊 Analizador Interactivo de Datasets")
st.markdown("Explora y visualiza cualquier conjunto de datos de forma interactiva")

# Sidebar para navegación
page = st.sidebar.radio("Navegación", ["Inicio", "Análisis Exploratorio", "Acerca de"])

# Página de inicio
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
        """)
        
        if current_df is not None:
            st.subheader("Vista previa de los datos")
            st.dataframe(current_df.head())
    
    with col2:
        if current_df is not None:
            st.markdown("### 📌 Resumen Rápido")
            st.write(f"- **Filas:** {current_df.shape[0]}")
            st.write(f"- **Columnas:** {current_df.shape[1]}")
            st.write(f"- **Variables numéricas:** {len(current_df.select_dtypes(include=np.number).columns)}")
            st.write(f"- **Variables categóricas:** {len(current_df.select_dtypes(include=['object', 'category']).columns)}")

# Página de análisis exploratorio
elif page == "Análisis Exploratorio":
    st.header("Análisis Exploratorio de Datos")
    
    if current_df is not None:
        st.markdown("""
        Explora tus datos mediante visualizaciones interactivas y análisis estadísticos
        """)
        
        # Sección de estadísticas
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(current_df.describe())
        
        # Matriz de correlación
        st.subheader("Matriz de Correlación")
        numeric_cols = current_df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) > 1:
            corr = current_df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot(fig)
        else:
            st.warning("Se necesitan al menos 2 variables numéricas para la matriz de correlación")
        
        # Selector de gráficos
        st.subheader("Generador de Gráficos")
        col1, col2 = st.columns(2)
        
        with col1:
            plot_type = st.selectbox("Tipo de gráfico", 
                                   ["Histograma", "Dispersión", "Barras"])
        
        with col2:
            x_var = st.selectbox("Variable X", current_df.columns)
            if plot_type != "Histograma":
                y_var = st.selectbox("Variable Y", current_df.columns)
        
        # Generar gráficos
        fig, ax = plt.subplots()
        try:
            if plot_type == "Histograma":
                sns.histplot(current_df[x_var], kde=True, ax=ax)
                ax.set_title(f'Distribución de {x_var}')
            elif plot_type == "Dispersión":
                sns.scatterplot(x=x_var, y=y_var, data=current_df, ax=ax)
                ax.set_title(f'{x_var} vs {y_var}')
            elif plot_type == "Barras":
                sns.barplot(x=x_var, y=y_var, data=current_df, ax=ax)
                ax.set_title(f'{x_var} vs {y_var}')
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error al generar gráfico: {str(e)}")

# Página Acerca de
elif page == "Acerca de":
    st.header("Acerca de la Aplicación")
    st.markdown("""
    ### Características Principales:
    - **Carga múltiples formatos:** CSV y Excel
    - **Limpieza automática:** Manejo de valores faltantes
    - **Visualización interactiva:** Gráficos personalizables
    - **Análisis estadístico:** Informes descriptivos completos
    
    Desarrollado con Streamlit y Python 🐍
    """)

if __name__ == "__main__":
    pass