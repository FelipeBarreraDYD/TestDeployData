"""
ANALIZADOR DE DATASETS CON IA
Aplicación que analiza automáticamente cualquier dataset usando IA
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from io import StringIO

# Configuración inicial
st.set_page_config(
    page_title="Analizador de Datasets con IA",
    page_icon="🤖",
    layout="wide"
)

# Cargar modelo de lenguaje con manejo de errores y configuración óptima
@st.cache_resource(show_spinner="Cargando modelo de IA...")
def load_ai_model():
    try:
        return pipeline(
            task="text2text-generation",  # Más específico para FLAN-T5
            model="google/flan-t5-small",
            device=-1,  # Forzar CPU
            torch_dtype="auto" if False else None,  # Evitar uso de GPU
            max_length=200,  # Longitud máxima por defecto
            truncation=True
        )
    except Exception as e:
        st.error(f"Error cargando el modelo: {str(e)}")
        return None

# Cargar con verificación
generator = load_ai_model()

if generator is None:
    st.stop()  # Detener la app si falla la carga
generator = load_ai_model()

# Función para generar texto con IA
def generate_ai_text(prompt, max_length=200):
    try:
        response = generator(
            prompt,
            max_length=max_length,
            do_sample=True,
            temperature=0.7
        )
        return response[0]['generated_text']
    except Exception as e:
        st.error(f"Error en la generación: {str(e)}")
        return ""

# Procesamiento de datos
def process_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            return None
        
        # Limpieza básica
        df = df.dropna(axis=1, how='all')
        df = df.dropna()
        return df
    
    except Exception as e:
        st.error(f"Error procesando archivo: {str(e)}")
        return None

# Interfaz principal
st.title("🤖 Analizador Inteligente de Datasets")
st.markdown("Carga cualquier dataset y obtén un análisis automático con IA")

# Sidebar para carga de datos
with st.sidebar:
    st.header("Configuración")
    uploaded_file = st.file_uploader("Sube tu dataset", type=["csv", "xlsx"])
    analyze_button = st.button("Analizar Dataset")

# Sección de análisis
if analyze_button and uploaded_file:
    df = process_data(uploaded_file)
    
    if df is not None:
        # Generar descripción inicial con IA
        sample_data = df.head(3).to_csv(index=False)
        prompt = f"""
        Describe este dataset basado en sus primeras filas: 
        {sample_data}
        Columnas: {', '.join(df.columns)}
        Características principales:
        """
        
        with st.spinner("Generando análisis con IA..."):
            ai_description = generate_ai_text(prompt)
            
        st.header("Descripción General del Dataset")
        st.write(ai_description)
        
        # Análisis estadístico automático
        st.header("Análisis Estadístico")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Resumen Estadístico")
            st.dataframe(df.describe())
        
        with col2:
            st.subheader("Tipos de Datos")
            dtype_info = StringIO()
            df.info(buf=dtype_info, verbose=False)
            st.text(dtype_info.getvalue())
        
        # Visualizaciones automáticas
        st.header("Visualizaciones Automáticas")
        
        # Seleccionar columnas numéricas
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Generar matriz de correlación con descripción IA
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.subheader("Matriz de Correlación")
            st.pyplot(fig)
            
            corr_prompt = f"""
            Explica esta matriz de correlación entre las variables numéricas: 
            {', '.join(numeric_cols)}. 
            Destaca las correlaciones más importantes.
            """
            corr_analysis = generate_ai_text(corr_prompt)
            st.write(corr_analysis)
        
        # Histogramas automáticos
        if numeric_cols:
            selected_col = st.selectbox("Selecciona una columna para histograma", numeric_cols)
            
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, ax=ax)
            st.pyplot(fig)
            
            hist_prompt = f"""
            Analiza la distribución de la columna {selected_col} basado en su histograma.
            """
            hist_analysis = generate_ai_text(hist_prompt)
            st.write(hist_analysis)
        
        # Análisis de variables categóricas
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            st.subheader("Análisis de Variables Categóricas")
            selected_cat = st.selectbox("Selecciona columna categórica", categorical_cols)
            
            fig, ax = plt.subplots()
            df[selected_cat].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
            
            cat_prompt = f"""
            Analiza la distribución de la variable categórica {selected_cat} 
            basado en su gráfico de barras.
            """
            cat_analysis = generate_ai_text(cat_prompt)
            st.write(cat_analysis)

elif analyze_button and not uploaded_file:
    st.warning("Por favor sube un archivo primero")

else:
    st.info("""
    Instrucciones:
    1. Sube tu dataset (CSV o Excel)
    2. Haz clic en 'Analizar Dataset'
    3. Espera los resultados generados por IA
    """)