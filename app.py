"""
APLICACIÓN STREAMLIT PARA DESPLIEGUE DEL MODELO DE PRECIOS DE VIVIENDAS
Esta aplicación permite a los usuarios interactuar con el modelo para predecir precios de viviendas
y visualizar el análisis exploratorio de datos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from PIL import Image

# Configurar la página
st.set_page_config(
    page_title="Predictor de Precios de Viviendas",
    page_icon="🏠",
    layout="wide"
)

# Cargar modelo y datos originales
@st.cache_data
def load_model_data():
    try:
        return pd.read_csv('housing_data.csv')
    except FileNotFoundError:
        st.error("Archivo de datos original no encontrado.")
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/housing_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Archivos del modelo no encontrados.")
        return None, None

model_df = load_model_data()
model, scaler = load_model()

# Sidebar para carga de datos y configuración
st.sidebar.header("Cargar y Configurar Datos")

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
eda_df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            eda_df = pd.read_csv(uploaded_file)
        else:
            eda_df = pd.read_excel(uploaded_file)
            
        # Aplicar limpieza
        if clean_method == "Rellenar con 0":
            eda_df.fillna(0, inplace=True)
        else:
            eda_df.dropna(inplace=True)
            
    except Exception as e:
        st.sidebar.error(f"Error al cargar archivo: {str(e)}")
else:
    eda_df = model_df.copy()

# Título de la aplicación
st.title("🏠 Predictor de Precios de Viviendas")
st.markdown("Esta aplicación permite predecir el precio de viviendas basado en características clave.")

# Sidebar para navegación
page = st.sidebar.radio("Navegación", ["Inicio", "Análisis Exploratorio", "Predicción", "Acerca de"])

# Página de inicio
if page == "Inicio":
    st.header("Bienvenido al Predictor de Precios de Viviendas")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📊 ¿Qué puede hacer esta aplicación?
        
        - **Explorar datos** de viviendas y sus características
        - **Visualizar relaciones** entre diferentes variables
        - **Predecir precios** basados en un modelo entrenado
        """)
        
        if eda_df is not None:
            st.subheader("Vista previa de los datos")
            st.dataframe(eda_df.head())
    
    with col2:
        if eda_df is not None:
            st.markdown("### 📈 Relación entre variables")
            numeric_cols = eda_df.select_dtypes(include=np.number).columns.tolist()
            
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("Variable X", numeric_cols)
                y_col = st.selectbox("Variable Y", numeric_cols)
                
                fig, ax = plt.subplots()
                sns.scatterplot(x=x_col, y=y_col, data=eda_df, ax=ax)
                ax.set_title(f'{x_col} vs {y_col}')
                st.pyplot(fig)
            else:
                st.warning("No hay suficientes variables numéricas para mostrar gráficos.")

# Página de análisis exploratorio
elif page == "Análisis Exploratorio":
    st.header("Análisis Exploratorio de Datos")
    
    if eda_df is not None:
        st.markdown("""
        Esta sección muestra diferentes visualizaciones de los datos para entender mejor las relaciones
        entre las variables y su impacto en el precio de las viviendas.
        """)
        
        # Matriz de correlación
        st.subheader("Matriz de Correlación")
        numeric_cols = eda_df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) > 1:
            corr = eda_df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                        square=True, linewidths=.5, annot=True, fmt='.2f')
            st.pyplot(fig)
        else:
            st.warning("No hay suficientes variables numéricas para la matriz de correlación.")
        
        # Distribución de variables
        st.subheader("Distribución de Variables")
        var = st.selectbox("Seleccione una variable", numeric_cols)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(eda_df[var], kde=True, ax=ax)
        ax.set_title(f'Distribución de {var}')
        st.pyplot(fig)
        
        # Exploración interactiva
        st.subheader("Exploración Interactiva")
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("Variable X", numeric_cols)
        
        with col2:
            y_var = st.selectbox("Variable Y", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=x_var, y=y_var, data=eda_df, ax=ax)
        ax.set_title(f'Relación entre {x_var} y {y_var}')
        st.pyplot(fig)

# Página de predicción
elif page == "Predicción":
    st.header("Predicción de Precios de Viviendas")
    
    if model and scaler and model_df is not None:
        with st.form("prediction_form"):
            st.subheader("Ingrese las características de la vivienda")
            
            col1, col2 = st.columns(2)
            
            with col1:
                rm = st.slider("Número medio de habitaciones (RM)", 
                               float(model_df['RM'].min()), 
                               float(model_df['RM'].max()), 
                               float(model_df['RM'].mean()))
                
                lstat = st.slider("% de población de estatus bajo (LSTAT)", 
                                  float(model_df['LSTAT'].min()), 
                                  float(model_df['LSTAT'].max()), 
                                  float(model_df['LSTAT'].mean()))
            
            with col2:
                ptratio = st.slider("Ratio alumno-profesor (PTRATIO)", 
                                    float(model_df['PTRATIO'].min()), 
                                    float(model_df['PTRATIO'].max()), 
                                    float(model_df['PTRATIO'].mean()))
                
                dis = st.slider("Distancia a centros de empleo (DIS)", 
                                float(model_df['DIS'].min()), 
                                float(model_df['DIS'].max()), 
                                float(model_df['DIS'].mean()))
            
            if st.form_submit_button("Predecir Precio"):
                input_data = np.array([[rm, lstat, ptratio, dis]])
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                st.success(f"Precio predicho: ${prediction * 1000:.2f} USD")
    else:
        st.error("Error al cargar el modelo. Verifique los archivos del modelo.")

# Página Acerca de
elif page == "Acerca de":
    st.header("Acerca de")
    st.markdown("""
    ### Características principales:
    - **Carga de datos**: Soporta archivos CSV y Excel
    - **Limpieza automática**: Manejo de valores faltantes
    - **Visualización interactiva**: Análisis exploratorio dinámico
    - **Modelo predictivo**: Basado en Random Forest Regressor
    """)

if __name__ == "__main__":
    pass