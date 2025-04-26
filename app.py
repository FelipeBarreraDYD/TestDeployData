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

# Funciones para cargar datos y modelos
@st.cache_data
def load_data():
    try:
        return pd.read_csv('housing_data.csv')
    except FileNotFoundError:
        st.error("No se encontró el archivo de datos. Por favor, asegúrate de que housing_data.csv existe en el directorio.")
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/housing_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo. Asegúrate de que los archivos existen en la carpeta models/.")
        return None, None

# Cargar datos y modelo
df = load_data()
model, scaler = load_model()

# Título de la aplicación
st.title("🏠 Predictor de Precios de Viviendas")
st.markdown("Esta aplicación permite predecir el precio de viviendas basado en características clave.")

# Sidebar para navegación
page = st.sidebar.radio("Navegación", ["Inicio", "Análisis Exploratorio", "Predicción", "Acerca de", "Contacto"])

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
        
        Utilice el menú de navegación para explorar las diferentes secciones.
        """)
        
        if df is not None:
            st.subheader("Vista previa de los datos")
            st.dataframe(df.head())
    
    with col2:
        if df is not None:
            st.markdown("### 📈 Precio promedio por número de habitaciones")
            # Agrupar por rango de habitaciones
            fig, ax = plt.subplots()
            df['RM_bin'] = pd.cut(df['RM'], bins=5)
            grouped = df.groupby('RM_bin')['PRICE'].mean().reset_index()
            sns.barplot(x='RM_bin', y='PRICE', data=grouped, ax=ax)
            ax.set_xlabel('Número de habitaciones (agrupado)')
            ax.set_ylabel('Precio promedio')
            plt.xticks(rotation=45)
            st.pyplot(fig)

# Página de análisis exploratorio
elif page == "Análisis Exploratorio":
    st.header("Análisis Exploratorio de Datos")
    
    if df is not None:
        st.markdown("""
        Esta sección muestra diferentes visualizaciones de los datos para entender mejor las relaciones
        entre las variables y su impacto en el precio de las viviendas.
        """)
        
        # Matriz de correlación
        st.subheader("Matriz de Correlación")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, annot=True, fmt='.2f')
        st.pyplot(fig)
        
        # Relaciones entre variables
        st.subheader("Relaciones con el Precio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x='RM', y='PRICE', data=df, ax=ax)
            ax.set_title('Habitaciones vs Precio')
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x='PTRATIO', y='PRICE', data=df, ax=ax)
            ax.set_title('Ratio Alumno-Profesor vs Precio')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x='LSTAT', y='PRICE', data=df, ax=ax)
            ax.set_title('Estatus Bajo (%) vs Precio')
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x='DIS', y='PRICE', data=df, ax=ax)
            ax.set_title('Distancia a Centros de Empleo vs Precio')
            st.pyplot(fig)
        
        # Distribución de precios
        st.subheader("Distribución de Precios")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['PRICE'], kde=True, ax=ax)
        ax.set_title('Distribución de Precios de Viviendas')
        st.pyplot(fig)
        
        # Exploración interactiva
        st.subheader("Exploración Interactiva")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("Variable X", options=df.columns.tolist())
        
        with col2:
            y_var = st.selectbox("Variable Y", options=df.columns.tolist(), index=4)  # Default to PRICE
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=x_var, y=y_var, data=df, ax=ax)
        ax.set_title(f'Relación entre {x_var} y {y_var}')
        st.pyplot(fig)
    else:
        st.error("No se pueden mostrar visualizaciones sin datos.")

# Página de predicción
elif page == "Predicción":
    st.header("Predicción de Precios de Viviendas")
    
    st.markdown("""
    En esta sección puede ingresar las características de una vivienda y obtener una predicción 
    del precio basada en el modelo entrenado.
    """)
    
    if model is not None and scaler is not None and df is not None:
        # Formulario para ingresar valores
        with st.form("prediction_form"):
            st.subheader("Ingrese las características de la vivienda")
            
            col1, col2 = st.columns(2)
            
            with col1:
                rm = st.slider("Número medio de habitaciones (RM)", 
                               float(df['RM'].min()), 
                               float(df['RM'].max()), 
                               float(df['RM'].mean()))
                
                lstat = st.slider("% de población de estatus bajo (LSTAT)", 
                                  float(df['LSTAT'].min()), 
                                  float(df['LSTAT'].max()), 
                                  float(df['LSTAT'].mean()))
            
            with col2:
                ptratio = st.slider("Ratio alumno-profesor (PTRATIO)", 
                                    float(df['PTRATIO'].min()), 
                                    float(df['PTRATIO'].max()), 
                                    float(df['PTRATIO'].mean()))
                
                dis = st.slider("Distancia a centros de empleo (DIS)", 
                                float(df['DIS'].min()), 
                                float(df['DIS'].max()), 
                                float(df['DIS'].mean()))
            
            submit_button = st.form_submit_button("Predecir Precio")
        
        # Mostrar predicción cuando se envía el formulario
        if submit_button:
            # Crear un array con los valores ingresados
            input_data = np.array([[rm, lstat, ptratio, dis]])
            
            # Estandarizar los datos
            input_scaled = scaler.transform(input_data)
            
            # Realizar la predicción
            prediction = model.predict(input_scaled)[0]
            
            # Mostrar el resultado
            st.success(f"El precio predicho para esta vivienda es: ${prediction:.2f}k")
            
            # Mostrar interpretación de la predicción
            st.subheader("Interpretación de la predicción")

elif page == "Contacto":
    st.header("Contacto")
    
    st.markdown("""
    Esta aplicación fue desarrollada como parte de un proyecto para predecir precios de viviendas
    utilizando un modelo de regresión. El modelo fue entrenado con el conjunto de datos Boston Housing.
    
    ### Desarrollador
    - **Nombre:** Felipe Barrera"
    - **Email:** fbarrerah9@gmail.com
                """)