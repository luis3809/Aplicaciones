import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Análisis de Correlación - Motor CAT G3600")

# Subir archivo data.csv
archivo = st.file_uploader("📂 Carga tu archivo data.csv", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    st.subheader("📄 Vista previa del dataset")
    st.dataframe(df)

    variables = ["RPM", "Temperatura_aceite", "Temperatura_refrigerante", "Presion_aceite", "Carga"]
    variables_presentes = [v for v in variables if v in df.columns]

    if len(variables_presentes) >= 2:
        st.subheader("📊 Matriz de correlación de Pearson")
        corr = df[variables_presentes].corr(method="pearson")
        st.dataframe(corr)

        st.subheader("🔥 Mapa de calor (Heatmap)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
    else:
        st.error("El archivo debe contener al menos dos columnas válidas.")