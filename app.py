import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Estilo profesional
sns.set_theme(style="whitegrid")

st.title("🔥 Mapa de Calor Profesional - Motor CAT G3600")

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

        st.subheader("🎨 Heatmap Profesional con Colores Intensos")

        fig, ax = plt.subplots(figsize=(9, 6))

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",          # Paleta con pigmentación fuerte
            center=0,               # Asegura contraste
            robust=True,            # Evita colores apagados
            linewidths=1,           # Líneas más visibles
            linecolor="black",      # Bordes profesionales
            cbar_kws={"shrink": 0.8, "label": "Coeficiente de correlación"},
            square=True,
            ax=ax
        )

        ax.set_title("Correlación entre Variables del Motor CAT G3600", fontsize=16, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        st.pyplot(fig)

    else:
        st.error("El archivo debe contener al menos dos columnas válidas.")