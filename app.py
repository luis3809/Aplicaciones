import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Estilo profesional
sns.set_theme(style="white")

st.title("🔍 Análisis de Correlación - Motor CAT G3600")

archivo = st.file_uploader("📂 Carga tu archivo data.csv", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    st.subheader("📄 Vista previa del dataset")
    st.dataframe(df)

    # Variables esperadas
    variables = ["RPM", "Temperatura_aceite", "Temperatura_refrigerante", "Presion_aceite", "Carga"]
    variables_presentes = [v for v in variables if v in df.columns]

    if len(variables_presentes) >= 2:

        # Matriz de correlación
        st.subheader("📊 Matriz de correlación de Pearson")
        corr = df[variables_presentes].corr(method="pearson")
        st.dataframe(corr)

        # Heatmap profesional
        st.subheader("🔥 Mapa de calor profesional")

        fig, ax = plt.subplots(figsize=(8, 6))

        # Paleta elegante
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Coeficiente de correlación"},
            square=True,
            ax=ax
        )

        ax.set_title("Correlación entre Variables del Motor CAT G3600", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        st.pyplot(fig)

    else:
        st.error("El archivo debe contener al menos dos columnas válidas.")