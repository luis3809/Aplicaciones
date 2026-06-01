import streamlit as st
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.title("🔍 Reducción de Dimensionalidad con UMAP - Dataset Industrial")

# Subir archivo data.csv
archivo = st.file_uploader("📂 Carga tu archivo data.csv", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    st.subheader("📄 Vista previa del dataset")
    st.dataframe(df)

    # Seleccionar columnas numéricas
    variables = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    st.subheader("📌 Columnas numéricas detectadas")
    st.write(variables)

    # Escalado (UMAP funciona mejor con datos normalizados)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[variables])

    # Aplicar UMAP
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42
    )

    embedding = reducer.fit_transform(X)

    # Graficar UMAP
    st.subheader("🌈 Proyección UMAP (2D)")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Colorear por una variable relevante (ej: Flujo inferido)
    color_var = "Flujo inferido (KBPD)" if "Flujo inferido (KBPD)" in df.columns else variables[0]

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=df[color_var],
        cmap="viridis",
        s=20,
        alpha=0.8
    )

    plt.colorbar(scatter, label=color_var)
    ax.set_title("UMAP - Proyección 2D del Dataset Industrial", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    st.pyplot(fig)