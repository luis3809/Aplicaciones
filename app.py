import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("📉 PCA 2D - Dataset Industrial")

archivo = st.file_uploader("📂 Carga tu archivo data.csv", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    st.subheader("📄 Vista previa del dataset")
    st.dataframe(df)

    # Seleccionar columnas numéricas
    variables = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    st.subheader("📌 Columnas numéricas detectadas")
    st.write(variables)

    # Escalado
    scaler = StandardScaler()
    X = scaler.fit_transform(df[variables])

    # PCA
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(X)

    # Seleccionar variable de color
    color_var = "Flujo inferido (KBPD)" if "Flujo inferido (KBPD)" in df.columns else variables[0]
    df[color_var] = pd.to_numeric(df[color_var], errors="coerce")

    if df[color_var].isna().all():
        color_var = variables[0]

    # Graficar
    st.subheader("🌈 Proyección PCA (2D)")

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=df[color_var],
        cmap="viridis",
        s=20,
        alpha=0.8
    )

    plt.colorbar(scatter, label=color_var)
    ax.set_title("PCA - Proyección 2D del Dataset Industrial", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")

    st.pyplot(fig)

    # Mostrar cargas de variables
    st.subheader("📊 Importancia de variables en PC1 y PC2")
    cargas = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=variables
    )
    st.dataframe(cargas)