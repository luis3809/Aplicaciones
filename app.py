import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("📊 PCA con Cargas de Variables")

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

    # Mostrar cargas
    st.subheader("📊 Cargas de Variables en PC1 y PC2")

    cargas = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=variables
    )

    st.dataframe(cargas)

    # Gráfico de cargas
    st.subheader("📈 Gráfico de Cargas (PC1 vs PC2)")

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(cargas["PC1"], cargas["PC2"])

    for i, var in enumerate(variables):
        ax.text(cargas["PC1"][i], cargas["PC2"][i], var)

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Cargas de Variables en PCA")

    st.pyplot(fig)