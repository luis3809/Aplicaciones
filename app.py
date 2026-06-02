import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("🔵 Clustering K-Means para Datos Industriales")

archivo = st.file_uploader("📂 Carga tu archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    st.subheader("📄 Vista previa del dataset")
    st.dataframe(df)

    # Seleccionar columnas numéricas
    variables = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    if len(variables) < 2:
        st.error("El dataset necesita al menos 2 columnas numéricas.")
        st.stop()

    st.subheader("📌 Columnas numéricas detectadas")
    st.write(variables)

    # Escalado
    scaler = StandardScaler()
    X = scaler.fit_transform(df[variables])

    # Selección de número de clusters
    k = st.slider("Selecciona el número de clusters", 2, 10, 3)

    # Modelo K-Means
    modelo = KMeans(n_clusters=k, random_state=42)
    df["cluster"] = modelo.fit_predict(X)

    st.subheader("📊 Resultados del clustering")
    st.write(df[["cluster"] + variables].head())

    # PCA para visualización 2D
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(X)

    st.subheader("🌈 Visualización 2D (PCA + K-Means)")

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=df["cluster"],
        cmap="tab10",
        s=40,
        alpha=0.8
    )

    plt.colorbar(scatter, label="Cluster")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
    ax.set_title("Clustering K-Means en 2D (PCA)")

    st.pyplot(fig)

    st.success("Clustering completado exitosamente.")