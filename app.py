import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("🔧 Clustering de Condiciones de Operación - Bomba Centrífuga API")

st.write("""
Carga tu archivo **data.csv** con las variables reales de la bomba centrífuga API.
El sistema agrupa automáticamente las condiciones de operación en modos:
normal, alerta y crítico.
""")

# --- CARGA DEL ARCHIVO data.csv ---
uploaded_file = st.file_uploader("📂 Sube tu archivo data.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 👀 Vista previa del dataset cargado")
    st.dataframe(df.head())

    # Columnas esperadas del dataset de la bomba
    columnas_bomba = ["horas_operacion", "temperatura", "vibracion", "presion", "fallas"]

    # Validación
    if not all(col in df.columns for col in columnas_bomba):
        st.error("❌ El archivo no contiene las columnas requeridas: " + ", ".join(columnas_bomba))
    else:
        st.success("✔ Archivo válido. Puedes ejecutar el clustering.")

        # Selección del número de clusters
        n_clusters = st.slider("Número de clusters", 2, 6, 3)

        if st.button("🚀 Ejecutar clustering"):
            # Modelo K-Means
            modelo = KMeans(n_clusters=n_clusters, random_state=42)
            df["cluster"] = modelo.fit_predict(df[columnas_bomba])

            # Mostrar centroides
            st.write("### 📌 Centroides de cada cluster")
            centroides = pd.DataFrame(modelo.cluster_centers_, columns=columnas_bomba)
            st.dataframe(centroides)

            # PCA para visualización 2D
            pca = PCA(n_components=2)
            componentes = pca.fit_transform(df[columnas_bomba])
            df["PC1"] = componentes[:, 0]
            df["PC2"] = componentes[:, 1]

            # Gráfico PCA
            fig, ax = plt.subplots()
            scatter = ax.scatter(df["PC1"], df["PC2"], c=df["cluster"], cmap="viridis")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("Visualización de Clusters - Bomba Centrífuga API")
            st.pyplot(fig)

            st.write("### 📄 Dataset con cluster asignado")
            st.dataframe(df)