import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#cCABECERA Y TITULO

st.markdown("""
    <div style='padding:10px 0 20px 0;'>
        <h1 style='font-size:42px; font-weight:800; margin-bottom:0;'>
            Aprendizaje automático no supervisado
        </h1>
        <h1 style='font-size:48px; font-weight:900; color:#4A90E2; margin-top:0;'>
            K-Means
        </h1>
    </div>
""", unsafe_allow_html=True)

st.title("Aprendizaje automático no supervisado K-Means ")

# CARGA DE ARCHIVO CSV
archivo = st.file_uploader("Carga tu archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    # LIMITE DE REGISTROS: 1000 filas y 5 columnas
    MAX_FILAS = 1000
    MAX_COLUMNAS = 5

    if df.shape[0] > MAX_FILAS:
        st.error(f"❌ El archivo tiene {df.shape[0]} filas. El máximo permitido es {MAX_FILAS}.")
        st.stop()

    if df.shape[1] > MAX_COLUMNAS:
        st.error(f"❌ El archivo tiene {df.shape[1]} columnas. El máximo permitido es {MAX_COLUMNAS}.")
        st.stop()

    st.subheader("Vista previa del dataset")
    st.dataframe(df)

    # Seleccionar columnas numéricas
    variables = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    st.subheader("Selecciona dos variables para visualizar")
    var_x = st.selectbox("Eje X", variables)
    var_y = st.selectbox("Eje Y", variables)

    # Escalado
    scaler = StandardScaler()
    X = scaler.fit_transform(df[variables])

    # K-Means
    k = st.slider("Número de clusters", 2, 10, 3)
    modelo = KMeans(n_clusters=k, random_state=42)
    df["cluster"] = modelo.fit_predict(X)

    # Gráfico
    st.subheader("Clustering K-Means (agrupamiento de datos)")

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(df[var_x], df[var_y], c=df["cluster"], cmap="tab10", s=40)

    plt.colorbar(scatter, label="Cluster")
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    ax.set_title("K-Means usando variables reales")

    st.pyplot(fig)