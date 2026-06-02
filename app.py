import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.markdown("""
    <div style='background-color:#0A1F44;padding:18px;border-radius:8px;margin-bottom:20px'>
        <h2 style='color:white;margin:0;'>🔵 K-Means para analizar datos de un SCADA</h2>
        <p style='color:#D9E1F2;margin:0;font-size:16px;'>
            Convierte esa curiosidad en interacción real. Carga tu archivo SCADA y descubre los modos de operación de tu equipo.
        </p>
    </div>
""", unsafe_allow_html=True)
st.title("Aprendizaje automático no supervisado K-Means ")

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