import os
import streamlit as st

from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# ====================================
# API KEY
# ====================================

os.environ["OPENAI_API_KEY"] = "TU_API_KEY"

# ====================================
# CONFIGURACIÓN APP
# ====================================

st.set_page_config(
    page_title="Industrial Reliability AI",
    layout="wide"
)

st.title("Industrial Reliability AI")

st.markdown("""
Asistente especializado en:

- Confiabilidad
- Operaciones
- KPIs SMRP
- Mantenimiento
- RCA
- Análisis técnico documental
""")

st.divider()

# ====================================
# CARGA PDF
# ====================================

uploaded_file = st.file_uploader(
    "Carga un datasheet o PDF técnico",
    type=["pdf"]
)

# ====================================
# PROCESAMIENTO
# ====================================

if uploaded_file:

    with open("temp.pdf","wb") as f:
        f.write(uploaded_file.read())

    st.info(
        "Procesando y vectorizando documento..."
    )

    documents = SimpleDirectoryReader(
        input_files=["temp.pdf"]
    ).load_data()

    llm = OpenAI(
        model="gpt-4o-mini"
    )

    index = VectorStoreIndex.from_documents(
        documents
    )

    query_engine = index.as_query_engine(
        llm=llm
    )

    st.success(
        "Documento procesado correctamente"
    )

    # ====================================
    # PROMPTS SUGERIDOS
    # ====================================

    st.subheader(
        "Preguntas sugeridas"
    )

    col1,col2,col3=st.columns(3)

    with col1:

        st.info(
            "¿Qué componentes son críticos?"
        )

        st.info(
            "¿Qué riesgos operacionales existen?"
        )

    with col2:

        st.info(
            "¿Qué afecta disponibilidad?"
        )

        st.info(
            "¿Qué modos de falla aparecen?"
        )

    with col3:

        st.info(
            "Resume mantenimiento recomendado"
        )

        st.info(
            "¿Qué impacta MTBF?"
        )

    st.divider()

    # ====================================
    # PREGUNTA
    # ====================================

    pregunta = st.text_input(
        "Haz una pregunta técnica"
    )

    if pregunta:

        prompt = f"""

Actúa como un experto senior en:

- confiabilidad industrial
- KPIs SMRP
- mantenimiento
- RCA
- disponibilidad
- MTBF
- MTTR
- análisis de criticidad
- operaciones industriales

Analiza el documento cargado.

Pregunta:

{pregunta}

Responde:

1. Hallazgos técnicos
2. Riesgos
3. Impacto operacional
4. Recomendaciones
5. Conclusión

"""

        respuesta = query_engine.query(
            prompt
        )

        st.subheader(
            "Respuesta"
        )

        st.write(respuesta)

st.divider()

st.caption(
"Industrial Reliability AI | RAG especializado"
)