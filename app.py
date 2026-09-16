import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Mi Asistente Drive Pro", page_icon="🚀")
st.title("🚀 Chatbot con mi Google Drive (Gemini Pro)")

try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except:
    st.warning("⚠️ Falta configurar la API Key de Gemini en los secrets de Streamlit.")

@st.cache_resource
def procesar_drive(folder_id):
    st.info("Leyendo documentos de Google Drive... 🔄 (Esto puede tardar un poquito)")
    try:
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            recursive=False 
        )
        docs = loader.load()
        if not docs:
            st.error("No se encontraron documentos en esa carpeta o no hay permisos.")
            return None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        textos_divididos = text_splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        base_de_datos = FAISS.from_documents(textos_divididos, embeddings)
        st.success(f"¡Listo! Se procesaron {len(docs)} documentos. ✅")
        return base_de_datos
    except Exception as e:
        st.error(f"Error al conectar con Drive: {e}")
        return None

with st.sidebar:
    st.header("Configuración ⚙️")
    st.markdown("Ingresá el ID de la carpeta de tu Google Drive que contiene los archivos.")
    carpeta_id = st.text_input("ID de la carpeta:", placeholder="Ej: 1A2b3C4d5E6f...")
    if st.button("Cargar Documentos"):
        if carpeta_id:
            st.session_state.vectorstore = procesar_drive(carpeta_id)
        else:
            st.error("Por favor, ingresá un ID válido.")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])

pregunta = st.chat_input("Preguntale algo a tus documentos de Drive...")

if pregunta:
    with st.chat_message("user"):
        st.markdown(pregunta)
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    
    with st.chat_message("assistant"):
        if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever()
            )
            with st.spinner("Gemini Pro está analizando tus documentos..."):
                respuesta = qa_chain.run(pregunta)
                st.markdown(respuesta)
                st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
        else:
            st.warning("⚠️ Primero tenés que cargar los documentos de Drive desde el menú de la izquierda.")
