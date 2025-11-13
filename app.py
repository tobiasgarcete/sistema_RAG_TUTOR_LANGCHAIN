"""
Tutor de Documentación Técnica de LangChain
Sistema RAG que permite consultar la documentación de LangChain por temas específicos
"""

import streamlit as st
from rag_core import RAGSystem
from utils import verificar_configuracion, mostrar_estadisticas
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Tutor LangChain",
    page_icon="",
    layout="wide"
)

# Inicializar el sistema RAG
@st.cache_resource(show_spinner="Inicializando sistema RAG...")
def init_rag_system(_force_reload=None):
    """Inicializa el sistema RAG con las colecciones de documentación"""
    return RAGSystem()

def main():
    st.title("Asistente RAG con PDFs - LangChain")
    st.markdown("---")
    
    # Verificar configuración antes de continuar
    if not verificar_configuracion():
        st.stop()
    
    # Inicializar sistema RAG
    rag_system = init_rag_system()
    
    # Sidebar para carga de documentos
    with st.sidebar:
        st.header("Sube tus PDFs")
        
        # Uploader de archivos PDF
        uploaded_files = st.file_uploader(
            "Selecciona uno o más PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            help="Sube los PDFs sobre los que quieres hacer preguntas"
        )
        
        # Botón para procesar PDFs
        if uploaded_files:
            if st.button("Procesar PDFs", use_container_width=True):
                with st.spinner("Procesando documentos..."):
                    try:
                        # Crear colección temporal con los PDFs subidos
                        success = rag_system.procesar_pdfs_subidos(uploaded_files)
                        if success:
                            st.success(f"{len(uploaded_files)} PDF(s) procesado(s) exitosamente")
                            st.session_state.pdfs_cargados = True
                        else:
                            st.error("Error al procesar PDFs")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        
        # Estado de los PDFs
        if st.session_state.get('pdfs_cargados', False):
            st.success("PDFs cargados y listos")
            if uploaded_files:
                st.info(f"{len(uploaded_files)} documento(s) activo(s)")
        else:
            st.warning("Sube PDFs para comenzar")
        
        st.markdown("---")
        
        # Botón para limpiar chat
        if st.button("Limpiar Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Botón para limpiar PDFs y empezar de nuevo
        if st.button("Limpiar PDFs", use_container_width=True):
            st.session_state.pdfs_cargados = False
            st.session_state.messages = []
            rag_system.limpiar_coleccion_temporal()
            st.rerun()
    
    # Inicializar el historial de mensajes
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Inicializar estado de PDFs
    if "pdfs_cargados" not in st.session_state:
        st.session_state.pdfs_cargados = False
    
    # Mostrar mensajes del historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input del usuario
    if prompt := st.chat_input("Pregunta sobre tus documentos..."):
        # Verificar que haya PDFs cargados
        if not st.session_state.get('pdfs_cargados', False):
            with st.chat_message("assistant"):
                st.warning("Por favor, sube y procesa PDFs primero")
            st.stop()
        
        # Agregar mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Obtener respuesta del sistema RAG
        with st.chat_message("assistant"):
            with st.spinner("Analizando documentos..."):
                try:
                    respuesta = rag_system.obtener_respuesta_temporal(
                        pregunta=prompt
                    )
                    st.markdown(respuesta)
                    
                    # Agregar respuesta al historial
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": respuesta
                    })
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()
