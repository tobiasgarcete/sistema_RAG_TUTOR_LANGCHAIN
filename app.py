import streamlit as st
from rag_core import RAGSystem
from utils import verificar_configuracion
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
def init_rag_system():
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
        st.header(" Gestión de Colecciones")
        
        # Input para el tema/categoría
        tema = st.text_input(
            "Tema/Categoría",
            placeholder="Ej: JavaScript, Python, RAG, Agents...",
            help="Asigna un tema para organizar tus PDFs en colecciones"
        )
        
        # Uploader de archivos PDF
        uploaded_files = st.file_uploader(
            "Selecciona uno o más PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            help="Sube los PDFs sobre los que quieres hacer preguntas"
        )
        
        # Botón para procesar PDFs
        if uploaded_files and tema:
            if st.button("Procesar PDFs", use_container_width=True):
                with st.spinner(f"Procesando documentos para tema '{tema}'..."):
                    try:
                        # Crear o actualizar colección por tema
                        success = rag_system.procesar_pdfs_subidos(uploaded_files, tema)
                        if success:
                            st.success(f" {len(uploaded_files)} PDF(s) agregados a '{tema}'")
                            st.session_state.pdfs_cargados = True
                            st.rerun()
                        else:
                            st.error(" Error al procesar PDFs")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        elif uploaded_files and not tema:
            st.warning(" Ingresa un tema antes de procesar")
        
        st.markdown("---")
        
        # Mostrar colecciones disponibles
        temas_disponibles = rag_system.listar_temas()
        if temas_disponibles:
            st.success(" Colecciones activas")
            
            # Mostrar estadísticas
            stats = rag_system.obtener_estadisticas()
            for tema_nombre in temas_disponibles:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📂 **{tema_nombre}**")
                with col2:
                    if st.button("🗑️", key=f"del_{tema_nombre}", help=f"Eliminar '{tema_nombre}'"):
                        rag_system.limpiar_coleccion(tema_nombre)
                        st.rerun()
            
            # Selector de tema para consultas
            st.markdown("---")
            st.subheader("Filtrar búsqueda")
            tema_seleccionado = st.selectbox(
                "Buscar en:",
                ["Todas las colecciones"] + temas_disponibles,
                help="Selecciona un tema específico o busca en todas"
            )
            st.session_state.tema_seleccionado = None if tema_seleccionado == "Todas las colecciones" else tema_seleccionado
        else:
            st.warning(" No hay colecciones. Sube PDFs para comenzar")
        
        st.markdown("---")
        
        # Botón para limpiar chat
        if st.button("Limpiar Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Botón para limpiar todas las colecciones
        if st.button(" Limpiar Todas las Colecciones", use_container_width=True):
            st.session_state.pdfs_cargados = False
            st.session_state.messages = []
            rag_system.limpiar_coleccion()
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
        # Verificar que haya colecciones cargadas
        if not rag_system.listar_temas():
            with st.chat_message("assistant"):
                st.warning(" Por favor, sube y procesa PDFs primero")
            st.stop()
        
        # Agregar mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Obtener tema seleccionado (si hay)
        tema_consulta = st.session_state.get('tema_seleccionado', None)
        
        # Obtener respuesta del sistema RAG
        with st.chat_message("assistant"):
            with st.spinner("Analizando documentos..."):
                try:
                    respuesta = rag_system.obtener_respuesta_temporal(
                        pregunta=prompt,
                        tema=tema_consulta
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
