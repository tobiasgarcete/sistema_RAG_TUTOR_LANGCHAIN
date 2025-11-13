"""
Utilidades auxiliares para el sistema RAG
"""

import os
from typing import List
import streamlit as st


def verificar_configuracion() -> bool:
    """
    Verifica que la configuración del sistema esté completa
    
    Returns:
        True si la configuración es válida
    """
    # Verificar que Ollama esté configurado
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    st.info(f"🤖 Usando modelo local: **{ollama_model}** con Ollama")
    
    # Verificar carpetas de documentos
    carpetas_requeridas = [
        "./documentos/general",
        "./documentos/rag",
        "./documentos/agents",
        "./documentos/chains",
        "./documentos/memory"
    ]
    
    carpetas_faltantes = [c for c in carpetas_requeridas if not os.path.exists(c)]
    
    if carpetas_faltantes:
        st.warning(f"""
        ⚠️ **Advertencia**
        
        Faltan carpetas de documentos:
        {', '.join(carpetas_faltantes)}
        
        El sistema creará estas carpetas automáticamente, pero necesitas agregar PDFs.
        """)
        # Crear carpetas faltantes
        for carpeta in carpetas_faltantes:
            os.makedirs(carpeta, exist_ok=True)
    
    return True


def contar_documentos() -> dict:
    """
    Cuenta los documentos PDF en cada colección
    
    Returns:
        Diccionario con el conteo de documentos por tema
    """
    temas = ["general", "rag", "agents", "chains", "memory"]
    conteo = {}
    
    for tema in temas:
        carpeta = f"./documentos/{tema}"
        if os.path.exists(carpeta):
            pdfs = [f for f in os.listdir(carpeta) if f.endswith('.pdf')]
            conteo[tema.capitalize()] = len(pdfs)
        else:
            conteo[tema.capitalize()] = 0
    
    return conteo


def formatear_fuentes(source_documents: List) -> str:
    """
    Formatea los documentos fuente para mostrar en la UI
    
    Args:
        source_documents: Lista de documentos fuente
        
    Returns:
        String formateado con las fuentes
    """
    if not source_documents:
        return ""
    
    fuentes_texto = "\n\n---\n**📚 Fuentes consultadas:**\n"
    fuentes_unicas = set()
    
    for doc in source_documents:
        fuente = doc.metadata.get("source", "Desconocido")
        pagina = doc.metadata.get("page", "N/A")
        nombre_archivo = os.path.basename(fuente)
        fuente_info = f"- 📄 {nombre_archivo} (Página {pagina})"
        fuentes_unicas.add(fuente_info)
    
    for fuente in sorted(fuentes_unicas):
        fuentes_texto += f"\n{fuente}"
    
    return fuentes_texto


def mostrar_estadisticas():
    """Muestra estadísticas del sistema en el sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Estadísticas")
    
    conteo = contar_documentos()
    total = sum(conteo.values())
    
    st.sidebar.metric("Total de Documentos", total)
    
    with st.sidebar.expander("Ver detalle por tema"):
        for tema, cantidad in conteo.items():
            if cantidad > 0:
                st.write(f"✅ **{tema}**: {cantidad} documentos")
            else:
                st.write(f"⚠️ **{tema}**: Sin documentos")
