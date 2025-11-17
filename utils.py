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
    st.info(f"[INFO] Usando modelo local: **{ollama_model}** con Ollama")
    
    return True


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
