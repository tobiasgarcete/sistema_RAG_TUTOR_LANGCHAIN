import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma


class RAGSystem:
    """
    Sistema RAG que gestiona múltiples colecciones de documentos
    y permite consultas basadas en contexto recuperado
    """
    
    def __init__(self):
        """Inicializa el sistema RAG con las configuraciones necesarias"""
        # Configuración para Ollama (local)
        self.embeddings = OllamaEmbeddings(
            model="llama3.2",
            base_url="http://localhost:11434"
        )
        self.llm = OllamaLLM(
            model="llama3.2",
            temperature=0.2,  
            num_predict=512,  
            base_url="http://localhost:11434"
        )
        
        # Diccionario para almacenar colecciones por tema
        self.colecciones = {}
        print("[INFO] Sistema RAG inicializado. Listo para procesar PDFs del usuario.")
    
    def procesar_pdfs_subidos(self, uploaded_files, tema: str):
        """
        Procesa PDFs subidos por el usuario y crea una colección por tema
        
        Args:
            uploaded_files: Lista de archivos subidos desde Streamlit
            tema: Nombre del tema/categoría para la colección
        
        Returns:
            bool: True si se procesaron correctamente
        """
        import tempfile
        import shutil
        
        try:
            # Crear directorio temporal
            temp_dir = tempfile.mkdtemp()
            
            # Guardar archivos subidos temporalmente
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Cargar documentos PDF
            loader = DirectoryLoader(
                temp_dir,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            documentos = loader.load()
            
            if not documentos:
                print("[WARN] No se pudieron cargar documentos")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False
            
            # Calcular texto total extraído
            texto_total = ""
            for doc in documentos:
                texto_total += doc.page_content.strip()
            
            # Si hay texto suficiente, usar extracción normal
            if len(texto_total) > 100:  
                print(f"[OK] Texto extraído correctamente: {len(texto_total)} caracteres")
                documentos_validos = [doc for doc in documentos if len(doc.page_content.strip()) > 10]
            else:
                # Si no hay texto, intentar OCR
                print("[WARN] PDF sin texto extraíble detectado")
                print("[OCR] Intentando extraer texto con OCR...")
                
                try:
                    documentos_ocr = self._extraer_texto_ocr(temp_dir)
                    if documentos_ocr:
                        documentos_validos = documentos_ocr
                        print(f"[OK] Texto extraído con OCR de {len(documentos_ocr)} páginas")
                    else:
                        documentos_validos = []
                except Exception as e:
                    print(f"[ERROR] Error en OCR: {str(e)}")
                    documentos_validos = []
            
            if not documentos_validos:
                print("[WARN] No se pudo extraer texto de los PDFs")
                print("[INFO] El PDF puede contener solo imágenes sin texto")
                print("[INFO] Intenta con un PDF que tenga texto seleccionable")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False
            
            print(f"[DOC] Documentos procesados: {len(documentos_validos)} páginas con texto válido")
            
            # Dividir en chunks más grandes para procesar más rápido
            print("[PROC] Dividiendo en chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  
                chunk_overlap=150,  
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len
            )
            chunks = text_splitter.split_documents(documentos_validos)
            
            # Filtrar chunks vacíos o muy cortos
            chunks_validos = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 20]
            
            if not chunks_validos:
                print("[WARN] No se generaron chunks válidos")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False
            
            print(f"[OK] Creados {len(chunks_validos)} chunks")
            print(f"[PROC] Generando embeddings para tema '{tema}'...")
            
            # Crear o actualizar colección por tema
            collection_name = f"tema_{tema.lower().replace(' ', '_')}"
            
            if tema in self.colecciones:
                # Agregar a colección existente
                print(f"[INFO] Agregando documentos a la colección existente '{tema}'")
                self.colecciones[tema].add_documents(chunks_validos)
            else:
                # Crear nueva colección
                self.colecciones[tema] = Chroma.from_documents(
                    documents=chunks_validos,
                    embedding=self.embeddings,
                    collection_name=collection_name
                )
                print(f"[OK] Nueva colección '{tema}' creada")
            
            print(f"[OK] Base de datos vectorial creada exitosamente")
            
            # Limpiar directorio temporal
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            print(f"[OK] Colección '{tema}' lista con {len(chunks_validos)} chunks")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error al procesar PDFs: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def obtener_respuesta_temporal(self, pregunta: str, tema: str = None) -> str:
        """
        Obtiene respuesta basada en la colección de un tema específico
        
        Args:
            pregunta: Pregunta del usuario
            tema: Tema específico a consultar (si es None, busca en todas)
        
        Returns:
            Respuesta generada por el LLM con contexto
        """
        if not self.colecciones:
            return "[WARN] No hay documentos cargados. Por favor, sube PDFs primero."
        
        # Si se especifica un tema, buscar solo en esa colección
        if tema and tema in self.colecciones:
            docs = self.colecciones[tema].similarity_search(pregunta, k=8)
            tema_info = f" en la colección '{tema}'"
        else:
            # Buscar en todas las colecciones
            docs = []
            for nombre_tema, coleccion in self.colecciones.items():
                docs.extend(coleccion.similarity_search(pregunta, k=4))
            tema_info = " en todas las colecciones"
        
        # Crear contexto desde los documentos
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Crear prompt optimizado que funcione bien con contenido multiidioma
        prompt_text = f"""Eres un asistente experto. Responde la pregunta del usuario basándote en la información de los documentos.

IMPORTANTE:
- Usa SOLO la información del contexto proporcionado
- Si el contexto está en inglés, traduce la información relevante al español
- Sé específico y detallado
- Si no encuentras la respuesta en el contexto, dilo claramente

CONTEXTO DE LOS DOCUMENTOS:
{context}

PREGUNTA: {pregunta}

RESPUESTA:"""
        
        # Generar respuesta con el LLM
        respuesta = self.llm.invoke(prompt_text)
        
        # Agregar información de fuentes y tema consultado
        if docs:
            respuesta += f"\n\n---\n** Búsqueda realizada{tema_info}**\n"
            respuesta += "\n**Fuentes consultadas:**\n"
            fuentes_unicas = set()
            for doc in docs:
                fuente = doc.metadata.get("source", "Desconocido")
                pagina = doc.metadata.get("page", "N/A")
                fuente_info = f"- {os.path.basename(fuente)} (Página {pagina})"
                fuentes_unicas.add(fuente_info)
            
            for fuente in sorted(fuentes_unicas):
                respuesta += f"\n{fuente}"
        
        return respuesta
    
    def listar_temas(self) -> List[str]:
        """Retorna la lista de temas disponibles"""
        return list(self.colecciones.keys())
    
    def obtener_estadisticas(self) -> Dict:
        """Retorna estadísticas de las colecciones"""
        stats = {}
        for tema, coleccion in self.colecciones.items():
            # Contar documentos en la colección
            stats[tema] = {
                "documentos": len(coleccion.get())
            }
        return stats
    
    def _extraer_texto_ocr(self, pdf_dir):
        """
        Extrae texto de PDFs usando OCR (para PDFs con imágenes)
        
        Args:
            pdf_dir: Directorio con los PDFs
            
        Returns:
            Lista de documentos con texto extraído
        """
        try:
            from pdf2image import convert_from_path
            import pytesseract
            from langchain_core.documents import Document
            
            # Configurar ruta de Tesseract
            tesseract_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\IPF-2025\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
            ]
            
            for path in tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"[OK] Usando Tesseract: {path}")
                    break
            
            # Configurar ruta de poppler si existe localmente
            poppler_path = None
            if os.path.exists("./poppler-24.08.0/Library/bin"):
                poppler_path = "./poppler-24.08.0/Library/bin"
                print(f"[OK] Usando Poppler local: {poppler_path}")
            
            documentos_ocr = []
            pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                
                # Convertir PDF a imágenes
                print(f"[DOC] Procesando {pdf_file} con OCR...")
                images = convert_from_path(
                    pdf_path, 
                    dpi=200,
                    poppler_path=poppler_path
                )
                
                for i, image in enumerate(images):
                    # Extraer texto con OCR
                    texto = pytesseract.image_to_string(image, lang='spa+eng')
                    
                    if len(texto.strip()) > 50:
                        doc = Document(
                            page_content=texto.strip(),
                            metadata={
                                "source": pdf_file,
                                "page": i,
                                "extraction_method": "OCR"
                            }
                        )
                        documentos_ocr.append(doc)
                        print(f"  [OK] Página {i+1}: {len(texto)} caracteres")
            
            return documentos_ocr
            
        except ImportError:
            print("[WARN] OCR no disponible. Instala: pip install pytesseract pdf2image")
            print("[WARN] También necesitas Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
            return []
        except Exception as e:
            print(f"[ERROR] Error en OCR: {str(e)}")
            return []
    
    def limpiar_coleccion(self, tema: str = None):
        """
        Limpia una colección específica o todas
        
        Args:
            tema: Tema a limpiar (si es None, limpia todas)
        """
        if tema:
            if tema in self.colecciones:
                del self.colecciones[tema]
                print(f"[OK] Colección '{tema}' eliminada")
        else:
            self.colecciones = {}
            print("[OK] Todas las colecciones eliminadas")

