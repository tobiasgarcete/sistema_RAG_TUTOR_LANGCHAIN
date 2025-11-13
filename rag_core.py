"""
Núcleo del sistema RAG para el Tutor de Documentación Técnica
Gestiona la ingesta de documentos, vectorización y recuperación de contexto
"""

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
            temperature=0.2,  # Reducido para respuestas más rápidas y consistentes
            num_predict=512,  # Limitar tokens de respuesta
            base_url="http://localhost:11434"
        )
        
        # Directorio para almacenar la base de datos vectorial
        self.persist_directory = "./chroma_db"
        
        # Diccionario para almacenar las colecciones
        self.colecciones = {}
        
        # Cargar o crear colecciones
        self._inicializar_colecciones()
        
        # Colección temporal para PDFs subidos por usuario
        self.coleccion_temporal = None
    

    
    def _inicializar_colecciones(self):
        """Inicializa las colecciones de documentos por tema"""
        # No inicializar colecciones pre-cargadas
        # El usuario subirá sus propios PDFs dinámicamente
        print("[INFO] Sistema RAG inicializado. Listo para procesar PDFs del usuario.")
        pass
    
    def _crear_o_cargar_coleccion(self, collection_name: str, docs_path: str):
        """Crea o carga una colección de documentos"""
        persist_path = os.path.join(self.persist_directory, collection_name)
        
        # Verificar si ya existe la colección
        if os.path.exists(persist_path) and os.listdir(persist_path):
            print(f"[INFO] Cargando colección existente: {collection_name}")
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_path
            )
        else:
            print(f"[LOAD] Creando nueva colección: {collection_name}")
            # Cargar documentos PDF del directorio
            loader = DirectoryLoader(
                docs_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            documentos = loader.load()
            
            if not documentos:
                print(f"[WARN] No se pudieron cargar documentos de {docs_path}")
                return None
            
            # Dividir documentos en chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documentos)
            
            print(f"   [DOC] Procesados {len(documentos)} documentos en {len(chunks)} chunks")
            
            # Verificar que tengamos chunks válidos
            if not chunks or len(chunks) == 0:
                print(f"   [WARN] No se pudieron extraer chunks de los documentos")
                print(f"   [INFO] Verifica que los PDFs contengan texto legible (no imágenes)")
                return None
            
            # Crear vectorstore
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=persist_path
            )
            
            print(f"   [OK] Colección creada y persistida")
        
        return vectorstore
    
    def obtener_respuesta(self, pregunta: str, coleccion: str = "General") -> str:
        """
        Obtiene una respuesta a la pregunta usando RAG
        
        Args:
            pregunta: La pregunta del usuario
            coleccion: El tema/colección a consultar
            
        Returns:
            Respuesta generada con contexto y fuentes citadas
        """
        # Verificar que la colección exista
        if coleccion not in self.colecciones:
            return f"""[ERROR] **Error**: No se encontró la colección para el tema "{coleccion}".
            
Por favor:
1. Crea la carpeta: `./documentos/{coleccion.lower()}/`
2. Añade archivos PDF de la documentación de LangChain
3. Reinicia la aplicación

**Colecciones disponibles**: {', '.join(self.colecciones.keys())}"""
        
        # Obtener el vectorstore de la colección
        vectorstore = self.colecciones[coleccion]
        
        # Obtener retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Buscar documentos relevantes
        docs = retriever.invoke(pregunta)
        
        # Crear contexto desde los documentos
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Crear prompt
        prompt_text = f"""Eres un tutor experto en LangChain. Tu trabajo es ayudar a programadores 
a entender y usar LangChain respondiendo preguntas basadas ÚNICAMENTE en la documentación oficial.

REGLAS IMPORTANTES:
1. Solo responde preguntas relacionadas con LangChain
2. Basa tus respuestas ÚNICAMENTE en el contexto proporcionado
3. Si la información no está en el contexto, di claramente que no tienes esa información
4. Cita las fuentes cuando sea posible (menciona la página o sección)
5. Proporciona ejemplos de código cuando sea relevante
6. Sé claro, conciso y educativo

Contexto de la documentación:
{context}

Pregunta del usuario: {pregunta}

Respuesta detallada:"""
        
        # Generar respuesta con el LLM
        respuesta = self.llm.invoke(prompt_text)
        
        # Agregar información de fuentes
        if docs:
            respuesta += "\n\n---\n**Fuentes consultadas: Fuentes consultadas:**\n"
            fuentes_unicas = set()
            for doc in docs:
                fuente = doc.metadata.get("source", "Desconocido")
                pagina = doc.metadata.get("page", "N/A")
                fuente_info = f"- {os.path.basename(fuente)} (Página {pagina})"
                fuentes_unicas.add(fuente_info)
            
            for fuente in fuentes_unicas:
                respuesta += f"\n{fuente}"
        
        return respuesta
    
    def ingestar_documentos(self, docs_path: str, coleccion: str):
        """
        Ingesta nuevos documentos en una colección existente
        
        Args:
            docs_path: Ruta a los documentos PDF
            coleccion: Nombre de la colección
        """
        if coleccion not in self.colecciones:
            raise ValueError(f"La colección {coleccion} no existe")
        
        # Cargar nuevos documentos
        loader = DirectoryLoader(
            docs_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documentos = loader.load()
        
        # Dividir en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documentos)
        
        # Agregar a la colección existente
        vectorstore = self.colecciones[coleccion]
        vectorstore.add_documents(chunks)
        
        print(f"[OK] Ingesta completada: {len(documentos)} documentos, {len(chunks)} chunks")
    
    def procesar_pdfs_subidos(self, uploaded_files):
        """
        Procesa PDFs subidos por el usuario y crea una colección temporal
        
        Args:
            uploaded_files: Lista de archivos subidos desde Streamlit
        
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
            if len(texto_total) > 100:  # Al menos 100 caracteres en total
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
                chunk_size=1500,  # Aumentado de 1000 a 1500
                chunk_overlap=150,  # Reducido de 200 a 150
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
            print(f"[PROC] Generando embeddings (esto puede tardar con documentos grandes)...")
            
            # Crear vectorstore temporal (en memoria)
            self.coleccion_temporal = Chroma.from_documents(
                documents=chunks_validos,
                embedding=self.embeddings,
                collection_name="temp_user_uploads"
            )
            
            print(f"[OK] Base de datos vectorial creada exitosamente")
            
            # Limpiar directorio temporal
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            print("[OK] Colección temporal creada")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error al procesar PDFs: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def obtener_respuesta_temporal(self, pregunta: str) -> str:
        """
        Obtiene respuesta basada en la colección temporal de PDFs del usuario
        
        Args:
            pregunta: Pregunta del usuario
        
        Returns:
            Respuesta generada por el LLM con contexto
        """
        if not self.coleccion_temporal:
            return "[WARN] No hay documentos cargados. Por favor, sube PDFs primero."
        
        # Buscar documentos relevantes usando búsqueda de similitud
        # Aumentamos k para obtener más contexto
        docs = self.coleccion_temporal.similarity_search(pregunta, k=8)
        
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
        
        # Agregar información de fuentes
        if docs:
            respuesta += "\n\n---\n**Fuentes consultadas: Fuentes consultadas:**\n"
            fuentes_unicas = set()
            for doc in docs:
                fuente = doc.metadata.get("source", "Desconocido")
                pagina = doc.metadata.get("page", "N/A")
                fuente_info = f"- {os.path.basename(fuente)} (Página {pagina})"
                fuentes_unicas.add(fuente_info)
            
            for fuente in fuentes_unicas:
                respuesta += f"\n{fuente}"
        
        return respuesta
    
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
    
    def limpiar_coleccion_temporal(self):
        """Limpia la colección temporal de PDFs del usuario"""
        self.coleccion_temporal = None
        print("🧹 Colección temporal limpiada")

