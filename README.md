# Sistema RAG para Consulta de Documentación Técnica

Sistema de Recuperación Aumentada por Generación (RAG) implementado con LangChain y Ollama que permite realizar consultas inteligentes sobre documentos PDF personalizados.

## Descripción

Este proyecto implementa un asistente conversacional que utiliza técnicas de RAG para responder preguntas basadas en documentos PDF cargados por el usuario. El sistema procesa documentos, los vectoriza y permite realizar consultas en lenguaje natural, proporcionando respuestas contextualizadas con referencias a las fuentes.

### Características Principales

- **Carga dinámica de PDFs**: Los usuarios pueden subir sus propios documentos
- **Procesamiento OCR**: Soporte para PDFs con imágenes mediante Tesseract
- **Búsqueda semántica**: Vectorización con embeddings de Ollama
- **Respuestas contextualizadas**: Basadas únicamente en el contenido de los documentos
- **Citación de fuentes**: Referencias automáticas a páginas específicas
- **Interfaz web intuitiva**: Desarrollada con Streamlit
- **100% Local**: No requiere servicios en la nube ni API keys

## Gif de Demostración
![Demo del Sistema RAG](https://raw.githubusercontent.com/tobiasgarcete/sistema_RAG_TUTOR_LANGCHAIN/main/gif_RAG.gif)

## Stack Tecnológico

- **Python 3.10+**: Lenguaje de programación principal
- **LangChain**: Framework para aplicaciones con LLMs
- **Ollama**: Servidor de modelos de lenguaje local (llama3.2)
- **ChromaDB**: Base de datos vectorial para embeddings
- **Streamlit**: Framework para la interfaz web
- **PyPDF**: Extracción de texto de PDFs
- **Tesseract OCR**: Reconocimiento óptico de caracteres
- **Poppler**: Conversión de PDF a imágenes

## Arquitectura del Sistema

```
Usuario (Web) 
    ↓
Streamlit (UI)
    ↓
RAG Core (Lógica)
    ↓
├─→ Ollama (LLM + Embeddings)
├─→ ChromaDB (Vector Store)
├─→ PyPDF (Extracción)
└─→ Tesseract OCR (Imágenes)
```

## Requisitos Previos

- Python 3.10 o superior
- Ollama instalado y ejecutándose localmente
- Tesseract OCR (para PDFs con imágenes)
- 4GB RAM mínimo (8GB recomendado)

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/sistema-rag-docs.git
cd sistema-rag-docs
```

### 2. Crear entorno virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar Ollama

Descarga e instala Ollama desde [ollama.ai](https://ollama.ai/)

```bash
# Descargar el modelo
ollama pull llama3.2
```

### 5. Instalar Tesseract OCR (opcional, para PDFs con imágenes)

**Windows:**
```bash
winget install UB-Mannheim.TesseractOCR
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-spa
```

**Mac:**
```bash
brew install tesseract
```

## Uso

### Iniciar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

### Pasos para usar el sistema

1. **Subir documentos PDF**
   - Haz clic en "Sube tus PDFs" en el panel lateral
   - Selecciona uno o más archivos PDF
   - Click en "Procesar PDFs"

2. **Esperar el procesamiento**
   - El sistema extraerá el texto
   - Si es necesario, aplicará OCR automáticamente
   - Generará embeddings y los almacenará

3. **Realizar preguntas**
   - Escribe tu pregunta en el chat
   - El sistema buscará información relevante
   - Recibirás una respuesta con fuentes citadas

4. **Gestionar sesión**
   - "Limpiar Chat": Borra el historial de conversación
   - "Limpiar PDFs": Elimina documentos y permite cargar nuevos

### Ejemplos de uso

```
Pregunta: ¿Qué temas principales se tratan en este documento?
Pregunta: ¿Cómo se implementa [concepto específico]?
Pregunta: ¿Qué dice el documento sobre [tema]?
```

## Estructura del Proyecto

```
sistema-rag-docs/
│
├── app.py                  # Aplicación principal Streamlit
├── rag_core.py            # Lógica del sistema RAG
├── utils.py               # Utilidades y validaciones
├── requirements.txt       # Dependencias Python
├── README.md             # Este archivo
├── LICENSE               # Licencia MIT
│
├── .streamlit/           # Configuración de Streamlit
│   └── config.toml
│
├── .gitignore            # Archivos ignorados por Git
│
├── documentos/           # PDFs de ejemplo (opcional)
│   ├── general/
│   ├── rag/
│   ├── agents/
│   ├── chains/
│   └── memory/
│
└── poppler-24.08.0/      # Binarios de Poppler (Windows)
```

## Configuración

### Configuración de Streamlit

Ubicación: `.streamlit/config.toml`

```toml
[server]
maxUploadSize = 1000  # 1GB de límite de carga

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
```

## Limitaciones Conocidas

- **Tiempo de procesamiento**: PDFs grandes (>50MB) pueden tardar varios minutos
- **Memoria**: Se recomienda 8GB RAM para documentos extensos
- **Idioma**: Optimizado para español e inglés
- **OCR**: La calidad depende de la resolución de las imágenes del PDF

## Troubleshooting

### Error: "Ollama not found"
```bash
# Verifica que Ollama esté corriendo
ollama list

# Si no está activo, inícialo
ollama serve
```

### Error: "Tesseract not found"
- Instala Tesseract OCR según tu sistema operativo
- Verifica que esté en el PATH del sistema

### PDFs no procesan correctamente
- Verifica que el PDF no esté protegido con contraseña
- Para PDFs escaneados, asegúrate de tener Tesseract instalado
- Reduce el tamaño del PDF si supera 200MB

## Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

## Agradecimientos

- [LangChain](https://www.langchain.com/) - Framework para aplicaciones LLM
- [Ollama](https://ollama.ai/) - Servidor de modelos locales
- [Streamlit](https://streamlit.io/) - Framework de UI
- [ChromaDB](https://www.trychroma.com/) - Base de datos vectorial

---

**Nota**: Este proyecto es con fines educativos y de demostración de capacidades de RAG.
