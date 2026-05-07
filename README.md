# Multimodal AI Hub

Plataforma inteligente que combina **Chat con PDF** y **Chat con OCR** usando IA de Gemini.

## Funcionalidades

### 📄 Chat con PDF
- Sube documentos PDF y haz preguntas sobre su contenido
- Búsqueda semántica usando embeddings locales
- Respuestas contextuales con Gemini AI

### 🖼️ Chat con OCR
- Extrae texto de imágenes (JPG, PNG)
- Reconocimiento óptico de caracteres con PaddleOCR
- Conversaciones inteligentes sobre el texto extraído

## Tecnologías
- FastAPI (Backend)
- Gemini AI (Modelo de lenguaje)
- PaddleOCR (Reconocimiento óptico)
- Sentence Transformers (Embeddings locales)

## Instalación

```bash
git clone https://github.com/tu-usuario/multimodal-ai-hub.git
cd multimodal-ai-hub
pip install -r requirements.txt
uvicorn main:app --reload
