# Documentación Técnica — AI Multimodal Hub
**Visual OCR & Chat con PDF · Versión 1.0**  
*Desplegado en: [antoniomenor01-visual-ocr-context-chat.hf.space](https://antoniomenor01-visual-ocr-context-chat.hf.space)*

---

## Índice

1. [Descripción general del proyecto](#1-descripción-general-del-proyecto)
2. [Arquitectura del sistema](#2-arquitectura-del-sistema)
3. [Decisiones tecnológicas](#3-decisiones-tecnológicas)
4. [Módulo OCR — Decisiones y detalles](#4-módulo-ocr--decisiones-y-detalles)
5. [Módulo PDF — Decisiones y detalles](#5-módulo-pdf--decisiones-y-detalles)
6. [Integración con Gemini AI](#6-integración-con-gemini-ai)
7. [Interfaz de usuario](#7-interfaz-de-usuario)
8. [Problemas encontrados durante el desarrollo](#8-problemas-encontrados-durante-el-desarrollo)
9. [Imágenes que funcionan bien y que no funcionan](#9-imágenes-que-funcionan-bien-y-que-no-funcionan)
10. [Despliegue en Hugging Face Spaces](#10-despliegue-en-hugging-face-spaces)
11. [Conclusiones y trabajo futuro](#11-conclusiones-y-trabajo-futuro)

---

## 1. Descripción general del proyecto

**AI Multimodal Hub** es una plataforma web que permite a los usuarios interactuar con documentos e imágenes mediante inteligencia artificial conversacional. Combina dos funcionalidades principales:

- **Chat con PDF**: el usuario sube un documento PDF y puede hacerle preguntas en lenguaje natural. El sistema recupera los fragmentos más relevantes mediante búsqueda semántica y genera una respuesta contextualizada con Gemini AI.
- **Chat con OCR**: el usuario sube una imagen (JPG o PNG), el sistema extrae el texto mediante reconocimiento óptico de caracteres (PaddleOCR) y habilita un chat inteligente sobre ese texto.

La aplicación está desplegada en Hugging Face Spaces y es accesible desde cualquier navegador sin instalación local.

---

## 2. Arquitectura del sistema

```
┌─────────────────────────────────────────────────┐
│                  FRONTEND (HTML/CSS/JS)         │
│   index.html · pdf.html · ocr.html · styles.css │
└───────────────────────┬─────────────────────────┘
                        │ HTTP / Fetch API
┌───────────────────────▼─────────────────────────┐
│              BACKEND — FastAPI (Python)         │
│                                                 │
│   /api/pdf/upload   →  pdf_logic.py             │
│   /api/pdf/chat     →  pdf_logic.py             │
│   /api/ocr/upload   →  ocr_logic.py             │
│   /api/ocr/chat     →  ocr_logic.py             │
│                                                 │
│   Archivos estáticos servidos desde /static     │
└────────────┬──────────────────────┬─────────────┘
             │                      │
┌────────────▼──────┐   ┌──────────▼──────────────┐
│  PaddleOCR (OCR)  │   │  SentenceTransformers   │
│  OpenCV (prepro.) │   │  (Embeddings locales)   │
└───────────────────┘   └──────────┬──────────────┘
                                   │
                        ┌──────────▼──────────────┐
                        │  Gemini AI (Google)     │
                        │  gemini-2.5-flash       │
                        └─────────────────────────┘
```

El backend sigue el patrón de **routers de FastAPI**: cada funcionalidad (`pdf_logic`, `ocr_logic`) es un módulo independiente que se registra en la aplicación principal (`main.py`) bajo su prefijo de URL correspondiente.

---

## 3. Decisiones tecnológicas

### 3.1 Framework backend: FastAPI

Se eligió **FastAPI** como framework web por varias razones:

- Tipado estático con Pydantic para validación automática de datos de entrada y salida.
- Rendimiento asíncrono nativo, útil para las operaciones de I/O con el API de Gemini.
- Generación automática de documentación OpenAPI en `/docs`.
- Soporte nativo para `UploadFile`, lo que simplifica la recepción de archivos del frontend.

### 3.2 Modelo de lenguaje: Gemini 2.5 Flash

Se utilizó **`gemini-2.5-flash`** de Google como LLM principal porque:

- Ofrece un buen equilibrio entre velocidad y calidad de respuesta.
- Tiene un contexto amplio, útil para procesar fragmentos de documentos largos.
- El SDK oficial de Python (`google-genai`) facilita la integración.

Se implementó además un **sistema de fallback** en el módulo OCR: si `gemini-2.5-flash` falla (por límites de tasa o error transitorio), el sistema reintenta hasta 2 veces antes de pasar a `gemini-3.1-flash-lite-preview`. Esto mejora la resiliencia en producción.

### 3.3 Embeddings locales: Sentence Transformers

Para el módulo PDF se necesitaba una forma de encontrar los fragmentos más relevantes del documento para cada pregunta. Se eligió **`all-MiniLM-L6-v2`** de Sentence Transformers porque:

- Es un modelo ligero (~80 MB) que funciona bien sin GPU.
- Genera embeddings semánticamente coherentes en inglés y español.
- No requiere llamadas externas, lo que reduce latencia y coste.

La similitud entre el embedding de la pregunta y los embeddings de los fragmentos se calcula mediante **similitud coseno**, con un umbral mínimo de 0.20 para filtrar fragmentos irrelevantes.

### 3.4 Frontend: HTML/CSS/JS puro

Se descartó el uso de frameworks de frontend como React o Vue para mantener la aplicación simple y sin proceso de compilación. Los archivos estáticos (`index.html`, `pdf.html`, `ocr.html`, `styles.css`) son servidos directamente por FastAPI mediante `StaticFiles`.

---

## 4. Módulo OCR — Decisiones y detalles

### 4.1 Motor OCR: PaddleOCR

Se evaluaron varias alternativas antes de elegir **PaddleOCR**:

| Motor       | Ventajas                            | Inconvenientes                             |
|-------------|-------------------------------------|--------------------------------------------|
| Tesseract   | Maduro, licencia libre              | Peor rendimiento en texto manuscrito       |
| EasyOCR     | Fácil de usar, buen soporte idiomas | Más lento, mayor consumo de RAM            |
| **PaddleOCR** | **Alta precisión, rápido, soporte español** | **Dependencia pesada (PaddlePaddle)** |

PaddleOCR se inicializa con `use_angle_cls=True` y `lang="es"`, lo que activa la clasificación de orientación del texto. Esto mejora el reconocimiento en imágenes con texto inclinado o rotado.

### 4.2 Preprocesado de imagen con OpenCV

Antes de pasar la imagen al OCR se aplica una conversión a escala de grises (`cv2.cvtColor`). Esta decisión busca:

- Reducir el ruido introducido por colores de fondo.
- Mejorar el contraste entre el texto y el fondo.
- Acelerar el procesamiento, ya que PaddleOCR opera internamente sobre canales de luminancia.

### 4.3 Flujo completo del módulo OCR

```
Usuario sube imagen (JPG/PNG)
         ↓
Validación de content-type
         ↓
Lectura de bytes → numpy array (cv2.imdecode)
         ↓
Preprocesado: conversión a escala de grises
         ↓
PaddleOCR extrae líneas de texto
         ↓
Concatenación de líneas en un string plano
         ↓
Texto devuelto al frontend → habilitación del chat
         ↓
Usuario escribe pregunta → se envía texto + historial + pregunta a Gemini
         ↓
Respuesta mostrada en el chat
```

---

## 5. Módulo PDF — Decisiones y detalles

### 5.1 Extracción de texto: pypdf

Se usa **pypdf** para extraer el texto de cada página del PDF. El procesamiento está limitado a **50 páginas máximo** para evitar saturar la memoria en el servidor de Hugging Face Spaces.

### 5.2 Fragmentación (chunking)

El texto extraído se divide en fragmentos (*chunks*) de **300 palabras** con un solapamiento de **50 palabras**. El solapamiento evita que información relevante quede cortada en la frontera entre dos fragmentos.

```python
def dividir_texto(texto, chunk_size=300, overlap=50):
    palabras = texto.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(palabras), step):
        chunk = palabras[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks
```

### 5.3 Base de datos vectorial en memoria

Se eligió **un diccionario Python en memoria** (`VECTOR_DB`) en lugar de una base de datos vectorial como Chroma o FAISS. Esta decisión responde a:

- Simplicidad de despliegue (sin dependencias adicionales).
- Documentos de tamaño moderado (máx. 50 páginas) que caben cómodamente en RAM.
- Limpiar automáticamente documentos más antiguos cuando se superan 20 documentos activos.

La limitación es que **los vectores se pierden al reiniciar el servidor**, lo que es aceptable en un entorno de demostración.

### 5.4 Recuperación de contexto

Para cada pregunta se calculan las similitudes coseno entre el embedding de la pregunta y todos los fragmentos. Se seleccionan los **10 fragmentos más similares** que superen el umbral de 0.20. Si ningún fragmento supera el umbral, se usan los 10 más similares de todos modos para garantizar una respuesta.

---

## 6. Integración con Gemini AI

El prompt enviado a Gemini en ambos módulos sigue la misma filosofía: **restringir la respuesta al contexto del documento**. Ejemplo del módulo OCR:

```
Eres un asistente que responde preguntas usando SOLO información del documento.
REGLAS IMPORTANTES:
- No copies el documento completo.
- Extrae SOLO la información necesaria.
- Responde de forma breve y directa.

DOCUMENTO: """..."""
HISTORIAL DE CHAT: [...]
PREGUNTA DEL USUARIO: ...
```

Esta restricción es importante para evitar que el modelo "alucine" información que no está en el documento subido por el usuario.

El **historial de conversación** se pasa como lista de strings (pares pregunta/respuesta) y está limitado a las últimas 6-10 entradas para no sobrepasar el contexto del modelo.

---

## 7. Interfaz de usuario

La interfaz está diseñada con un estilo **glassmorphism oscuro/azul marino** consistente en todas las páginas. Las decisiones de diseño principales fueron:

- **Diseño de dos columnas**: panel de carga/extracción a la izquierda, chat a la derecha. Permite ver simultáneamente el estado del documento y la conversación.
- **Indicadores de estado**: badges visuales que muestran en qué estado se encuentra el sistema (esperando imagen → procesando → contexto listo).
- **Historial de chat visible**: el chat se acumula en pantalla, permitiendo releer respuestas anteriores sin necesidad de navegar.
- **Deshabilitación progresiva de controles**: el input de chat permanece deshabilitado hasta que el OCR o el PDF han sido procesados correctamente, evitando peticiones sin contexto.

---

## 8. Problemas encontrados durante el desarrollo

### 8.1 Tiempo de carga inicial de PaddleOCR

**Problema**: PaddleOCR descarga modelos de detección y reconocimiento en la primera ejecución (~300 MB). En Hugging Face Spaces esto causaba timeouts en el primer request tras el despliegue.

**Solución**: inicializar el objeto `PaddleOCR` a nivel de módulo (fuera de cualquier endpoint), de forma que se cargue al arrancar la aplicación, no al recibir la primera petición.

```python
# Se carga al importar el módulo, no en el endpoint
ocr = PaddleOCR(use_angle_cls=True, lang="es")
```

### 8.2 Conflictos de versiones entre OpenCV y PaddleOCR

**Problema**: `paddleocr` requiere `opencv-contrib-python` para algunas utilidades internas, pero instalar esta versión junto con `opencv-python` (estándar) genera conflictos de importación.

**Solución**: incluir explícitamente `opencv-python-headless` en `requirements.txt` y asegurarse de no mezclar las tres variantes de OpenCV en el mismo entorno.

### 8.3 Límites de tasa de la API de Gemini

**Problema**: en sesiones de prueba intensiva con varios usuarios simultáneos, la API devolvía errores `429 Too Many Requests`.

**Solución**: implementar el sistema de reintentos con `sleep(2)` entre intentos y el modelo de fallback (`gemini-3.1-flash-lite-preview`), que tiene cuotas independientes.

### 8.4 PDFs sin texto extraíble

**Problema**: muchos PDFs en circulación son en realidad escaneos (imágenes embebidas). `pypdf` devuelve una cadena vacía en estos casos.

**Solución**: se añadió una comprobación explícita del texto extraído y se devuelve un error HTTP 500 con el mensaje "No se pudo extraer texto". Como mejora futura se podría integrar OCR también para PDFs escaneados (por ejemplo, usando `pdf2image` + PaddleOCR).

### 8.5 Embeddings lentos en CPU para documentos largos

**Problema**: calcular embeddings para 200-300 chunks con `all-MiniLM-L6-v2` en CPU tardaba varios segundos en el entorno gratuito de Hugging Face Spaces.

**Solución**: se implementó `embedding_batch_safe()` que procesa los textos en batches de 32, reduciendo el overhead de llamadas al modelo. Los embeddings se calculan una sola vez al subir el PDF y se almacenan en `VECTOR_DB`.

### 8.6 Pérdida de documentos tras reinicio del servidor

**Problema**: al usar un diccionario en memoria, cualquier reinicio del worker de Uvicorn (por inactividad en el tier gratuito de HF Spaces) borraba todos los documentos procesados.

**Solución parcial**: se informó al usuario mediante el mensaje de error "Documento no encontrado" para que vuelva a subir el archivo. Como mejora futura se consideró persistencia en disco o en una base de datos vectorial.

---

## 9. Imágenes que funcionan bien y que no funcionan

### ✅ Imágenes con buen rendimiento

| Tipo de imagen | Características | Resultado esperado |
|---|---|---|
| Capturas de pantalla de texto | Fondo blanco/claro, tipografía digital clara | Extracción casi perfecta |
| Documentos escaneados a alta resolución | ≥ 300 DPI, texto horizontal, contraste alto | Buena extracción, errores mínimos |
| Fotografías de carteles o señales | Texto grande, fuente sans-serif, iluminación uniforme | Buen reconocimiento |
| Facturas y recibos impresos | Texto tabular, fuente monoespacio, impresión limpia | Extracción fiable de datos numéricos |
| Texto en español o inglés | Idiomas configurados en PaddleOCR | Resultados óptimos |

### ❌ Imágenes con bajo rendimiento

| Tipo de imagen | Problema | Motivo técnico |
|---|---|---|
| Texto manuscrito | Extracción incorrecta o vacía | PaddleOCR está optimizado para texto impreso; la escritura a mano varía enormemente en forma |
| Imágenes con bajo contraste (texto gris sobre blanco) | Caracteres perdidos | El preprocesado de escala de grises no mejora suficientemente el contraste |
| Fotos tomadas en ángulo pronunciado (perspectiva) | Texto distorsionado o no detectado | La clasificación de ángulo (`use_angle_cls`) solo corrige rotación simple, no transformaciones de perspectiva |
| Imágenes con resolución muy baja (< 72 DPI) | Texto borroso, caracteres ininteligibles | PaddleOCR necesita una resolución mínima para que los modelos de detección funcionen |
| Texto sobre fondos con muchas texturas o patrones | Ruido en la detección | El preprocesado actual (solo escala de grises) no elimina texturas complejas |
| Imágenes con múltiples idiomas mezclados | Caracteres erróneos en el idioma secundario | El modelo está configurado solo para español; otros alfabetos (cirílico, árabe, chino) producen basura |
| Texto muy pequeño (< 8px en la imagen original) | No detectado | Los modelos de detección de PaddleOCR tienen una escala mínima de operación |
| Capturas con sombras fuertes | Partes del texto no detectadas | Las sombras crean gradientes locales que confunden la binarización |

### 💡 Recomendaciones para el usuario

Para obtener los mejores resultados:
- Usar imágenes con al menos **150 DPI**.
- Asegurarse de que el texto esté **horizontal** o con una inclinación menor de 30°.
- Evitar fondos con patrones, texturas o gradientes.
- Si la imagen está muy oscura o con poco contraste, editarla previamente (aumentar brillo/contraste).

---

## 10. Despliegue en Hugging Face Spaces

La aplicación se desplegó en **Hugging Face Spaces** usando el SDK de **Docker** (o el runner de Python con `uvicorn`). Los puntos clave del despliegue son:

- La API key de Gemini se almacena como **secret** en la configuración del Space, no en el código.
- Las dependencias se instalan desde `requirements.txt`, que incluye versiones fijadas para garantizar reproducibilidad.
- El servidor arranca con `uvicorn main:app --host 0.0.0.0 --port 7860`.
- Los archivos estáticos (HTML, CSS) se sirven desde el directorio `static/`.

**Limitaciones del tier gratuito:**
- CPU compartida, sin GPU → PaddleOCR funciona pero es más lento.
- El Space se "duerme" tras un periodo de inactividad, borrando los datos en memoria.
- Memoria RAM limitada a ~16 GB, suficiente para el modelo de embeddings y PaddleOCR.

---

## 11. Conclusiones y trabajo futuro

### Lo que funciona bien
- El flujo completo OCR → Chat es fluido e intuitivo para el usuario final.
- La búsqueda semántica en el módulo PDF recupera fragmentos relevantes de forma consistente.
- El sistema de fallback de Gemini mejora la resiliencia ante errores de la API.
- La interfaz de usuario da retroalimentación clara en cada etapa del proceso.

### Áreas de mejora

| Área | Mejora propuesta |
|---|---|
| OCR en PDFs escaneados | Integrar `pdf2image` + PaddleOCR para PDFs sin texto nativo |
| Persistencia | Almacenar vectores en disco (SQLite + numpy) o en Chroma/FAISS |
| Preprocesado de imagen | Añadir binarización adaptativa (Otsu) y corrección de perspectiva |
| Autenticación | Añadir un sistema de usuarios para separar documentos por sesión |
| Soporte multiidioma OCR | Configurar PaddleOCR con múltiples idiomas o permitir al usuario seleccionarlo |
| Streaming de respuestas | Usar la API de streaming de Gemini para mostrar la respuesta palabra a palabra |

---

