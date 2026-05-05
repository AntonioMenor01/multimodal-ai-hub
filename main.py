from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
import numpy as np
import uuid
from typing import List
from pydantic import Field
from sentence_transformers import SentenceTransformer
from google import genai
import os

# -------------------------
# APP
# -------------------------

app = FastAPI()

VECTOR_DB = {}
MAX_DOCS = 20

MODEL = "gemini-2.5-flash"
MAX_PAGES = 50

MAX_CHUNKS = 10
SIMILARITY_THRESHOLD = 0.20

load_dotenv()

# -------------------------
# MODELO EMBEDDING LOCAL
# -------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embedding(texto):
    return embed_model.encode(texto)

def embedding_batch_safe(textos, batch_size=32):
    all_vectors = []

    for i in range(0, len(textos), batch_size):
        batch = textos[i:i + batch_size]
        vectors = embed_model.encode(batch)
        all_vectors.extend(vectors)

    return all_vectors

# -------------------------
# UTILIDADES
# -------------------------

def dividir_texto(texto, chunk_size=300, overlap=50):
    palabras = texto.split()
    chunks = []

    step = max(1, chunk_size - overlap)

    for i in range(0, len(palabras), step):
        chunk = palabras[i:i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks

def similitud(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0

    return np.dot(a, b) / (norm_a * norm_b)

# -------------------------
# MODELO CHAT
# -------------------------

class ChatRequest(BaseModel):
    pregunta: str
    doc_id: str
    historial: List[str] = Field(default_factory=list)

# -------------------------
# UPLOAD PDF
# -------------------------

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        raise HTTPException(400, "Solo PDF")

    try:
        reader = PdfReader(file.file)
        texto = ""

        if len(reader.pages) > MAX_PAGES:
            raise HTTPException(400, "PDF demasiado grande (máx 50 páginas)")

        for page in reader.pages:
            try:
                texto += page.extract_text() or ""
            except:
                continue

        if not texto.strip():
            raise Exception("No se pudo extraer texto")

        # chunking
        chunks = dividir_texto(texto)

        vectors = embedding_batch_safe(chunks)

        vectores = [
            {"texto": chunk, "vector": vec}
            for chunk, vec in zip(chunks, vectors)
        ]

        doc_id = str(uuid.uuid4())
        VECTOR_DB[doc_id] = vectores

        if len(VECTOR_DB) > MAX_DOCS:
            VECTOR_DB.pop(next(iter(VECTOR_DB)))

        return {
            "doc_id": doc_id,
            "chunks": len(vectores),
            "mensaje": "PDF procesado correctamente",
            "paginas_estimadas": len(reader.pages)
        }

    except Exception as e:
        raise HTTPException(500, str(e))

# -------------------------
# CHAT RAG
# -------------------------

@app.post("/api/chat")
async def chat(req: ChatRequest):

    if req.doc_id not in VECTOR_DB:
        raise HTTPException(404, "Documento no encontrado")

    try:
        q_vec = embedding(req.pregunta)
        chunks = VECTOR_DB[req.doc_id]

        scored = []

        for c in chunks:
            sim = similitud(q_vec, c["vector"])
            if sim > SIMILARITY_THRESHOLD:
                scored.append((sim, c["texto"]))

        if not scored:
            scored = [(similitud(q_vec, c["vector"]), c["texto"]) for c in chunks]

        scored.sort(reverse=True)
        top_chunks = scored[:MAX_CHUNKS]

        contexto = "\n\n---\n\n".join([c[1] for c in top_chunks])
        historial = "\n".join(req.historial[-6:])

        prompt = f"""
Eres un asistente experto en análisis de documentos.

Responde SOLO usando el contexto proporcionado.
Si no está en el contexto, di claramente que no aparece.

HISTORIAL:
{historial}

CONTEXTO:
{contexto}

PREGUNTA:
{req.pregunta}
"""

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        res = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        return {"respuesta": res.text or "No se pudo generar respuesta"}

    except Exception as e:
        raise HTTPException(500, str(e))

# -------------------------
# FRONTEND
# -------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")