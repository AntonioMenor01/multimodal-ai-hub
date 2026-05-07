from fastapi import APIRouter, File, UploadFile, HTTPException
from google import genai
from pydantic import BaseModel
import cv2
import numpy as np
from paddleocr import PaddleOCR
from dotenv import load_dotenv
import os
from time import sleep
from google.genai import types

load_dotenv()

# Configuración del Router
router = APIRouter()

# -------------------------
# CONFIGURACIÓN Y MODELOS
# -------------------------
MODEL_ID = "models/gemini-2.5-flash"
MODEL_ID2 = "models/gemini-3.1-flash-lite-preview"

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
config = types.GenerateContentConfig(temperature=0)

# Motor OCR (se inicializa una vez al cargar el módulo)
ocr = PaddleOCR(use_angle_cls=True, lang="es")

# -------------------------
# MODELOS DE DATOS (Pydantic)
# -------------------------
class ChatRequest(BaseModel):
    pregunta: str
    contexto: str  
    historial: list[str] = []

class ChatResponse(BaseModel):
    respuesta: str

class OCRRequest(BaseModel):
    text: str

# -------------------------
# UTILIDADES
# -------------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def construir_prompt(documento, historial, pregunta): 
    return f"""
        Eres un asistente que responde preguntas usando SOLO información del documento.
        REGLAS IMPORTANTES:
        - No copies el documento completo.
        - Extrae SOLO la información necesaria.
        - Responde de forma breve y directa.

        DOCUMENTO:
        \"\"\"{documento}\"\"\"

        HISTORIAL DE CHAT:
        {historial}

        PREGUNTA DEL USUARIO:
        {pregunta}

        RESPUESTA:
    """

def intentos_modelos(max_reintentos, prompt_text):
    ultimo_error = None
    for intento in range(max_reintentos):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt_text,
                config=config
            )
            return ChatResponse(respuesta=response.text)
        except Exception as e:
            ultimo_error = e
            sleep(2) 

    try:
        response = client.models.generate_content(
            model=MODEL_ID2,
            contents=prompt_text,
            config=config
        )
        return ChatResponse(respuesta=f"[fallback] {response.text}")
    except Exception as e_fallback:
        raise HTTPException(
            status_code=500,
            detail={"error_principal": str(ultimo_error), "error_fallback": str(e_fallback)}
        )

# -------------------------
# ENDPOINTS (Rutas)
# -------------------------

@router.post("/upload", response_model=OCRRequest)
async def cargar_imagen(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Formato no soportado.")
    
    try:
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img_final = preprocess_image(img)
        results = ocr.ocr(img_final)
        texto_extraido = ""

        if results and len(results) > 0:
            for linea in results[0]:
                texto_extraido += linea[1][0] + " "

        return OCRRequest(text=texto_extraido.strip() if texto_extraido else "No se detectó texto")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat_ocr(request: ChatRequest):
    prompt_text = construir_prompt(request.contexto, request.historial, request.pregunta)
    return intentos_modelos(2, prompt_text)