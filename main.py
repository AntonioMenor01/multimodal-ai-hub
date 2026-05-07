from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routers import pdf_logic, ocr_logic
import os

app = FastAPI(title="AI Multimodal Hub")

# Routers de lógica
app.include_router(pdf_logic.router, prefix="/api/pdf", tags=["PDF"])
app.include_router(ocr_logic.router, prefix="/api/ocr", tags=["OCR"])

# Servir archivos estáticos (CSS, JS, imágenes, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ruta para el menú principal
@app.get("/")
async def read_index():
    return FileResponse(os.path.join("static", "index.html"))

# Rutas específicas para cada funcionalidad
@app.get("/pdf")
async def read_pdf():
    return FileResponse(os.path.join("static", "pdf.html"))

@app.get("/ocr") 
async def read_ocr():
    return FileResponse(os.path.join("static", "ocr.html"))