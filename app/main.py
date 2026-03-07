import io
import os
import json
import re
import tempfile
import urllib.request
import ssl
from dotenv import load_dotenv

import PyPDF2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cloudinary
import cloudinary.api
import cloudinary.uploader
from google import genai
from starlette.middleware.cors import CORSMiddleware

# Librerías para el re-entrenamiento automático
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz

load_dotenv()

app = FastAPI(title="Microservicio Orientación Vocacional (IA + Storage + Dashboard)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de Cloudinary desde .env
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Configuración de Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    client = None
    print(f"Error inicializando Gemini: {e}")

# Variables globales de memoria
training_context = ""
processed_pdfs = []

# IDs para persistencia en Cloudinary
PROCESSED_FILES_ID = "sov_lima_processed_v1.json"
TRAINING_CONTEXT_ID = "sov_lima_context_v1.txt"


# --- UTILITARIOS DE PERSISTENCIA ---

def get_file_from_cloudinary(public_id):
    """Descarga un archivo de texto o JSON desde Cloudinary."""
    try:
        res = cloudinary.api.resource(public_id, resource_type="raw")
        url = res.get('secure_url')
        if url:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            contexto_ssl = ssl._create_unverified_context()
            with urllib.request.urlopen(req, context=contexto_ssl) as response:
                return response.read().decode('utf-8')
    except Exception:
        return None
    return None


def upload_file_to_cloudinary(content_str, public_id):
    """Sube el contexto o lista de PDFs a Cloudinary como archivo plano."""
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt", encoding='utf-8') as temp:
            temp.write(content_str)
            temp_path = temp.name

        cloudinary.uploader.upload(
            temp_path,
            resource_type="raw",
            public_id=public_id,
            overwrite=True
        )
        os.remove(temp_path)
    except Exception as e:
        print(f"Error sincronizando con Cloudinary: {e}")


# --- LÓGICA DE PROCESAMIENTO ---

async def process_new_pdfs():
    global training_context, processed_pdfs
    contexto_ssl = ssl._create_unverified_context()

    print("Iniciando Sincronización Total con Cloudinary...")
    try:
        recursos_actuales = []
        for r_type in ["image", "raw"]:
            next_cursor = None
            while True:
                res = cloudinary.api.resources(resource_type=r_type, type="upload", max_results=100,
                                               next_cursor=next_cursor)
                recursos_actuales.extend(res.get('resources', []))
                next_cursor = res.get('next_cursor')
                if not next_cursor: break

        nuevo_texto_acumulado = ""
        lista_final_pdfs = []

        for resource in recursos_actuales:
            url_pdf = resource.get('secure_url')
            if url_pdf and url_pdf.endswith('.pdf'):
                print(f"Procesando: {url_pdf}")
                try:
                    req = urllib.request.Request(url_pdf, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, context=contexto_ssl) as response:
                        pdf_file = io.BytesIO(response.read())
                        reader = PyPDF2.PdfReader(pdf_file)
                        for page in reader.pages:
                            nuevo_texto_acumulado += (page.extract_text() or "") + "\n"
                    lista_final_pdfs.append(url_pdf)
                except Exception as e:
                    print(f"Error en {url_pdf}: {e}")

        # Reemplazo total de la memoria
        training_context = nuevo_texto_acumulado
        processed_pdfs = lista_final_pdfs

        # Sincronizar archivos de persistencia
        upload_file_to_cloudinary(training_context, TRAINING_CONTEXT_ID)
        upload_file_to_cloudinary(json.dumps(processed_pdfs), PROCESSED_FILES_ID)
        print("Sincronización finalizada. Memoria actualizada según la nube.")

    except Exception as e:
        print(f"Error en Full Sync: {e}")

@app.on_event("startup")
async def startup_event():
    global training_context, processed_pdfs

    # 1. Cargar memoria previa desde Cloudinary
    print("Cargando memoria del sistema...")

    saved_pdfs = get_file_from_cloudinary(PROCESSED_FILES_ID)
    if saved_pdfs:
        processed_pdfs = json.loads(saved_pdfs)

    saved_context = get_file_from_cloudinary(TRAINING_CONTEXT_ID)
    if saved_context:
        training_context = saved_context
        print("Memoria cargada exitosamente.")

    # 2. Ejecutar una revisión inicial
    await process_new_pdfs()

    # 3. Configurar el Scheduler (Domingos 11:00 PM hora Lima)
    lima_tz = pytz.timezone('America/Lima')
    scheduler = AsyncIOScheduler(timezone=lima_tz)
    scheduler.add_job(process_new_pdfs, 'cron', day_of_week='sun', hour=23, minute=0)
    scheduler.start()
    print("Programador activado: Revisión de PDFs cada domingo a las 23:00.")


# --- ENDPOINTS ---

class ChatRequest(BaseModel):
    mensaje: str


@app.post("/chat")
async def chat_con_ia(request: ChatRequest):
    if not GEMINI_API_KEY or client is None:
        raise HTTPException(status_code=500, detail="API Gemini no configurada.")

    # Usamos el contexto acumulado
    files_str = training_context if training_context else "conocimiento general de carreras en Perú"

    system_instruction = (
        f"Eres un orientador vocacional experto en jóvenes peruanos. "
        f"Tu tono es motivador, inspirador y profesional, pero cercano. "
        f"Recomienda basándote en: {files_str[:50000]}. "  # Limitamos para no saturar el prompt
        "REGLA DE INTERPRETACIÓN SAGAZ: Si el usuario pide algo fuera de lugar (recetas, juegos, etc.), "
        "interpreta su gusto y recomiéndale una carrera. Ej: Recetas -> Gastronomía; Hackeo -> Ciberseguridad. "
        "FORMATO: Máximo 6 líneas y 2 emojis. Habla de 'tú' y sé directo. 🎓🚀"
    )

    try:
        respuesta = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[system_instruction, f"Usuario: {request.mensaje}"]
        )
        return {"respuesta": respuesta.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error con Gemini: {str(e)}")


@app.get("/")
async def root():
    return {"status": "online", "pdfs_procesados": len(processed_pdfs)}