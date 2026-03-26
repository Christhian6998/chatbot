import io
import os
import json
import re
import tempfile
import urllib.request
import ssl
from dotenv import load_dotenv

import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import cloudinary
import cloudinary.api
import cloudinary.uploader
from google import genai
from starlette.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import pytz

load_dotenv()

app = FastAPI(title="Microservicio Orientación Vocacional SOV LIMA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    client = None
    print(f"Error inicializando Gemini: {e}")

training_context = ""
processed_pdfs = []

PROCESSED_FILES_ID = "sov_lima_processed_v2.json"
TRAINING_CONTEXT_ID = "sov_lima_context_v2.txt"


# --- UTILITARIOS ---

def get_file_from_cloudinary(public_id):
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
    try:
        ext = os.path.splitext(public_id)[1] or ".txt"
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=ext, encoding='utf-8') as temp_file:
            temp_file.write(content_str)
            temp_path = temp_file.name

        cloudinary.uploader.upload(
            temp_path,
            resource_type="raw",
            public_id=public_id,
            overwrite=True
        )
        os.remove(temp_path)
    except Exception as e:
        print(f"Error subiendo a Cloudinary: {e}")


def get_relevant_context(context_str, query, max_chars=80000):
    if not context_str or len(context_str) <= max_chars:
        return context_str

    lines = context_str.split('\n')
    chunks, current_chunk = [], ""
    for line in lines:
        current_chunk += line + "\n"
        if len(current_chunk) > 3000:
            chunks.append(current_chunk)
            current_chunk = ""
    if current_chunk: chunks.append(current_chunk)

    query_words = set(w.lower() for w in re.findall(r'\w+', query) if len(w) > 3)
    if not query_words: return context_str[:max_chars]

    scored_chunks = [(sum(chunk.lower().count(w) for w in query_words), chunk) for chunk in chunks]
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    relevant_text = ""
    for score, chunk in scored_chunks:
        if len(relevant_text) + len(chunk) > max_chars: break
        if score > 0: relevant_text += chunk + "\n"

    return relevant_text if len(relevant_text) >= 2000 else context_str[:max_chars]


# --- LÓGICA DE PROCESAMIENTO ---

async def process_new_pdfs():
    global training_context, processed_pdfs
    contexto_ssl = ssl._create_unverified_context()
    print("Buscando nuevos PDFs en Cloudinary...")

    try:
        recursos_totales = []
        for r_type in ["image", "raw"]:
            next_cursor = None
            while True:
                try:
                    res = cloudinary.api.resources(resource_type=r_type, type="upload", max_results=100,
                                                   next_cursor=next_cursor)
                    recursos_totales.extend(res.get('resources', []))
                    next_cursor = res.get('next_cursor')
                    if not next_cursor: break
                except:
                    break

        texto_nuevo, nuevos_pdfs = "", 0

        for resource in recursos_totales:
            url_pdf = resource.get('secure_url')
            if url_pdf and url_pdf.endswith('.pdf') and url_pdf not in processed_pdfs:
                print(f"Leyendo nuevo PDF: {url_pdf}")
                try:
                    req = urllib.request.Request(url_pdf, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, context=contexto_ssl) as response:
                        pdf_data = response.read()
                        doc = fitz.open(stream=pdf_data, filetype="pdf")
                        for page in doc: texto_nuevo += page.get_text() + "\n"
                        doc.close()

                    processed_pdfs.append(url_pdf)
                    nuevos_pdfs += 1
                except Exception as e:
                    print(f"Error leyendo {url_pdf}: {e}")

        if nuevos_pdfs > 0:
            training_context += "\n" + texto_nuevo
            upload_file_to_cloudinary(json.dumps(processed_pdfs), PROCESSED_FILES_ID)
            upload_file_to_cloudinary(training_context, TRAINING_CONTEXT_ID)
            print(f"Actualizados {nuevos_pdfs} PDFs.")
        else:
            print("No hay PDFs nuevos.")

    except Exception as e:
        print(f"Error en procesamiento: {e}")


@app.on_event("startup")
async def startup_event():
    global training_context, processed_pdfs

    saved_pdfs = get_file_from_cloudinary(PROCESSED_FILES_ID)
    if saved_pdfs: processed_pdfs = json.loads(saved_pdfs)

    saved_context = get_file_from_cloudinary(TRAINING_CONTEXT_ID)
    if saved_context: training_context = saved_context

    await process_new_pdfs()

    lima_tz = pytz.timezone('America/Lima')
    scheduler = AsyncIOScheduler(timezone=lima_tz)
    scheduler.add_job(process_new_pdfs, 'cron', day_of_week='sun', hour=23, minute=0)
    scheduler.start()


# --- ENDPOINTS ---

class ChatRequest(BaseModel):
    user_id: str = "usuario_anonimo"
    mensaje: str


historial_usuarios = {}


@app.post("/chat")
async def chat_con_ia(request: Request, chat_req: ChatRequest):
    if not GEMINI_API_KEY or client is None:
        raise HTTPException(status_code=500, detail="API Gemini no configurada.")

    user_id = request.client.host if chat_req.user_id == "usuario_anonimo" and request.client else chat_req.user_id
    if user_id not in historial_usuarios: historial_usuarios[user_id] = []

    files_str = get_relevant_context(training_context,
                                     chat_req.mensaje) if training_context else "base de datos general"

    system_instruction = (
        f"Eres un orientador vocacional experto en jóvenes de Lima Metropolitana. "
        f"Limítate estrictamente a instituciones y realidades de Lima Metropolitana. "
        f"Tu tono es motivador y profesional. Recomienda basándote en: {files_str}. "
        "REGLA CRÍTICA: Nunca digas 'no puedo responder eso'. Si piden algo fuera de lugar (recetas, juegos, etc.), "
        "interpreta su gusto y recomiéndale una carrera. Ej: Recetas -> Gastronomía; Hackeo -> Ciberseguridad. "
        "FORMATO: Máximo 6 líneas y 2 emojis. Habla de 'tú' y sé directo. 🎓🚀"
    )

    historial_usuarios[user_id].append(f"Usuario: {chat_req.mensaje}")
    historial_reciente = "\n".join(historial_usuarios[user_id][-10:])

    try:
        respuesta = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[system_instruction, f"Historial:\n{historial_reciente}\n\nResponde al último mensaje."]
        )
        historial_usuarios[user_id].append(f"Asistente: {respuesta.text}")
        return {"respuesta": respuesta.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error con Gemini: {str(e)}")


@app.get("/")
async def root():
    return {"status": "online", "pdfs_procesados": len(processed_pdfs)}