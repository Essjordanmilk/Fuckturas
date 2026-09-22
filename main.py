import os
import json
from fastapi import FastAPI, Request, HTTPException
import google.generativeai as genai
from pydantic import BaseModel

app = FastAPI()

# Configurar la API Key de Gemini desde las variables de entorno
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ ADVERTENCIA: La variable de entorno GEMINI_API_KEY no está configurada.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Estructura del JSON que queremos que la IA nos devuelva obligatoriamente
class RespuestaFactura(BaseModel):
    total: float
    categoria: str
    comercio: str

@app.get("/")
def home():
    return {"status": "ok", "message": "Servidor del Bot de Facturas Activo"}

@app.post("/procesar-factura")
async def procesar_factura(request: Request):
    try:
        # 1. Leer los datos binarios de la imagen enviados por n8n
        contenido_imagen = await request.body()
        if not contenido_imagen:
            raise HTTPException(status_code=400, detail="No se recibieron datos de imagen")

        # 2. Configurar el modelo Gemini
        model = genai.GenerativeModel('gemini-3.6-flash')

        # 3. Detectar de forma automática el tipo de imagen según los primeros bytes
        tipo_mime = "image/jpeg"
        if contenido_imagen.startswith(b'\x89PNG'):
            tipo_mime = "image/png"
        elif contenido_imagen.startswith(b'\x47\x49\x46'):
            tipo_mime = "image/gif"

        datos_imagen = {
            "mime_type": tipo_mime,
            "data": contenido_imagen
        }

        # 4. Diseñar el Prompt con instrucciones estrictas de formato
        prompt = (
            "Analiza esta imagen de factura o recibo de compra. "
            "Extrae el monto total cobrado, el nombre del comercio o lugar de compra, "
            "y determina la categoría del gasto únicamente entre las siguientes opciones: "
            "'mercado', 'transporte', 'ocio', 'servicios', 'salud', 'educacion' u 'otros'. "
            "Debes responder EXCLUSIVAMENTE con un objeto JSON válido que contenga las llaves: "
            "'total' (número), 'categoria' (texto) y 'comercio' (texto). No incluyas texto extra."
        )

        # 5. Llamar a la Inteligencia Artificial usando Structured Outputs para asegurar el JSON
        response = model.generate_content(
            [prompt, datos_imagen],
            generation_config={"response_mime_type": "application/json"}
        )

        # 6. Parsear la respuesta de texto a un JSON real y validarlo
        # Eliminamos posibles caracteres extraños o bloques de código markdown que a veces añade la IA
        texto_limpio = response.text.strip().replace("```json", "").replace("```", "")
        datos_extraidos = json.loads(texto_limpio)
        
        # Devolver la respuesta limpia a n8n
        return datos_extraidos

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la factura: {str(e)}")
