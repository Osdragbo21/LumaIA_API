# [MOTOR IA] Ruta: app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="LumaIA Engine API")

class Medicamento(BaseModel):
    id: str
    nombre_farmaco: str
    horarios_especificos: List[str]

class Cita(BaseModel):
    titulo_evento: str
    fecha_hora: datetime
    ubicacion: Optional[str] = "Sin ubicación"

class EvaluacionRequest(BaseModel):
    pregunta_id: str
    medicamentos_activos: List[Medicamento] = []
    citas: List[Cita] = [] 
    tomas_hoy_ids: List[str] = []
    prompt_forzado: Optional[str] = None

# NUEVO: Modelo para recibir el audio
class AudioRequest(BaseModel):
    audio_base64: str

@app.post("/evaluar")
async def evaluar_pregunta(req: EvaluacionRequest):
    try:
        medicamentos_pendientes = [m for m in req.medicamentos_activos if m.id not in req.tomas_hoy_ids]
        medicamentos_tomados = [m for m in req.medicamentos_activos if m.id in req.tomas_hoy_ids]
        
        respuesta_texto = ""
        intencion = "CONSULTA_GENERAL" 
        
        if req.pregunta_id == "PROXIMA_TOMA":
            if not req.medicamentos_activos:
                respuesta_texto = "No tienes ningún medicamento registrado en tu rutina."
            elif not medicamentos_pendientes:
                respuesta_texto = "¡Felicidades! Ya te tomaste todas tus medicinas por hoy."
            else:
                respuesta_texto = "Aún tienes pendiente tomar: "
                for med in medicamentos_pendientes:
                    horas = ", ".join(med.horarios_especificos) if med.horarios_especificos else "tus horarios asignados"
                    respuesta_texto += f"{med.nombre_farmaco} a las {horas}. "

        elif req.pregunta_id == "TOMAS_REALIZADAS":
            if not medicamentos_tomados:
                respuesta_texto = "Aún no te has tomado ninguna medicina el día de hoy. ¡Recuerda revisar tus horarios!"
            else:
                respuesta_texto = "¡Muy bien! Hoy ya te tomaste: "
                respuesta_texto += ", ".join([m.nombre_farmaco for m in medicamentos_tomados])
                respuesta_texto += ". ¡Sigue así con tu tratamiento!"

        elif req.pregunta_id == "CITAS_HOY":
                if not req.citas:
                    respuesta_texto = "No tienes citas médicas en tu agenda para el día de hoy."
                else:
                    respuesta_texto = "Hoy tienes: "
                    for cita in req.citas:
                        hora = cita.fecha_hora.strftime("%H:%M")
                        respuesta_texto += f"{cita.titulo_evento} a las {hora} en {cita.ubicacion}. "
                        
        elif req.pregunta_id == "RESUMEN_DIA":
            if not req.citas:
                citas_str = "No tienes citas hoy."
            else:
                citas_str = "Tus citas de hoy: " + ", ".join([f"{c.titulo_evento} a las {c.fecha_hora.strftime('%H:%M')}" for c in req.citas])
            
            tomados_str = "Aún no tomas pastillas hoy." if not medicamentos_tomados else "Ya tomaste: " + ", ".join([m.nombre_farmaco for m in medicamentos_tomados])
            pendientes_str = "No tienes tomas pendientes." if not medicamentos_pendientes else "Te falta tomar: " + ", ".join([f"{m.nombre_farmaco} a las {m.horarios_especificos[0] if m.horarios_especificos else 'su hora'}" for m in medicamentos_pendientes])
            
            respuesta_texto = f"¡Hola! Aquí tienes el resumen de tu día. Primero, {citas_str} Segundo, {tomados_str} Por último, {pendientes_str} ¡Espero que tengas un excelente día!"

        elif "SINTOMA" in req.pregunta_id or "DOLOR" in req.pregunta_id:
            respuesta_texto = "Por tu seguridad, no puedo dar diagnósticos. Por favor, presiona el botón SOS."
        else:
            respuesta_texto = "¿Puedes repetirlo? Estoy aquí para ayudarte."

        return {
            "respuesta_texto": respuesta_texto,
            "intencion": intencion
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NUEVO: Endpoint para procesar el dictado de voz
@app.post("/transcribir")
async def transcribir_audio(req: AudioRequest):
    try:
        # AQUÍ CONECTARÁS WHISPER EN EL FUTURO.
        # Ejemplo:
        # import base64
        # audio_bytes = base64.b64decode(req.audio_base64)
        # texto_transcrito = modelo_whisper.transcribe(audio_bytes)
        
        texto_simulado = "Esta es una transcripción de prueba. El motor de voz está conectado correctamente al backend."
        
        return {"texto": texto_simulado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))