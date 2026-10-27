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
    citas: List[Cita] = [] # Ahora el backend de NestJS nos manda solo las de HOY
    tomas_hoy_ids: List[str] = []

@app.post("/evaluar")
async def evaluar_pregunta(req: EvaluacionRequest):
    try:
        # Cruce de datos
        medicamentos_pendientes = [m for m in req.medicamentos_activos if m.id not in req.tomas_hoy_ids]
        medicamentos_tomados = [m for m in req.medicamentos_activos if m.id in req.tomas_hoy_ids]
        
        # SYSTEM PROMPT ACTUALIZADO (Para cuando se conecte a la API de OpenAI/Gemini)
        system_prompt = f"""
        Eres Luma, un asistente médico amigable para adultos mayores.
        REGLA CRÍTICA: NO puedes dar diagnósticos médicos.
        
        CONTEXTO DE HOY:
        - Citas de hoy: {len(req.citas)} programadas.
        - Ya se tomó: {', '.join([m.nombre_farmaco for m in medicamentos_tomados]) if medicamentos_tomados else 'Ninguno'}.
        - Le falta tomar: {', '.join([m.nombre_farmaco for m in medicamentos_pendientes]) if medicamentos_pendientes else 'Ninguno'}.
        
        NUEVAS INSTRUCCIONES:
        1) Si pregunta por 'Tomas Realizadas', felicítalo y lista SOLO lo que ya se tomó.
        2) Si pide 'Resumen del Día', dale un reporte amigable en 3 partes: Citas, Ya tomados y Pendientes.
        """
        
        # === SIMULACIÓN DE RESPUESTAS ESTRUCTURADAS (MVP Local) ===
        respuesta = ""
        
        if req.pregunta_id == "PROXIMA_TOMA":
            if not req.medicamentos_activos:
                respuesta = "No tienes ningún medicamento registrado en tu rutina."
            elif not medicamentos_pendientes:
                respuesta = "¡Felicidades! Ya te tomaste todas tus medicinas por hoy."
            else:
                respuesta = "Aún tienes pendiente tomar: "
                for med in medicamentos_pendientes:
                    horas = ", ".join(med.horarios_especificos) if med.horarios_especificos else "tus horarios asignados"
                    respuesta += f"{med.nombre_farmaco} a las {horas}. "

        elif req.pregunta_id == "TOMAS_REALIZADAS":
            if not medicamentos_tomados:
                respuesta = "Aún no te has tomado ninguna medicina el día de hoy. ¡Recuerda revisar tus horarios!"
            else:
                respuesta = "¡Muy bien! Hoy ya te tomaste: "
                respuesta += ", ".join([m.nombre_farmaco for m in medicamentos_tomados])
                respuesta += ". ¡Sigue así con tu tratamiento!"

        elif req.pregunta_id == "CITAS_HOY":
             if not req.citas:
                 respuesta = "No tienes citas médicas en tu agenda para el día de hoy."
             else:
                 respuesta = "Hoy tienes: "
                 for cita in req.citas:
                     hora = cita.fecha_hora.strftime("%H:%M")
                     respuesta += f"{cita.titulo_evento} a las {hora} en {cita.ubicacion}. "
                     
        elif req.pregunta_id == "RESUMEN_DIA":
            # 1. Citas
            if not req.citas:
                citas_str = "No tienes citas hoy."
            else:
                citas_str = "Tus citas de hoy: " + ", ".join([f"{c.titulo_evento} a las {c.fecha_hora.strftime('%H:%M')}" for c in req.citas])
            
            # 2. Tomados
            tomados_str = "Aún no tomas pastillas hoy." if not medicamentos_tomados else "Ya tomaste: " + ", ".join([m.nombre_farmaco for m in medicamentos_tomados])
            
            # 3. Pendientes
            pendientes_str = "No tienes tomas pendientes." if not medicamentos_pendientes else "Te falta tomar: " + ", ".join([f"{m.nombre_farmaco} a las {m.horarios_especificos[0] if m.horarios_especificos else 'su hora'}" for m in medicamentos_pendientes])
            
            respuesta = f"¡Hola! Aquí tienes el resumen de tu día. Primero, {citas_str} Segundo, {tomados_str} Por último, {pendientes_str} ¡Espero que tengas un excelente día!"

        elif "SINTOMA" in req.pregunta_id or "DOLOR" in req.pregunta_id:
            respuesta = "Por tu seguridad, no puedo dar diagnósticos. Por favor, presiona el botón SOS."
        else:
            respuesta = "¿Puedes repetirlo? Estoy aquí para ayudarte."

        return {"respuesta": respuesta}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))