# [MOTOR IA] Ruta: app/models/schemas.py

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# --- ESQUEMAS DEL ASISTENTE CLÁSICO ---
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

class AudioRequest(BaseModel):
    audio_base64: str

# --- ESQUEMAS PARA MACHINE LEARNING ---
class PacienteHistorico(BaseModel):
    retraso_promedio_minutos: float
    consultas_sintomas: int
    alertas_sos: int
    riesgo_abandono: int

class EntrenamientoRequest(BaseModel):
    datos_historicos: List[PacienteHistorico]

class PrediccionRiesgoRequest(BaseModel):
    retraso_promedio_minutos: float
    consultas_sintomas: int
    alertas_sos: int

class PrediccionRiesgoResponse(BaseModel):
    riesgo_regla_fija: bool
    riesgo_machine_learning: bool
    exactitud_modelo_ml: float
    # NUEVOS CAMPOS PARA LA FASE 1
    f1_score_ml: float
    algoritmo_utilizado: str