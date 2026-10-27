# Ruta: app/models/schemas.py
from pydantic import BaseModel
from typing import List

class Medicamento(BaseModel):
    nombre_farmaco: str
    horarios_especificos: List[str]

class EvaluacionRequest(BaseModel):
    pregunta_id: str
    medicamentos_activos: List[Medicamento]

class EvaluacionResponse(BaseModel):
    respuesta: str