from datetime import datetime
from app.models.schemas import EvaluacionRequest

def evaluar_reglas(payload: EvaluacionRequest) -> str:
    # REGLA RD-01: Intercepción de preguntas de diagnóstico médico o dosis
    preguntas_bloqueadas = ["DIAGNOSTICO", "AJUSTE_DOSIS", "SINTOMAS"]
    if payload.pregunta_id in preguntas_bloqueadas:
        return "Por seguridad, no puedo emitir diagnósticos ni alterar tratamientos. Consulte a su médico."

    match payload.pregunta_id:
        case "PROXIMA_TOMA":
            if not payload.medicamentos_activos:
                return "No tienes tomas registradas en este momento."

            hora_actual_str = datetime.now().strftime("%H:%M")
            proxima_toma_hora = None
            farmaco_proximo = None

            # Árbol de decisión iterativo para buscar la toma más cercana del día
            for medicamento in payload.medicamentos_activos:
                for hora_programada in medicamento.horarios_especificos:
                    # Condición 1: La hora programada es en el futuro (hoy)
                    if hora_programada >= hora_actual_str:
                        # Condición 2: Es la toma más temprana encontrada hasta ahora
                        if proxima_toma_hora is None or hora_programada < proxima_toma_hora:
                            proxima_toma_hora = hora_programada
                            farmaco_proximo = medicamento.nombre_farmaco

            # Evaluación final de la premisa
            if proxima_toma_hora:
                return f"Tu próxima toma es {farmaco_proximo} a las {proxima_toma_hora}."
            else:
                return "Has completado o sobrepasado todas tus tomas programadas para el día de hoy."

        case _:
            return "No reconozco la solicitud. Por favor, revisa tu agenda o consulta a tu cuidador."