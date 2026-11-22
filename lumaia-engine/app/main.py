# [MOTOR IA] Ruta: app/main.py

from fastapi import FastAPI, HTTPException
from app.models.schemas import EvaluacionRequest, AudioRequest, PrediccionRiesgoRequest, PrediccionRiesgoResponse, EntrenamientoRequest
from app.ml.supervised_learning import predecir_riesgo_ml, predecir_riesgo_regla_fija, entrenar_modelos_supervisados
from app.ml.unsupervised_learning import entrenar_modelos_no_supervisados, analizar_paciente_kmeans_pca

app = FastAPI(title="LumaIA Engine API")

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


@app.post("/transcribir")
async def transcribir_audio(req: AudioRequest):
    try:
        texto_simulado = "Esta es una transcripción de prueba. El motor de voz está conectado correctamente al backend."
        return {"texto": texto_simulado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =================================================================
# ENDPOINTS DE MACHINE LEARNING
# =================================================================

@app.post("/entrenar")
async def entrenar_ia(req: EntrenamientoRequest):
    try:
        datos_dict = [paciente.dict() for paciente in req.datos_historicos]
        
        # Entrenamos y obtenemos las métricas del modelo ganador
        resultados = entrenar_modelos_supervisados(datos_dict)

        # Entrenamos No Supervisado (K-Means, PCA)
        entrenar_modelos_no_supervisados(datos_dict)
        
        return {
            "mensaje": f"Modelos evaluados exitosamente. El ganador fue {resultados['algoritmo']}.", 
            "exactitud": resultados['exactitud']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predecir-riesgo", response_model=PrediccionRiesgoResponse)
async def predecir_riesgo_abandono(req: PrediccionRiesgoRequest):
    try:
        # 1. Predicción con Regla Fija
        riesgo_regla = predecir_riesgo_regla_fija(
            req.retraso_promedio_minutos, 
            req.consultas_sintomas, 
            req.alertas_sos
        )

        # 2. Predicción con el Mejor Modelo de ML
        resultado_ml = predecir_riesgo_ml(
            req.retraso_promedio_minutos, 
            req.consultas_sintomas, 
            req.alertas_sos
        )

        # Obtenemos el perfil y las coordenadas PCA
        resultado_no_sup = analizar_paciente_kmeans_pca(req.retraso_promedio_minutos, req.consultas_sintomas, req.alertas_sos)

        # 3. Retornamos la comparación ampliada
        return PrediccionRiesgoResponse(
            riesgo_regla_fija=riesgo_regla,
            riesgo_machine_learning=resultado_ml["tiene_riesgo"],
            exactitud_modelo_ml=resultado_ml["exactitud_modelo"],
            f1_score_ml=resultado_ml["f1_score_modelo"],
            algoritmo_utilizado=resultado_ml["algoritmo"],
            perfil_paciente=resultado_no_sup["perfil_academico"],
            coordenada_pca_x=resultado_no_sup["pca_x"],
            coordenada_pca_y=resultado_no_sup["pca_y"],
            es_comportamiento_atipico=resultado_no_sup["es_anomalia"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
