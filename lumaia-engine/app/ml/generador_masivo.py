# [MOTOR IA] Ruta: app/ml/generador_masivo.py

import os
import numpy as np
import pandas as pd

RUTA_CSV_MASIVO = "dataset_masivo.csv"

def generar_big_data():
    """
    REFERENCIA: PRÁCTICA 24 (Generador de datos sintéticos masivos).
    Genera 1,000,000 de registros con distribuciones estadísticas.
    """
    print("\n--- INICIANDO GENERACIÓN DE BIG DATA ---")
    cantidad = 1000000

    # Generamos datos con distribuciones normales y uniformes
    retraso = np.random.normal(loc=25, scale=15, size=cantidad)
    retraso = np.clip(retraso, 0, 120) # Limitamos entre 0 y 120 min

    sintomas = np.random.poisson(lam=1.5, size=cantidad)
    sintomas = np.clip(sintomas, 0, 10)

    sos = np.random.choice([0, 1, 2], size=cantidad, p=[0.85, 0.10, 0.05])

    # Regla base de riesgo
    riesgo = np.where((retraso > 40) | (sintomas >= 2) | (sos > 0), 1, 0)
    
    # Inyectamos 5% de ruido para que el ML aprenda patrones complejos
    ruido = np.random.rand(cantidad)
    riesgo = np.where(ruido > 0.95, 1 - riesgo, riesgo)

    df = pd.DataFrame({
        "retraso_promedio_minutos": retraso,
        "consultas_sintomas": sintomas,
        "alertas_sos": sos,
        "riesgo_abandono": riesgo
    })

    # --- REFERENCIA: PRÁCTICA 3 (Preparación para Limpieza) ---
    # Inyectamos intencionalmente valores Nulos (NaN) y Duplicados
    # para que nuestros scripts de ML tengan algo que limpiar.
    print("Inyectando valores nulos y duplicados intencionales...")
    indices_nulos = np.random.choice(df.index, size=5000, replace=False)
    df.loc[indices_nulos, 'retraso_promedio_minutos'] = np.nan

    # Duplicamos las primeras 10,000 filas y las pegamos al final
    df = pd.concat([df, df.head(10000)], ignore_index=True)

    # Guardamos el CSV masivo
    df.to_csv(RUTA_CSV_MASIVO, index=False)
    print(f"✅ Big Data generado exitosamente: {RUTA_CSV_MASIVO} ({len(df)} registros).")

if __name__ == "__main__":
    generar_big_data()