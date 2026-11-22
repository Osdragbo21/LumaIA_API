# [MOTOR IA] Ruta: app/ml/risk_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

_modelo_arbol = None
_exactitud_modelo = 0.0
RUTA_CSV = "datos_historicos_pacientes.csv"

def entrenar_modelo_con_datos_reales(datos_lista: list) -> float:
    """
    Recibe los datos de MongoDB, crea el CSV y entrena el modelo 80/20.
    """
    global _modelo_arbol, _exactitud_modelo

    # 1. Convertimos los datos de MongoDB a un DataFrame de Pandas
    df = pd.DataFrame(datos_lista)

    # 2. Generamos el archivo CSV físicamente (Práctica 2)
    df.to_csv(RUTA_CSV, index=False, encoding="utf-8")
    print(f"Archivo {RUTA_CSV} generado con {len(df)} registros reales de MongoDB.")

    # 3. SEPARACIÓN DE VARIABLES (X e y)
    X = df[["retraso_promedio_minutos", "consultas_sintomas", "alertas_sos"]]
    y = df["riesgo_abandono"]

    # 4. DIVISIÓN DE DATOS: 80% Entrenamiento, 20% Prueba (test_size=0.20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 5. ENTRENAMIENTO DEL MODELO (Árbol de Decisión)
    _modelo_arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
    _modelo_arbol.fit(X_train, y_train)

    # 6. EVALUACIÓN (Exactitud)
    predicciones = _modelo_arbol.predict(X_test)
    _exactitud_modelo = accuracy_score(y_test, predicciones)
    
    return round(_exactitud_modelo * 100, 2)

def predecir_riesgo_ml(retraso: float, sintomas: int, sos: int) -> dict:
    if _modelo_arbol is None:
        raise Exception("El modelo aún no ha sido entrenado con datos de MongoDB.")

    datos_reales = pd.DataFrame({
        "retraso_promedio_minutos": [retraso],
        "consultas_sintomas": [sintomas],
        "alertas_sos": [sos]
    })

    resultado = _modelo_arbol.predict(datos_reales)[0]
    return {
        "tiene_riesgo": bool(resultado == 1),
        "exactitud_modelo": round(_exactitud_modelo * 100, 2)
    }

def predecir_riesgo_regla_fija(retraso: float, sintomas: int, sos: int) -> bool:
    if retraso >= 40 or sintomas >= 2 or sos >= 1:
        return True
    return False