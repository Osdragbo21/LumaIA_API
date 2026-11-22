# [MOTOR IA] Ruta: app/ml/supervised_learning.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# Variables globales para mantener el mejor modelo en memoria (Singleton)
_mejor_modelo = None
_nombre_mejor_modelo = ""
_metricas_modelo = {}

RUTA_CSV = "datos_historicos_pacientes.csv"

def entrenar_modelos_supervisados(datos_lista: list) -> dict:
    """
    Recibe los datos de MongoDB, compara Decision Tree, Random Forest y KNN,
    optimiza sus hiperparámetros y selecciona el mejor.
    """
    global _mejor_modelo, _nombre_mejor_modelo, _metricas_modelo

    # --- REFERENCIA: PRÁCTICA 2 (Creación de Dataset y Exportación) ---
    df = pd.DataFrame(datos_lista)
    df.to_csv(RUTA_CSV, index=False, encoding="utf-8")
    print(f"Archivo {RUTA_CSV} generado con {len(df)} registros.")

    # --- REFERENCIA: PRÁCTICA 4 (Separación de Variables y Train/Test) ---
    X = df[["retraso_promedio_minutos", "consultas_sintomas", "alertas_sos"]]
    y = df["riesgo_abandono"]

    # División 80/20 estratificada para mantener proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # --- REFERENCIA: PRÁCTICA 6 (Validación Cruzada Estratificada) ---
    # Usamos 3 particiones (n_splits=3) porque nuestro dataset de prueba es pequeño (100 registros)
    validacion_cruzada = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # --- REFERENCIA: PRÁCTICA 5 (Definición de Algoritmos) ---
    modelos = {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [2, 3, 5, None], "criterion": ["gini", "entropy"]}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {"n_estimators": [50, 100], "max_depth": [3, 5, None]}
        ),
        "KNN": (
            # REFERENCIA: PRÁCTICA 8/9 (StandardScaler en Pipeline para evitar fuga de datos)
            Pipeline([("escalador", StandardScaler()), ("clasificador", KNeighborsClassifier())]),
            {"clasificador__n_neighbors": [3, 5, 7], "clasificador__weights": ["uniform", "distance"]}
        )
    }

    mejor_f1_global = -1
    mejor_modelo_global = None
    nombre_ganador = ""

    print("\n--- INICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS ---")
    
    # --- REFERENCIA: PRÁCTICA 6 (GridSearchCV) ---
    for nombre, (modelo, parametros) in modelos.items():
        busqueda = GridSearchCV(
            estimator=modelo,
            param_grid=parametros,
            scoring="f1_macro", # Usamos F1 Macro para balancear aciertos en ambas clases
            cv=validacion_cruzada,
            n_jobs=-1
        )
        
        # Entrenamos la búsqueda con el 80% de los datos
        busqueda.fit(X_train, y_train)
        
        print(f"Modelo: {nombre} | Mejor F1 (CV): {busqueda.best_score_:.4f}")

        # Comparamos para encontrar el campeón absoluto
        if busqueda.best_score_ > mejor_f1_global:
            mejor_f1_global = busqueda.best_score_
            mejor_modelo_global = busqueda.best_estimator_
            nombre_ganador = nombre

    # --- REFERENCIA: PRÁCTICA 4 (Evaluación Final con Datos de Prueba) ---
    # Evaluamos el modelo ganador con el 20% de datos que NUNCA ha visto
    predicciones_test = mejor_modelo_global.predict(X_test)
    
    exactitud_final = accuracy_score(y_test, predicciones_test)
    f1_final = f1_score(y_test, predicciones_test, average="macro", zero_division=0)

    # Guardamos el campeón en memoria
    _mejor_modelo = mejor_modelo_global
    _nombre_mejor_modelo = nombre_ganador
    _metricas_modelo = {
        "accuracy": round(exactitud_final * 100, 2),
        "f1_score": round(f1_final * 100, 2)
    }

    print(f"\n🏆 CAMPEÓN: {_nombre_mejor_modelo} (Exactitud: {_metricas_modelo['accuracy']}%)")
    
    return {
        "algoritmo": _nombre_mejor_modelo,
        "exactitud": _metricas_modelo["accuracy"],
        "f1_score": _metricas_modelo["f1_score"]
    }

def predecir_riesgo_ml(retraso: float, sintomas: int, sos: int) -> dict:
    """
    Recibe datos REALES de MongoDB y devuelve la predicción del MEJOR modelo.
    """
    if _mejor_modelo is None:
        raise Exception("El modelo aún no ha sido entrenado con datos de MongoDB.")

    # Formateamos los datos reales para el modelo
    datos_reales = pd.DataFrame({
        "retraso_promedio_minutos": [retraso],
        "consultas_sintomas": [sintomas],
        "alertas_sos": [sos]
    })

    # Generamos la predicción
    resultado = _mejor_modelo.predict(datos_reales)[0]
    
    return {
        "tiene_riesgo": bool(resultado == 1),
        "exactitud_modelo": _metricas_modelo["accuracy"],
        "f1_score_modelo": _metricas_modelo["f1_score"],
        "algoritmo": _nombre_mejor_modelo
    }

def predecir_riesgo_regla_fija(retraso: float, sintomas: int, sos: int) -> bool:
    """
    REFERENCIA: PRÁCTICA 1 (Regla Fija vs Aprendizaje Automático)
    """
    if retraso >= 40 or sintomas >= 2 or sos >= 1:
        return True
    return False