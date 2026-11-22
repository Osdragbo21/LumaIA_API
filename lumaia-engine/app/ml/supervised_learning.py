# [MOTOR IA] Ruta: app/ml/supervised_learning.py

import os
import pandas as pd
import numpy as np

# Configuración crucial para servidores web: usar backend 'Agg' para no abrir ventanas GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Variables globales
_mejor_modelo = None
_nombre_mejor_modelo = ""
_metricas_modelo = {}

RUTA_CSV = "datos_historicos_pacientes.csv"
# NUEVO: Carpeta para guardar las gráficas de los desarrolladores
RUTA_GRAFICAS = "app/ml/graficas"
os.makedirs(RUTA_GRAFICAS, exist_ok=True)

def generar_grafica_comparativa(resultados_modelos: dict):
    """ Genera una gráfica de barras comparando el F1-Score de los algoritmos """
    nombres = list(resultados_modelos.keys())
    f1_scores = [resultados_modelos[nombre] for nombre in nombres]

    plt.figure(figsize=(10, 6))
    barras = plt.bar(nombres, f1_scores, color=['#3b82f6', '#10b981', '#f59e0b'])
    
    plt.ylim(0, 1.05)
    plt.ylabel("F1-Score (Macro)")
    plt.xlabel("Algoritmo")
    plt.title("Comparación de Modelos de Clasificación (Riesgo de Abandono)")
    plt.grid(axis="y", alpha=0.3)

    # Agregar el valor numérico sobre cada barra
    for barra in barras:
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2, yval + 0.01, f"{yval:.2%}", ha='center', va='bottom')

    plt.tight_layout()
    ruta_img = os.path.join(RUTA_GRAFICAS, "fase1_comparacion_modelos.png")
    plt.savefig(ruta_img, dpi=300)
    plt.close() # CRÍTICO: Liberar memoria
    print(f"Gráfica guardada: {ruta_img}")

def generar_matriz_confusion(y_real, y_pred, nombre_modelo):
    """ Genera la matriz de confusión del modelo ganador """
    matriz = confusion_matrix(y_real, y_pred, labels=[0, 1])
    
    visualizacion = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["Adherente (0)", "En Riesgo (1)"])
    visualizacion.plot(cmap="Blues", values_format="d")
    
    plt.title(f"Matriz de Confusión - {nombre_modelo}")
    plt.tight_layout()
    
    ruta_img = os.path.join(RUTA_GRAFICAS, "fase1_matriz_confusion_campeon.png")
    plt.savefig(ruta_img, dpi=300)
    plt.close()
    print(f"Gráfica guardada: {ruta_img}")

def entrenar_modelos_supervisados(datos_lista: list) -> dict:
    global _mejor_modelo, _nombre_mejor_modelo, _metricas_modelo

    df = pd.DataFrame(datos_lista)
    df.to_csv(RUTA_CSV, index=False, encoding="utf-8")

    X = df[["retraso_promedio_minutos", "consultas_sintomas", "alertas_sos"]]
    y = df["riesgo_abandono"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    validacion_cruzada = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    modelos = {
        "Decision Tree": (DecisionTreeClassifier(random_state=42), {"max_depth": [2, 3, 5, None], "criterion": ["gini", "entropy"]}),
        "Random Forest": (RandomForestClassifier(random_state=42, n_jobs=-1), {"n_estimators": [50, 100], "max_depth": [3, 5, None]}),
        "KNN": (Pipeline([("escalador", StandardScaler()), ("clasificador", KNeighborsClassifier())]), {"clasificador__n_neighbors": [3, 5, 7], "clasificador__weights": ["uniform", "distance"]})
    }

    mejor_f1_global = -1
    mejor_modelo_global = None
    nombre_ganador = ""
    
    # Diccionario para guardar los resultados y graficarlos
    resultados_para_graficar = {}

    print("\n--- INICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS ---")
    
    for nombre, (modelo, parametros) in modelos.items():
        busqueda = GridSearchCV(estimator=modelo, param_grid=parametros, scoring="f1_macro", cv=validacion_cruzada, n_jobs=-1)
        busqueda.fit(X_train, y_train)
        
        f1_obtenido = busqueda.best_score_
        resultados_para_graficar[nombre] = f1_obtenido
        print(f"Modelo: {nombre} | Mejor F1 (CV): {f1_obtenido:.4f}")

        if f1_obtenido > mejor_f1_global:
            mejor_f1_global = f1_obtenido
            mejor_modelo_global = busqueda.best_estimator_
            nombre_ganador = nombre

    # Evaluamos el modelo ganador con el conjunto de prueba
    predicciones_test = mejor_modelo_global.predict(X_test)
    exactitud_final = accuracy_score(y_test, predicciones_test)
    f1_final = f1_score(y_test, predicciones_test, average="macro", zero_division=0)

    # --- GENERACIÓN DE GRÁFICAS PARA DESARROLLADORES ---
    generar_grafica_comparativa(resultados_para_graficar)
    generar_matriz_confusion(y_test, predicciones_test, nombre_ganador)

    _mejor_modelo = mejor_modelo_global
    _nombre_mejor_modelo = nombre_ganador
    _metricas_modelo = {"accuracy": round(exactitud_final * 100, 2), "f1_score": round(f1_final * 100, 2)}

    print(f"\n🏆 CAMPEÓN: {_nombre_mejor_modelo} (Exactitud: {_metricas_modelo['accuracy']}%)")
    
    return {
        "algoritmo": _nombre_mejor_modelo,
        "exactitud": _metricas_modelo["accuracy"],
        "f1_score": _metricas_modelo["f1_score"]
    }

def predecir_riesgo_ml(retraso: float, sintomas: int, sos: int) -> dict:
    if _mejor_modelo is None:
        raise Exception("El modelo aún no ha sido entrenado con datos de MongoDB.")

    datos_reales = pd.DataFrame({"retraso_promedio_minutos": [retraso], "consultas_sintomas": [sintomas], "alertas_sos": [sos]})
    resultado = _mejor_modelo.predict(datos_reales)[0]
    
    return {
        "tiene_riesgo": bool(resultado == 1),
        "exactitud_modelo": _metricas_modelo["accuracy"],
        "f1_score_modelo": _metricas_modelo["f1_score"],
        "algoritmo": _nombre_mejor_modelo
    }

def predecir_riesgo_regla_fija(retraso: float, sintomas: int, sos: int) -> bool:
    if retraso >= 40 or sintomas >= 2 or sos >= 1:
        return True
    return False