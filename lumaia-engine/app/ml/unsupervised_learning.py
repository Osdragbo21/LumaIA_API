# [MOTOR IA] Ruta: app/ml/unsupervised_learning.py

import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

RUTA_GRAFICAS = "app/ml/graficas"
os.makedirs(RUTA_GRAFICAS, exist_ok=True)

# Variables globales para mantener el modelo en memoria
_modelo_kmeans = None
_modelo_pca = None
_escalador = None
_mapa_perfiles = {
    0: "Requiere Acompañamiento",
    1: "Compromiso Intermedio",
    2: "Alta Adherencia"
}

def graficar_codo_y_silueta(X_escalado):
    """ REFERENCIA: PRÁCTICA 7 (Método del Codo y Silhouette Score) """
    valores_k = range(2, 7)
    inercias = []
    siluetas = []

    for k in valores_k:
        modelo = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        etiquetas = modelo.fit_predict(X_escalado)
        inercias.append(modelo.inertia_)
        siluetas.append(silhouette_score(X_escalado, etiquetas))

    # 1. Gráfica del Codo
    plt.figure(figsize=(8, 5))
    plt.plot(list(valores_k), inercias, marker="o", linewidth=2, color="#3b82f6")
    plt.title("Método del Codo (K-Means)")
    plt.xlabel("Número de grupos (K)")
    plt.ylabel("Inercia (WCSS)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_GRAFICAS, "fase2_metodo_codo.png"), dpi=300)
    plt.close()

    # 2. Gráfica de Silueta
    plt.figure(figsize=(8, 5))
    plt.plot(list(valores_k), siluetas, marker="s", linewidth=2, color="#10b981")
    plt.title("Evaluación Silhouette (K-Means)")
    plt.xlabel("Número de grupos (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_GRAFICAS, "fase2_silhouette.png"), dpi=300)
    plt.close()

def graficar_pca_2d(datos_pca, etiquetas):
    """ REFERENCIA: PRÁCTICA 9 (Reducción de Dimensionalidad y Visualización) """
    plt.figure(figsize=(10, 7))
    
    # Colores para los 3 perfiles
    colores = ['#ef4444', '#f59e0b', '#10b981']
    
    for i in range(3):
        puntos = datos_pca[etiquetas == i]
        plt.scatter(puntos[:, 0], puntos[:, 1], label=_mapa_perfiles[i], color=colores[i], alpha=0.75, s=45)

    plt.title("Proyección PCA en 2D (Segmentación de Pacientes)")
    plt.xlabel("Componente Principal 1 (PC1)")
    plt.ylabel("Componente Principal 2 (PC2)")
    plt.axhline(y=0, linewidth=0.7, alpha=0.4, color="black")
    plt.axvline(x=0, linewidth=0.7, alpha=0.4, color="black")
    plt.legend(title="Perfil del Paciente")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_GRAFICAS, "fase2_pca_2d.png"), dpi=300)
    plt.close()

def entrenar_modelos_no_supervisados(datos_lista: list):
    """
    Entrena K-Means y PCA con los datos históricos de MongoDB.
    """
    global _modelo_kmeans, _modelo_pca, _escalador

    df = pd.DataFrame(datos_lista)
    X = df[["retraso_promedio_minutos", "consultas_sintomas", "alertas_sos"]]

    # --- REFERENCIA: PRÁCTICA 7 (Escalamiento y K-Means) ---
    _escalador = StandardScaler()
    X_escalado = _escalador.fit_transform(X)

    # Generamos gráficas para los desarrolladores
    graficar_codo_y_silueta(X_escalado)

    # Entrenamos el modelo final con K=3 (definido por negocio)
    _modelo_kmeans = KMeans(n_clusters=3, init="k-means++", n_init=20, random_state=42)
    etiquetas = _modelo_kmeans.fit_predict(X_escalado)

    # Ordenamos los centroides para que el clúster 0 sea el peor y el 2 el mejor
    # (Basado en el retraso promedio: a mayor retraso, peor perfil)
    centroides = _modelo_kmeans.cluster_centers_
    orden = np.argsort(centroides[:, 0])[::-1] # Orden descendente por retraso
    
    # Reasignamos las etiquetas para que coincidan con nuestro mapa
    etiquetas_ordenadas = np.zeros_like(etiquetas)
    for nuevo_id, viejo_id in enumerate(orden):
        etiquetas_ordenadas[etiquetas == viejo_id] = nuevo_id

    # --- REFERENCIA: PRÁCTICA 9 (PCA) ---
    _modelo_pca = PCA(n_components=2)
    datos_pca = _modelo_pca.fit_transform(X_escalado)

    # Generamos la gráfica 2D
    graficar_pca_2d(datos_pca, etiquetas_ordenadas)

    print("\n--- MODELOS NO SUPERVISADOS ENTRENADOS ---")
    print(f"Varianza explicada por PCA: {sum(_modelo_pca.explained_variance_ratio_) * 100:.2f}%")

def analizar_paciente_kmeans_pca(retraso: float, sintomas: int, sos: int) -> dict:
    """
    Recibe los datos de un paciente y devuelve su perfil (K-Means) y coordenadas (PCA).
    """
    if _modelo_kmeans is None or _modelo_pca is None:
        raise Exception("Los modelos no supervisados aún no han sido entrenados.")

    datos_reales = pd.DataFrame({
        "retraso_promedio_minutos": [retraso],
        "consultas_sintomas": [sintomas],
        "alertas_sos": [sos]
    })

    # Escalar datos
    datos_escalados = _escalador.transform(datos_reales)

    # 1. K-Means (Perfil)
    cluster_crudo = _modelo_kmeans.predict(datos_escalados)[0]
    
    # Ajustar al orden correcto (Peor a Mejor)
    centroides = _modelo_kmeans.cluster_centers_
    orden = np.argsort(centroides[:, 0])[::-1]
    cluster_real = int(np.where(orden == cluster_crudo)[0][0])
    
    perfil_texto = _mapa_perfiles[cluster_real]

    # 2. PCA (Coordenadas)
    coordenadas = _modelo_pca.transform(datos_escalados)[0]

    # 3. Detección de Anomalías (Percentil 95 - Distancia al centroide)
    # REFERENCIA: PRÁCTICA 8 (Análisis Avanzado y Atípicos)
    centroide_asignado = centroides[cluster_crudo]
    distancia_euclidiana = np.linalg.norm(datos_escalados[0] - centroide_asignado)
    es_atipico = bool(distancia_euclidiana > 2.5) # Umbral estático para el MVP (Z-Score > 2.5)

    return {
        "cluster_id": cluster_real,
        "perfil_academico": perfil_texto,
        "pca_x": float(coordenadas[0]),
        "pca_y": float(coordenadas[1]),
        "es_anomalia": es_atipico
    }