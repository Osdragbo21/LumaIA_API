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

RUTA_CSV_MASIVO = "dataset_masivo.csv"
RUTA_GRAFICAS = "app/ml/graficas"
os.makedirs(RUTA_GRAFICAS, exist_ok=True)

_modelo_kmeans = None
_modelo_pca = None
_escalador = None
_mapa_perfiles = {0: "Requiere Acompañamiento", 1: "Compromiso Intermedio", 2: "Alta Adherencia"}

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

def entrenar_modelos_no_supervisados():
    global _modelo_kmeans, _modelo_pca, _escalador

    if not os.path.exists(RUTA_CSV_MASIVO):
        raise FileNotFoundError(f"No se encontró {RUTA_CSV_MASIVO}.")

    df = pd.read_csv(RUTA_CSV_MASIVO)
    
    # --- PRÁCTICA 3: Limpieza ---
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    X = df[["retraso_promedio_minutos", "consultas_sintomas", "alertas_sos"]]
    
    # Muestra de 10,000 para evitar crashear la RAM calculando la Silueta
    X_sample = X.sample(n=10000, random_state=42)

    _escalador = StandardScaler()
    X_escalado_sample = _escalador.fit_transform(X_sample)

    graficar_codo_y_silueta(X_escalado_sample)

    # Entrenamos el modelo final con TODOS los datos limpios
    X_escalado_full = _escalador.transform(X)
    _modelo_kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
    etiquetas_full = _modelo_kmeans.fit_predict(X_escalado_full)

    centroides = _modelo_kmeans.cluster_centers_
    orden = np.argsort(centroides[:, 0])[::-1] 
    
    # PCA para la gráfica 2D (usamos la muestra para que la gráfica no sea un bloque sólido de 1M de puntos)
    _modelo_pca = PCA(n_components=2)
    datos_pca_sample = _modelo_pca.fit_transform(X_escalado_sample)
    
    etiquetas_sample = _modelo_kmeans.predict(X_escalado_sample)
    etiquetas_ordenadas_sample = np.zeros_like(etiquetas_sample)
    for nuevo_id, viejo_id in enumerate(orden):
        etiquetas_ordenadas_sample[etiquetas_sample == viejo_id] = nuevo_id

    graficar_pca_2d(datos_pca_sample, etiquetas_ordenadas_sample)

    print("\n--- MODELOS NO SUPERVISADOS ENTRENADOS ---")
    print(f"Varianza explicada por PCA: {sum(_modelo_pca.explained_variance_ratio_) * 100:.2f}%")

def analizar_paciente_kmeans_pca(retraso: float, sintomas: int, sos: int) -> dict:
    if _modelo_kmeans is None or _modelo_pca is None:
        raise Exception("Los modelos no supervisados aún no han sido entrenados.")

    datos_reales = pd.DataFrame({"retraso_promedio_minutos": [retraso], "consultas_sintomas": [sintomas], "alertas_sos": [sos]})
    datos_escalados = _escalador.transform(datos_reales)

    cluster_crudo = _modelo_kmeans.predict(datos_escalados)[0]
    centroides = _modelo_kmeans.cluster_centers_
    orden = np.argsort(centroides[:, 0])[::-1]
    cluster_real = int(np.where(orden == cluster_crudo)[0][0])
    perfil_texto = _mapa_perfiles[cluster_real]

    coordenadas = _modelo_pca.transform(datos_escalados)[0]
    centroide_asignado = centroides[cluster_crudo]
    distancia_euclidiana = np.linalg.norm(datos_escalados[0] - centroide_asignado)
    es_atipico = bool(distancia_euclidiana > 2.5) 

    return {
        "cluster_id": cluster_real,
        "perfil_academico": perfil_texto,
        "pca_x": float(coordenadas[0]),
        "pca_y": float(coordenadas[1]),
        "es_anomalia": es_atipico
    }