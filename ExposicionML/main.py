import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform

# --------------------
# 1. Funciones auxiliares
# --------------------

def load_data(uploaded_file=None, url=None):
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
        else:
            df = pd.read_csv(url, sep=';', encoding='utf-8-sig')
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None


def estimate_eps(X, k=5):
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)
    d_k = np.sort(distances[:, k-1])
    first_derivative = np.gradient(d_k)
    second_derivative = np.gradient(first_derivative)
    eps_est = d_k[np.argmax(second_derivative)]
    return eps_est, d_k


def create_pipeline(numeric_feats, categorical_feats):
    transformers = []
    if numeric_feats:
        transformers.append(('num', StandardScaler(), numeric_feats))
    if categorical_feats:
        transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_feats))
    return ColumnTransformer(transformers, remainder='drop')


# Nueva función para preparar datos mixtos con enfoque personalizado
def prepare_mixed_data(df, numeric_cols, categorical_cols, encoding_method="onehot"):
    """
    Prepara datos mixtos (numéricos y categóricos) para clustering.
    
    Parameters:
        df: DataFrame con los datos
        numeric_cols: Lista de columnas numéricas
        categorical_cols: Lista de columnas categóricas
        encoding_method: Método de codificación ('onehot', 'label', 'ordinal')
    """
    df_processed = df.copy()
    
    # Procesar columnas numéricas
    if numeric_cols:
        scaler = StandardScaler()
        df_processed[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Procesar columnas categóricas según el método elegido
    if categorical_cols:
        if encoding_method == "onehot":
            # One-hot encoding con balance de pesos
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_cats = encoder.fit_transform(df[categorical_cols])
            
            # Crear columnas para el encoding
            encoded_cols = []
            for i, col in enumerate(categorical_cols):
                categories = encoder.categories_[i][1:]  # Omitir el primero por drop='first'
                for cat in categories:
                    encoded_cols.append(f"{col}_{cat}")
            
            # Añadir al DataFrame procesado con peso ajustado
            encoded_df = pd.DataFrame(encoded_cats * (1/np.sqrt(len(encoded_cols))), columns=encoded_cols)
            df_processed = pd.concat([df_processed[numeric_cols], encoded_df], axis=1)
            
        elif encoding_method == "label":
            # Label encoding
            for col in categorical_cols:
                df_processed[col] = df[col].astype('category').cat.codes
        
        elif encoding_method == "ordinal":
            # Ordinal encoding con categorías ordenadas
            for col in categorical_cols:
                categories = df[col].value_counts().index.tolist()
                mapping = {cat: i for i, cat in enumerate(categories)}
                df_processed[col] = df[col].map(mapping)
                
    return df_processed


# Nueva función para calcular matriz de distancia con métrica personalizada para datos mixtos
def mixed_distance_matrix(df, numeric_cols, categorical_cols):
    """
    Calcula una matriz de distancia personalizada para datos mixtos.
    Combina la distancia euclidiana para variables numéricas y
    la distancia de Gower para variables categóricas.
    """
    # Extraer columnas numéricas y categóricas
    X_num = df[numeric_cols].values if numeric_cols else None
    X_cat = df[categorical_cols].values if categorical_cols else None
    
    n_samples = len(df)
    dist_matrix = np.zeros((n_samples, n_samples))
    
    # Calcular distancias numéricas (normalizadas)
    if X_num is not None and X_num.shape[1] > 0:
        # Normalizar datos numéricos
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
        # Distancia euclidiana para variables numéricas
        num_dist = squareform(pdist(X_num_scaled, metric='euclidean'))
        dist_matrix += num_dist
    
    # Calcular distancias categóricas
    if X_cat is not None and X_cat.shape[1] > 0:
        for j in range(X_cat.shape[1]):
            # Crear matriz de coincidencia (1 si son diferentes, 0 si son iguales)
            cat_dist = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for k in range(n_samples):
                    cat_dist[i, k] = 0 if X_cat[i, j] == X_cat[k, j] else 1
            
            # Agregar a la matriz de distancia total
            dist_matrix += cat_dist
    
    # Normalizar por el número total de variables
    total_vars = (len(numeric_cols) if numeric_cols else 0) + (len(categorical_cols) if categorical_cols else 0)
    if total_vars > 0:
        dist_matrix /= total_vars
        
    return dist_matrix

# --------------------
# 2. Streamlit config
# --------------------

st.set_page_config(page_title="DBSCAN Clustering Explorer", layout="wide")
st.title("🔍 Explorador de Clustering con DBSCAN")

# --------------------
# 3. Carga de datos
# --------------------
with st.sidebar.expander("Carga de datos", expanded=True):
    url_csv = st.text_input("URL del CSV", value="https://raw.githubusercontent.com/Emma-Ok/BootcampTalentoTech/main/beneficiarios.csv")
    uploaded_file = st.file_uploader("O sube un archivo CSV", type=['csv'])
    df = load_data(uploaded_file, url_csv)

if df is None:
    st.stop()

# --------------------
# 4. Exploración básica
# --------------------
st.subheader("Vista previa y descripción del dataset")
st.dataframe(df.head())
col1, col2 = st.columns(2)
with col1:
    st.write(f"Filas: {df.shape[0]}")
    st.write(f"Columnas: {df.shape[1]}")
with col2:
    buf = StringIO()
    df.info(buf=buf)
    st.text(buf.getvalue())

# Añadir análisis exploratorio básico
st.subheader("Análisis exploratorio básico")
col1, col2 = st.columns(2)

with col1:
    # Distribución de edades
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df['EDAD'], kde=True, ax=ax)
    ax.set_title('Distribución de edades')
    st.pyplot(fig)

with col2:
    # Distribución de plataformas educativas
    fig, ax = plt.subplots(figsize=(8,4))
    plataformas = df['PLATAFORMA_EDUCATIVA'].value_counts()
    plataformas.plot(kind='bar', ax=ax)
    ax.set_title('Distribución de plataformas educativas')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# --------------------
# 5. Selección de características y preprocesamiento
# --------------------
st.sidebar.subheader("Preprocesamiento")

# Método de codificación para variables categóricas
encoding_method = st.sidebar.selectbox(
    "Método de codificación categórica",
    ["onehot", "label", "ordinal"],
    format_func=lambda x: {
        "onehot": "One-Hot Encoding", 
        "label": "Label Encoding", 
        "ordinal": "Ordinal Encoding"
    }[x]
)

# Selección de distancia
distance_metric = st.sidebar.selectbox(
    "Métrica de distancia",
    ["euclidean", "mixed"],
    format_func=lambda x: {
        "euclidean": "Euclidiana estándar",
        "mixed": "Personalizada para datos mixtos"
    }[x]
)

# Selección de características
st.sidebar.subheader("Selección de características")
cols = df.columns.tolist()
selected = st.sidebar.multiselect("Elige columnas para clustering", cols,
                                  default=[c for c in cols if df[c].dtype in ['int64','float64']][:1] +
                                          [c for c in cols if df[c].dtype=='object'][:1])

if len(selected) < 1:
    st.warning("Selecciona al menos una característica.")
    st.stop()

numeric = [c for c in selected if df[c].dtype in ['int64','float64']]
categorical = [c for c in selected if df[c].dtype=='object']

# --------------------
# 6. Preprocesamiento de datos
# --------------------
st.subheader("Preprocesamiento de datos")

# Si solo tenemos una columna numérica, crear una sintética para visualización
if len(numeric) == 1 and len(selected) == 1:
    st.warning("Solo has seleccionado una variable numérica. Añadiremos ruido aleatorio para visualizar mejor.")
    # Añadir variable sintética con pequeño ruido aleatorio
    df["EDAD_NOISE"] = df[numeric[0]] + np.random.normal(0, 0.5, size=len(df))
    numeric.append("EDAD_NOISE")
    selected.append("EDAD_NOISE")

# Preparar datos según el método seleccionado
if distance_metric == "mixed":
    st.info("Usando métrica de distancia personalizada para datos mixtos.")
    # Calcular matriz de distancia personalizada
    dist_matrix = mixed_distance_matrix(df, numeric, categorical)
    # No se necesita X_scaled para DBSCAN cuando proporcionamos matriz de distancia
    X_scaled = None
    # Para visualización PCA necesitamos datos numéricos
    processed_df = prepare_mixed_data(df, numeric, categorical, encoding_method)
    coords = PCA(n_components=2).fit_transform(processed_df)
else:
    st.info(f"Usando codificación {encoding_method} y distancia euclidiana estándar.")
    # Preprocesar con pipeline estándar
    preprocessor = create_pipeline(numeric, categorical)
    X_scaled = preprocessor.fit_transform(df[selected])
    dist_matrix = None
    # PCA para visualización
    coords = PCA(n_components=2).fit_transform(X_scaled)

# --------------------
# 7. k-dist plot con análisis mejorado
# --------------------
st.subheader("k-distance plot para estimar eps (mejorado)")
k_values = st.select_slider("k vecinos (para k-dist)", options=list(range(2, 21)), value=5)

if distance_metric == "mixed":
    # Usar la matriz de distancia personalizada para el k-dist plot
    neigh = NearestNeighbors(n_neighbors=k_values, metric='precomputed')
    nbrs = neigh.fit(dist_matrix)
    distances, _ = nbrs.kneighbors(dist_matrix)
    d_k = np.sort(distances[:, k_values-1])
else:
    # Cálculo estándar de k-dist
    eps_auto, d_k = estimate_eps(X_scaled, k_values)

# Visualización mejorada del k-dist plot
fig_k, ax_k = plt.subplots(figsize=(10,5))
ax_k.plot(np.sort(d_k), linewidth=2)
ax_k.set_xlabel('Índice ordenado', fontsize=12)
ax_k.set_ylabel(f'Distancia al {k_values}-ésimo vecino', fontsize=12)

# Análisis de "codo" para sugerir valores de eps
# Calcular la segunda derivada para encontrar puntos de inflexión
x = np.arange(len(d_k))
y = np.sort(d_k)
first_derivative = np.gradient(y)
second_derivative = np.gradient(first_derivative)

# Encontrar potenciales valores de eps basados en la segunda derivada
potential_eps_indices = np.argsort(second_derivative)[-5:]  # Top 5 puntos de inflexión
potential_eps_values = [d_k[i] for i in sorted(potential_eps_indices)]

# Marcar potenciales valores de eps en el gráfico
for i, eps_val in enumerate(potential_eps_values):
    ax_k.axhline(eps_val, color=f'C{i+1}', ls='--', 
                label=f'Sugerencia {i+1}: eps≈{eps_val:.2f}')

ax_k.legend(fontsize=10)
st.pyplot(fig_k)

# Mostrar tabla con sugerencias de eps
eps_suggestions_df = pd.DataFrame({
    'Sugerencia': [f"Opción {i+1}" for i in range(len(potential_eps_values))],
    'Valor eps': [round(eps, 3) for eps in potential_eps_values]
})
st.table(eps_suggestions_df)

# --------------------
# 8. Parámetros DBSCAN
# --------------------
st.sidebar.subheader("Parámetros DBSCAN")

# Si tenemos sugerencias de eps, usar la primera como valor por defecto
default_eps = potential_eps_values[0] if potential_eps_values else 0.5

eps = st.sidebar.slider("Epsilon (eps)", 0.01, 2.0, default_eps, 0.01)
min_s = st.sidebar.slider("min_samples", 2, 100, 5, 1)

# --------------------
# 9. Clustering y métricas
# --------------------
if st.sidebar.button("Ejecutar DBSCAN", use_container_width=True):
    with st.spinner("Ejecutando DBSCAN..."):
        # Configurar DBSCAN según el tipo de distancia
        if distance_metric == "mixed":
            db = DBSCAN(eps=eps, min_samples=min_s, metric='precomputed')
            labels = db.fit_predict(dist_matrix)
        else:
            db = DBSCAN(eps=eps, min_samples=min_s)
            labels = db.fit_predict(X_scaled)
        
        # Añadir etiquetas al dataframe
        df_res = df.copy()
        df_res['Cluster'] = labels
        
        # Calcular métricas
        n_clust = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        st.success("Clustering completado!")
        
        # Resultados principales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Número de clusters", n_clust)
        with col2:
            st.metric("Puntos de ruido", n_noise)
        with col3:
            st.metric("Puntos clasificados", len(labels) - n_noise)
        
        # Validar número de clusters
        if n_clust < 2:
            st.warning("""
            ### ⚠️ Se formó solo un cluster
            
            Esto puede deberse a:
            1. El valor de eps es demasiado grande
            2. Las características seleccionadas no muestran agrupaciones claras
            3. Los datos podrían requerir otra transformación
            
            **Sugerencia:** Prueba los siguientes ajustes:
            - Reduce el valor de eps (intenta con valores más pequeños)
            - Aumenta min_samples para mayor robustez
            - Selecciona diferentes combinaciones de características
            - Prueba otro método de codificación para las variables categóricas
            """)
        
        # Silhouette si procede
        if n_clust > 1:
            # Para calcular silhouette necesitamos datos numéricos
            if distance_metric == "mixed":
                # Usamos los datos procesados para PCA
                score = silhouette_score(processed_df, labels)
            else:
                score = silhouette_score(X_scaled, labels)
            
            # Interpretar el silhouette score
            interpretation = ""
            if score > 0.7:
                interpretation = "👍 Muy buena separación de clusters"
            elif score > 0.5:
                interpretation = "✅ Buena separación de clusters"
            elif score > 0.25:
                interpretation = "⚠️ Clusters con cierta superposición"
            else:
                interpretation = "❗ Clusters muy superpuestos o mal formados"
                
            st.write(f"Silhouette score: {score:.3f} - {interpretation}")
        
        # Resumen de clusters
        summary = df_res['Cluster'].value_counts().rename_axis('Cluster').reset_index(name='Cantidad')
        st.subheader("Resumen de clusters")
        
        # Mostrar tabla y gráfico de distribución de clusters
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(summary)
        with col2:
            fig_dist = plt.figure(figsize=(8, 4))
            ax = fig_dist.add_subplot(111)
            sns.barplot(data=summary, x='Cluster', y='Cantidad', palette='viridis', ax=ax)
            ax.set_title('Distribución de muestras por cluster')
            st.pyplot(fig_dist)
        
        # Visualización PCA
        st.subheader("Visualización de clusters (PCA)")
        fig3, ax3 = plt.subplots(figsize=(10,6))
        scatter = ax3.scatter(coords[:,0], coords[:,1], c=labels, cmap='tab20', alpha=0.7, s=50)
        ax3.set_xlabel('Componente Principal 1', fontsize=12)
        ax3.set_ylabel('Componente Principal 2', fontsize=12)
        ax3.set_title('Visualización de clusters mediante PCA', fontsize=14)
        
        # Añadir leyenda para los clusters
        legend_labels = sorted(set(labels))
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=scatter.cmap(scatter.norm(label)), 
                             markersize=10, label=f'Cluster {label}' if label != -1 else 'Ruido')
                  for label in legend_labels]
        ax3.legend(handles=handles, loc='upper right', title='Clusters')
        
        st.pyplot(fig3)
        
        # Análisis de clusters por variables categóricas
        if len(categorical) > 0 and n_clust > 0:
            st.subheader("Análisis de composición de clusters")
            
            # Seleccionar un cluster para análisis detallado
            cluster_to_analyze = st.selectbox(
                "Selecciona un cluster para analizar en detalle",
                options=sorted(list(set(labels))),
                format_func=lambda x: f"Cluster {x}" if x != -1 else "Ruido (Cluster -1)"
            )
            
            # Mostrar composición del cluster seleccionado
            st.write(f"### Composición del Cluster {cluster_to_analyze}")
            
            cluster_data = df_res[df_res['Cluster'] == cluster_to_analyze]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Mostrar distribución de edad en el cluster
                if 'EDAD' in df_res.columns:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(data=df_res, x='EDAD', hue='Cluster', element='step', 
                                common_norm=False, stat='probability',
                                hue_order=[cluster_to_analyze])
                    plt.title(f'Distribución de edad en Cluster {cluster_to_analyze} vs otros')
                    st.pyplot(fig)
            
            with col2:
                # Mostrar estadísticas numéricas
                if len(numeric) > 0:
                    st.write("Estadísticas numéricas:")
                    st.dataframe(cluster_data[numeric].describe())
            
            # Para cada variable categórica, mostrar su distribución en el cluster
            for cat_col in categorical:
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Calcular proporciones
                prop_data = pd.crosstab(
                    df_res['Cluster'], 
                    df_res[cat_col], 
                    normalize='index'
                ).reset_index()
                
                # Filtrar solo el cluster actual
                if cluster_to_analyze in prop_data['Cluster'].values:
                    cluster_props = prop_data[prop_data['Cluster'] == cluster_to_analyze].melt(
                        id_vars=['Cluster'],
                        var_name=cat_col,
                        value_name='Proporción'
                    )
                    
                    # Ordenar por proporción para mejor visualización
                    cluster_props = cluster_props.sort_values('Proporción', ascending=False)
                    
                    # Gráfico de barras
                    sns.barplot(data=cluster_props, x=cat_col, y='Proporción', ax=ax)
                    plt.title(f'Distribución de {cat_col} en Cluster {cluster_to_analyze}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Mostrar tabla con valores y proporciones
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Valores más frecuentes de {cat_col} en el cluster:")
                        st.dataframe(cluster_data[cat_col].value_counts().head(10))
                    
                    with col2:
                        st.write(f"Proporción respecto al total:")
                        total_counts = df_res[cat_col].value_counts()
                        cluster_counts = cluster_data[cat_col].value_counts()
                        props_df = pd.DataFrame({
                            'En_Cluster': cluster_counts,
                            'Total': total_counts,
                            'Proporción': cluster_counts / total_counts
                        }).sort_values('Proporción', ascending=False)
                        st.dataframe(props_df.head(10))

        # Opción para descargar resultados
        st.download_button(
            label="Descargar datos con etiquetas de cluster",
            data=df_res.to_csv(index=False, sep=';').encode('utf-8'),
            file_name="datos_con_clusters.csv",
            mime="text/csv"
        )

# --------------------
# 10. Explicaciones
# --------------------
with st.expander("¿Qué es DBSCAN?"):
    st.write("""
    ### DBSCAN: Clustering basado en densidad
    
    **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) es un algoritmo que:
    
    - Agrupa puntos que están densamente agrupados
    - Identifica puntos en regiones de baja densidad como ruido
    - No requiere especificar el número de clusters de antemano
    - Puede encontrar clusters de formas arbitrarias
    
    **Conceptos clave:**
    
    - **Puntos núcleo**: Tienen al menos `min_samples` puntos (incluyéndose a sí mismos) dentro de la distancia `eps`
    - **Puntos borde**: Están dentro de la distancia `eps` de un punto núcleo, pero no son núcleos
    - **Puntos ruido**: No son ni núcleo ni borde (identificados con `-1` en las etiquetas)
    
    **Ventajas:**
    - Robusto ante valores atípicos
    - No asume clusters esféricos
    - Un único recorrido de los datos
    
    **Desventajas:**
    - Sensible a los parámetros `eps` y `min_samples`
    - Problemas con clusters de densidades muy diferentes
    - Dificultad con datos de alta dimensionalidad
    """)

with st.expander("Interpretación resultados"):
    st.write("""
    ### Interpretación de resultados de DBSCAN
    
    **Selección de parámetros:**
    - **eps**: Representa la distancia máxima entre dos puntos para considerarlos vecinos
        - Si es demasiado pequeño → muchos clusters pequeños y ruido
        - Si es demasiado grande → pocos clusters grandes, posiblemente un único cluster
        - El "codo" en el gráfico k-distance suele ser un buen punto de partida
    
    - **min_samples**: Número mínimo de puntos en un vecindario para considerar un punto como núcleo
        - Valores más altos → mayor robustez ante ruido
        - Valores más bajos → clusters más pequeños y detallados
        - Regla práctica: 2 × dimensionalidad de los datos
    
    **Métricas de calidad:**
    - **Silhouette score**: Mide qué tan similar es un punto a su propio cluster comparado con otros clusters
        - Rango: [-1, 1]. Valores más altos indican mejor definición de clusters
        - > 0.7: Excelente separación
        - 0.5-0.7: Buena separación
        - 0.25-0.5: Estructura débil
        - < 0.25: No se identificó estructura significativa
    
    **Problemas comunes:**
    - **Un solo cluster**: Disminuye eps y/o aumenta min_samples
    - **Demasiados puntos de ruido**: Aumenta eps y/o disminuye min_samples
    - **Silhouette score bajo**: Los clusters no están bien definidos, prueba diferentes características o parámetros
    """)

with st.expander("Datos categóricos y DBSCAN"):
    st.write("""
    ### Trabajo con datos categóricos en DBSCAN
    
    DBSCAN está diseñado para trabajar con datos numéricos, pero para datos categóricos hay diferentes estrategias:
    
    **1. One-Hot Encoding:**
    - Convierte cada valor categórico en una columna binaria
    - **Ventajas**: Preserva la naturaleza nominal de los datos
    - **Desventajas**: Aumenta la dimensionalidad (curse of dimensionality)
    
    **2. Label Encoding:**
    - Asigna un valor numérico único a cada categoría
    - **Ventajas**: Simple, no aumenta dimensionalidad
    - **Desventajas**: Impone una relación ordinal que podría no existir
    
    **3. Métricas de distancia personalizadas:**
    - Combinan distancias para variables categóricas y numéricas
    - **Ventajas**: Más adecuadas para datos mixtos
    - **Desventajas**: Mayor complejidad computacional
    
    **Recomendaciones para interpretar clusters con datos categóricos:**
    - Analizar la distribución de valores categóricos dentro de cada cluster
    - Identificar qué categorías están sobre/subrepresentadas en cada cluster
    - Considerar el efecto del preprocesamiento en la interpretación final
    - Las variables categóricas con muchos valores únicos pueden "dominar" el clustering
    """)

