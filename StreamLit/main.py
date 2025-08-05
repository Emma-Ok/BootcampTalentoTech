import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import json
import requests
import os
import io
from folium import GeoJson, GeoJsonTooltip
from matplotlib.patches import Patch
import warnings
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

warnings.filterwarnings('ignore')

# Modificar configuraciones de página con más opciones
st.set_page_config(
    page_title="Plataformas Educativas Colombia",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Emma-Ok/BootcampTalentoTech',
        'Report a bug': "https://github.com/Emma-Ok/BootcampTalentoTech/issues",
        'About': "# Análisis de Beneficiarios de Plataformas Educativas\n\nCreado por BootcampTalentoTech"
    }
)

# Añadir CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    /* Personalización de colores y estilos */
    :root {
        --primary-color: #4e89ae;
        --secondary-color: #43658b;
        --text-color: #1e3d59;
        --highlight-color: #ff6e40;
        --background-color: #f5f0e1;
    }
    
    /* Estilo para títulos */
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 700;
        border-bottom: 2px solid var(--highlight-color);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Estilo para métricas */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 15px 10px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
    }
    
    /* Estilo para tablas */
    div[data-testid="stTable"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Estilo para widgets interactivos */
    div.stSelectbox, div.stSlider {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    
    /* Mejora visual de las pestañas */
    button[data-baseweb="tab"] {
        font-weight: bold;
        border-radius: 5px 5px 0 0;
        padding: 10px 15px;
        background-color: rgba(255, 255, 255, 0.9);
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid var(--highlight-color);
        color: var(--text-color);
    }
    
    /* Footer personalizado */
    .footer {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        margin-top: 30px;
        font-size: 0.8em;
        color: #555;
    }
    
    /* Mejora visualización de gráficos */
    .stPlotlyChart {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Animación de carga */
    .stProgress > div > div > div > div {
        background-color: var(--highlight-color);
    }
</style>
""", unsafe_allow_html=True)

# Configuraciones globales de gráficos
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Función para cargar datos
@st.cache_data
def cargar_datos():
    """
    Carga y realiza la limpieza básica de los datos de beneficiarios
    """
    url = "https://raw.githubusercontent.com/Emma-Ok/BootcampTalentoTech/main/beneficiarios.csv"
    df = pd.read_csv(url, delimiter=';', encoding='utf-8-sig')
    
    # Limpieza inicial (para quitar espacios en nombres de columnas)
    df.columns = df.columns.str.strip()
    
    # Eliminar filas con datos faltantes en columnas clave
    df = df.dropna(subset=['EDAD', 'PLATAFORMA_EDUCATIVA'])
    
    # Asegurar que la edad sea tipo entero
    df['EDAD'] = df['EDAD'].astype(int)
    
    # Normalizar plataforma educativa
    df['PLATAFORMA_EDUCATIVA'] = df['PLATAFORMA_EDUCATIVA'].str.strip().str.upper()
    
    # Normalizar nombres de departamentos
    df['DEPARTAMENTO'] = df['DEPARTAMENTO'].str.upper().str.strip()
    
    return df

# Función para crear grupos etarios
def crear_grupos_etarios(df):
    """
    Crea grupos etarios para facilitar el análisis
    """
    # Crear grupos etarios con etiquetas personalizadas
    bins = [0, 25, 35, 45, 55, 100]
    labels = ['18-25', '26-35', '36-45', '46-55', '56+']
    df['GRUPO_ETARIO'] = pd.cut(df['EDAD'], bins=bins, labels=labels, right=True)
    
    return df

# Función para crear gráfico de beneficiarios por departamento
def grafico_departamentos(df):
    """
    Crea un gráfico de barras de beneficiarios por departamento
    """
    # Configuración de estilo
    plt.style.use('classic')
    sns.set(rc={'axes.facecolor':'#f8f9fa', 'figure.facecolor':'white'})
    
    # Datos
    departamentos = df['DEPARTAMENTO'].value_counts().index.tolist()
    cantidades = df['DEPARTAMENTO'].value_counts().values.tolist()
    promedio = np.mean(cantidades)
    max_cantidad = max(cantidades)
    
    # Definir colores por categoría
    color_max = '#FF6B6B'  # Color para el máximo
    color_arriba_prom = '#4C72B0'  # Color para arriba del promedio
    color_abajo_prom = '#55A868'  # Color para abajo del promedio
    
    # Crear lista de colores según categorías
    colores = []
    for cantidad in cantidades:
        if cantidad == max_cantidad:
            colores.append(color_max)
        elif cantidad > promedio:
            colores.append(color_arriba_prom)
        else:
            colores.append(color_abajo_prom)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=departamentos, y=cantidades, palette=colores,
                     edgecolor='black', linewidth=0.8, alpha=0.85, ax=ax)
    
    # --- MEJORAS VISUALES ---
    # Título y etiquetas
    plt.title('Distribución de Beneficiarios por Departamento\n',
              fontsize=16, fontweight='bold', pad=20, color='#333333')
    plt.xlabel('Departamento', fontsize=12, labelpad=10, color='#555555')
    plt.ylabel('Número de Beneficiarios', fontsize=12, labelpad=10, color='#555555')
    
    # Rotación de etiquetas
    plt.xticks(rotation=45, ha='right', fontsize=10, color='#555555')
    plt.yticks(color='#555555')
    
    # Añadir valores en las barras
    for i, v in enumerate(cantidades):
        ax.text(i, v + (0.02*max_cantidad), str(v),
                ha='center',
                fontsize=10,
                fontweight='bold',
                color='#333333')
    
    # Línea horizontal de referencia mejorada
    plt.axhline(y=promedio,
                color='#DE8F05',
                linestyle='--',
                linewidth=2,
                alpha=0.8,
                label=f'Promedio: {int(promedio):,}',
                zorder=0)
    
    # Sombreado del área alrededor del promedio
    plt.fill_between(x=ax.get_xlim(),
                    y1=promedio*0.95,
                    y2=promedio*1.05,
                    color='#DE8F05',
                    alpha=0.1,
                    zorder=-1)
    
    # Leyenda personalizada
    legend_elements = [
        Patch(facecolor=color_max, label='Mayor cantidad'),
        Patch(facecolor=color_arriba_prom, label='Arriba del promedio'),
        Patch(facecolor=color_abajo_prom, label='Abajo del promedio'),
        plt.Line2D([0], [0],
                   color='#DE8F05',
                   linestyle='--',
                   linewidth=2,
                   alpha=0.8,
                   label=f'Promedio: {int(promedio):,}')
    ]
    plt.legend(handles=legend_elements, frameon=True, framealpha=1)
    
    # Ajustar límites del eje Y
    plt.ylim(0, max(cantidades) * 1.18)
    
    # Eliminar bordes innecesarios
    sns.despine(left=True, bottom=True)
    
    # Guardar gráfico en un buffer para mostrarlo en Streamlit
    plt.tight_layout()
    return fig

# Función para crear gráfico de distribución por género
def grafico_genero(df):
    """
    Crea un gráfico de dona mostrando la distribución por género
    """
    # Configuración de estilo profesional
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("pastel")
    
    # Datos
    gender_counts = df['GENERO'].value_counts()
    labels = gender_counts.index.tolist()
    sizes = gender_counts.values.tolist()
    
    # Crear figura con tamaño adecuado
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    
    # Función para formato de porcentaje corregida
    def format_autopct(pct, sizes):
        total = sum(sizes)
        val = int(round(pct*total/100))
        return f'{pct:.1f}%\n({val:,})'
    
    # Gráfico de dona profesional
    plt.pie(sizes,
            labels=labels,
            autopct=lambda p: format_autopct(p, sizes),
            startangle=90,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
            textprops={'fontsize': 12, 'color': 'black', 'fontweight': 'bold'},
            colors=['#3498db', '#e91e63'],
            pctdistance=0.85)
    
    # Añadir círculo central para convertir en dona
    plt.gca().add_artist(plt.Circle((0,0), 0.70, fc='white'))
    
    # Título y anotaciones
    plt.title('Distribución por Género\n',
              fontsize=16, fontweight='bold', pad=20, color='#333333')
    
    # Leyenda profesional
    plt.legend(labels,
               title="Género",
               loc="center left",
               bbox_to_anchor=(1, 0.5),
               frameon=True,
               shadow=True)
    
    # Añadir anotación con total
    total = sum(sizes)
    plt.text(0, 0, f'Total\n{total:,}',
             ha='center',
             va='center',
             fontsize=12,
             fontweight='bold',
             color='black')
    
    # Eliminar eje Y
    plt.ylabel('')
    
    # Guardar gráfico
    plt.tight_layout()
    return fig

# Función para crear histograma de distribución de edades
def grafico_edad(df):
    """
    Crea un histograma de la distribución de edades
    """
    sns.set_style('whitegrid')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        data=df,
        x='EDAD',
        bins=10,
        kde=True,
        color='darkblue',
        edgecolor='black',
        alpha=0.7,
        ax=ax
    )
    
    # Media y mediana
    media = df['EDAD'].mean()
    mediana = df['EDAD'].median()
    plt.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.1f}')
    plt.axvline(mediana, color='orange', linestyle=':', linewidth=2, label=f'Mediana: {mediana:.1f}')
    plt.legend()
    
    # Etiquetas
    plt.title('📊 Distribución de Edades de los Beneficiarios', fontsize=16, fontweight='bold')
    plt.xlabel('Edad', fontsize=12)
    plt.ylabel('Cantidad de personas', fontsize=12)
    
    # Guías y bordes
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine()
    
    plt.tight_layout()
    return fig

# Función para crear gráfico de plataformas educativas
def grafico_plataformas(df):
    """
    Crea un gráfico de barras de las plataformas educativas
    """
    # Limpiar y contar plataformas
    conteo = df['PLATAFORMA_EDUCATIVA'].value_counts()
    total = conteo.sum()
    
    # Crear DataFrame
    df_conteo = conteo.reset_index()
    df_conteo.columns = ['Plataforma', 'Cantidad']
    df_conteo['Porcentaje'] = (df_conteo['Cantidad'] / total * 100).round(1)
    
    # Asignar colores tipo semáforo según % y uso
    colores = []
    max_val = df_conteo['Cantidad'].max()
    
    for _, row in df_conteo.iterrows():
        if row['Cantidad'] == max_val:
            colores.append('#e74c3c')  # rojo
        elif row['Porcentaje'] < 10:
            colores.append('#2ecc71')  # verde
        else:
            colores.append('#f39c12')  # naranja
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = sns.barplot(
        data=df_conteo,
        x='Plataforma',
        y='Cantidad',
        palette=colores,
        edgecolor='black',
        ax=ax
    )
    
    # Etiquetas sobre las barras
    for i, row in df_conteo.iterrows():
        plt.text(
            i,
            row['Cantidad'] + total * 0.01,
            f"{row['Cantidad']} ({row['Porcentaje']}%)",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    # Leyenda manual
    leyenda = [
        Patch(color='#e74c3c', label=' Plataforma más usada'),
        Patch(color='#f39c12', label=' Uso intermedio (≥10%)'),
        Patch(color='#2ecc71', label=' Menos del 10% del total')
    ]
    plt.legend(handles=leyenda, title='Leyenda', loc='upper right')
    
    # Estética
    plt.title('Uso de Plataformas Educativas', fontsize=16, fontweight='bold')
    plt.xlabel('Plataforma', fontsize=12)
    plt.ylabel('Cantidad de Usuarios', fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    sns.despine()
    
    plt.tight_layout()
    return fig

# Función para crear heatmap de plataformas por género
def grafico_plataformas_genero(df):
    """
    Crea un heatmap de uso de plataformas por género
    """
    # Análisis cruzado: Plataforma por género
    cross_gen_plat = pd.crosstab(df['PLATAFORMA_EDUCATIVA'], df['GENERO'])
    
    # Ordenar plataformas por uso total (de mayor a menor)
    orden = cross_gen_plat.sum(axis=1).sort_values(ascending=False).index
    cross_gen_plat_sorted = cross_gen_plat.loc[orden]
    
    # Calcular porcentajes por fila
    cross_pct = cross_gen_plat_sorted.div(cross_gen_plat_sorted.sum(axis=1), axis=0) * 100
    
    # Etiquetas combinadas: cantidad + porcentaje
    labels = cross_gen_plat_sorted.astype(str) + "\n(" + cross_pct.round(1).astype(str) + "%)"
    
    # Configuración general de estilo
    sns.set_style("white")
    sns.set(font_scale=1.0)
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Crear heatmap
    sns.heatmap(
        cross_gen_plat_sorted,
        annot=labels,
        fmt='',
        cmap='rocket_r',
        linewidths=0.6,
        linecolor='white',
        cbar_kws={'label': 'Cantidad de usuarios'},
        ax=ax
    )
    
    # Títulos y etiquetas
    plt.title('Distribución del Uso de Plataformas Educativas por Género',
              fontsize=16, fontweight='bold', loc='left', pad=15)
    plt.xlabel('Género', fontsize=13, fontweight='bold', labelpad=10)
    plt.ylabel('Plataforma Educativa', fontsize=13, fontweight='bold', labelpad=10)
    
    # Ajustes de ticks
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    return fig

# Función para crear heatmap de plataformas por grupo etario
def grafico_plataformas_edad(df):
    """
    Crea un heatmap de uso de plataformas por grupo etario
    """
    # Tabla cruzada
    cross = pd.crosstab(df['PLATAFORMA_EDUCATIVA'], df['GRUPO_ETARIO'])
    
    # Ordenar plataformas de mayor a menor uso total
    platform_order = cross.sum(axis=1).sort_values(ascending=False).index
    cross = cross.loc[platform_order]
    
    # Calcular porcentajes por fila
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    
    # Crear anotaciones con valor + porcentaje
    annot = cross.astype(str) + '\n' + cross_pct.round(1).astype(str) + '%'
    
    # Visualización
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.set_style('whitegrid')
    cmap = sns.light_palette("navy", as_cmap=True)
    
    sns.heatmap(
        cross,
        annot=annot,
        fmt='',
        cmap=cmap,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Cantidad de usuarios'},
        annot_kws={"size": 9},
        ax=ax
    )
    
    plt.title('🧠 Uso de Plataformas por Grupo Etario', fontsize=18, fontweight='bold', loc='left')
    plt.xlabel('Grupo Etario', fontsize=13, fontweight='bold')
    plt.ylabel('Plataforma Educativa (ordenadas de mayor a menor uso)', fontsize=13, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    
    plt.tight_layout()
    return fig

# Función para crear gráfico de barras agrupadas por grupo etario y plataforma
def grafico_grupos_plataformas_barras(df):
    """
    Crea un gráfico de barras agrupadas por grupo etario y plataforma
    """
    # Conteo de combinaciones
    conteo = df.groupby(['GRUPO_ETARIO', 'PLATAFORMA_EDUCATIVA']).size().unstack(fill_value=0)
    
    # Paleta personalizada
    custom_palette = sns.color_palette("Set2", n_colors=conteo.columns.shape[0])
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(14, 7))
    conteo.plot(kind='bar', stacked=False, figsize=(14, 7), color=custom_palette, edgecolor='black', ax=ax)
    
    # Agregar etiquetas de valor
    for bars in ax.containers:
        ax.bar_label(
            bars,
            label_type='edge',
            fontsize=9,
            padding=3,
            color='black',
            weight='bold'
        )
    
    # Estética
    plt.title('🎓 Distribución de Plataformas por Grupo Etario', fontsize=16, fontweight='bold')
    plt.ylabel('Número de Beneficiarios', fontsize=12)
    plt.xlabel('Grupo Etario', fontsize=12)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Plataforma Educativa', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title_fontsize=10)
    
    plt.tight_layout()
    return fig

# Función para analizar diversidad de plataformas por departamento
def analisis_diversidad_departamentos(df):
    """
    Analiza la diversidad de plataformas por departamento
    """
    # Seleccionar top 10 departamentos
    top_10_deptos = df['DEPARTAMENTO'].value_counts().head(10).index
    df_top_deptos = df[df['DEPARTAMENTO'].isin(top_10_deptos)]
    
    # Calcular índice de diversidad
    diversity_data = []
    for dept in df_top_deptos['DEPARTAMENTO'].unique():
        dept_data = df_top_deptos[df_top_deptos['DEPARTAMENTO'] == dept]
        platform_counts = dept_data['PLATAFORMA_EDUCATIVA'].value_counts()
        # Calcular índice de Simpson (diversidad)
        total = len(dept_data)
        simpson_index = sum([(count/total)**2 for count in platform_counts])
        diversity_index = 1 - simpson_index  # Más alto = más diverso
        diversity_data.append({'Departamento': dept, 'Diversidad': diversity_index})

    diversity_df = pd.DataFrame(diversity_data).sort_values('Diversidad')
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=diversity_df, x='Diversidad', y='Departamento', palette="viridis", ax=ax)
    plt.title('Índice de Diversidad de Plataformas por Departamento', fontsize=14, fontweight='bold')
    plt.xlabel('Índice de Diversidad (0-1)', fontsize=12)
    
    plt.tight_layout()
    return fig, diversity_df

# Función para crear mapas interactivos - corregida para evitar error de pickle
def crear_mapa_beneficiarios(df):
    """
    Crea mapas interactivos con Folium mostrando la distribución por departamento
    """
    # Análisis detallado por departamento y plataforma
    conteo_plataformas = df.groupby(['DEPARTAMENTO', 'PLATAFORMA_EDUCATIVA']).size().reset_index(name='cantidad')
    
    # Crear matriz de distribución por departamento
    distribucion_matriz = conteo_plataformas.pivot(index='DEPARTAMENTO', columns='PLATAFORMA_EDUCATIVA', values='cantidad').fillna(0)
    distribucion_matriz['TOTAL'] = distribucion_matriz.sum(axis=1)
    
    # Calcular porcentajes por departamento
    distribucion_porcentajes = distribucion_matriz.div(distribucion_matriz['TOTAL'], axis=0) * 100
    distribucion_porcentajes = distribucion_porcentajes.drop('TOTAL', axis=1)
    
    # Colores más claros y visibles
    colores_claros = {
        'COURSERA': '#FFA500',      # Naranja brillante
        'PLATZI': '#32CD32',        # Verde lima
        'DATACAMP': '#4169E1',      # Azul real
        'MICROSOFT AZURE': '#FF6347', # Tomate
        'EDX': '#9370DB',           # Violeta medio
        'UDACITY': '#CD853F'        # Marrón arena
    }
    
    # Función para asignar color según plataforma dominante
    def crear_color_dominante(departamento):
        """Asigna color de la plataforma dominante"""
        if departamento not in distribucion_porcentajes.index:
            return '#f0f0f0'  # Gris muy claro
            
        dept_data = distribucion_porcentajes.loc[departamento]
        plataforma_dominante = dept_data.idxmax()
        porcentaje_dominante = dept_data.max()
        
        color_base = colores_claros.get(plataforma_dominante, '#808080')
        
        # Hacer el color más claro si la dominancia no es muy alta
        if porcentaje_dominante < 50:
            return f"{color_base}80"  # Añadir transparencia
        else:
            return color_base
    
    # Función para tooltip
    def crear_resumen_tooltip(departamento):
        """Crea un resumen conciso para el tooltip"""
        if departamento not in distribucion_matriz.index:
            return "Sin datos", 0
            
        total = int(distribucion_matriz.loc[departamento, 'TOTAL'])
        dept_data = distribucion_porcentajes.loc[departamento]
        
        # Solo top 3 plataformas para tooltip
        top_3 = dept_data.nlargest(3)
        
        resumen_parts = []
        for plataforma, porcentaje in top_3.items():
            if porcentaje >= 5:  # Solo mostrar si tiene al menos 5%
                resumen_parts.append(f"{plataforma}: {porcentaje:.0f}%")
                
        resumen = " | ".join(resumen_parts[:2])  # Solo top 2 para ser más corto
        if len(top_3) > 2 and top_3.iloc[2] >= 5:
            resumen += " | +"
            
        return resumen, total
    
    try:
        # Cargar GeoJSON
        url_geojson = "https://raw.githubusercontent.com/Emma-Ok/BootcampTalentoTech/main/colombia.geo.json"
        response = requests.get(url_geojson)
        geojson_data = response.json()
        
        # Crear mapa con colores dominantes
        mapa_dominante = folium.Map(location=[4.5709, -74.2973], zoom_start=6)
        
        # Clonar GeoJSON para no modificar el original
        geojson_dominante = json.loads(json.dumps(geojson_data))
        
        for feature in geojson_dominante['features']:
            nombre_dpto = feature['properties']['NOMBRE_DPT'].upper()
            if nombre_dpto == 'SANTAFE DE BOGOTA D.C':
                nombre_dpto = 'BOGOTA'
                
            resumen, total = crear_resumen_tooltip(nombre_dpto)
            
            feature['properties']['total_participantes'] = total
            feature['properties']['distribucion_resumen'] = resumen
            feature['properties']['color_dominante'] = crear_color_dominante(nombre_dpto)
            
            # Información de la plataforma dominante
            if nombre_dpto in distribucion_porcentajes.index:
                dept_data = distribucion_porcentajes.loc[nombre_dpto]
                feature['properties']['plataforma_principal'] = dept_data.idxmax()
                feature['properties']['porcentaje_principal'] = f"{dept_data.max():.0f}%"
            else:
                feature['properties']['plataforma_principal'] = 'Sin datos'
                feature['properties']['porcentaje_principal'] = '0%'
        
        # Función de estilo para evitar error de pickle con lambda
        def style_function(feature):
            return {
                'fillColor': feature['properties']['color_dominante'],
                'color': '#333333',
                'weight': 1.5,
                'fillOpacity': 0.8
            }
        
        # Agregar GeoJSON al mapa
        GeoJson(
            geojson_dominante,
            style_function=style_function,
            tooltip=GeoJsonTooltip(
                fields=['NOMBRE_DPT', 'total_participantes', 'plataforma_principal', 'porcentaje_principal'],
                aliases=['Departamento:', 'Total beneficiarios:', 'Plataforma principal:', 'Porcentaje:'],
                localize=True,
                style=("background-color: white; color: #333333; font-family: arial; font-size: 11px; padding: 6px; border-radius: 3px; max-width: 250px;")
            )
        ).add_to(mapa_dominante)
        
        # Leyenda para el mapa
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 280px; height: 200px;
                    background-color: white; border:2px solid #333; z-index:9999;
                    font-size:12px; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);">
        <p><b>🎓 Plataforma Dominante por Departamento</b></p>
        <p style="font-size:10px; margin-bottom:8px; color:#666;">Color = Plataforma con mayor participación</p>
        '''
        
        # Añadir cada plataforma con su color a la leyenda
        for plat, color in colores_claros.items():
            legend_html += f'<p><span style="color:{color}; font-size:16px;">⬤</span> {plat}</p>'
            
        legend_html += f'''
        <hr style="margin:8px 0;">
        <p style="font-size:10px; color:#666;">Total: {len(df):,} beneficiarios</p>
        </div>
        '''
        
        mapa_dominante.get_root().html.add_child(folium.Element(legend_html))
        
        # Devolvemos el mapa y las matrices para su uso posterior, pero no como un objeto cacheable
        return mapa_dominante, distribucion_matriz, distribucion_porcentajes
        
    except Exception as e:
        st.error(f"Error al crear el mapa: {e}")
        return None, None, None

# Función para crear mapa proporcional - corregida para evitar error de pickle
def crear_mapa_proporcional(df):
    """
    Crea un mapa proporcional de beneficiarios
    """
    # 1. Análisis detallado por departamento y plataforma
    conteo_plataformas = df.groupby(['DEPARTAMENTO', 'PLATAFORMA_EDUCATIVA']).size().reset_index(name='cantidad')
    df_departamentos = df.groupby('DEPARTAMENTO').size().reset_index(name='total_participantes')

    # 2. Crear matriz de distribución por departamento
    distribucion_matriz = conteo_plataformas.pivot(index='DEPARTAMENTO', columns='PLATAFORMA_EDUCATIVA', values='cantidad').fillna(0)
    distribucion_matriz['TOTAL'] = distribucion_matriz.sum(axis=1)

    # Calcular porcentajes por departamento
    distribucion_porcentajes = distribucion_matriz.div(distribucion_matriz['TOTAL'], axis=0) * 100
    distribucion_porcentajes = distribucion_porcentajes.drop('TOTAL', axis=1)

    # 3. Colores más claros y visibles
    colores_claros = {
        'COURSERA': '#FFA500',      # Naranja brillante
        'PLATZI': '#32CD32',        # Verde lima
        'DATACAMP': '#4169E1',      # Azul real
        'MICROSOFT AZURE': '#FF6347', # Tomate
        'EDX': '#9370DB',           # Violeta medio
        'UDACITY': '#CD853F'        # Marrón arena
    }

    # 5. Función para crear color sólido basado en la plataforma dominante (más claro)
    def crear_color_dominante_claro(departamento):
        """
        Asigna color de la plataforma dominante pero más claro y visible
        """
        if departamento not in distribucion_porcentajes.index:
            return '#f0f0f0'  # Gris muy claro

        dept_data = distribucion_porcentajes.loc[departamento]
        plataforma_dominante = dept_data.idxmax()
        porcentaje_dominante = dept_data.max()

        color_base = colores_claros.get(plataforma_dominante, '#808080')

        # Hacer el color más claro si la dominancia no es muy alta
        if porcentaje_dominante < 50:
            # Mezclar con blanco para hacer más claro
            return f"{color_base}80"  # Añadir transparencia
        else:
            return color_base

    try:
        # 7. Cargar GeoJSON
        url_geojson = "https://raw.githubusercontent.com/Emma-Ok/BootcampTalentoTech/main/colombia.geo.json"
        response = requests.get(url_geojson)
        geojson_data = response.json()

        # MAPA 2: GRÁFICOS DE BARRAS SUPERPUESTOS
        mapa_barras = folium.Map(location=[4.5709, -74.2973], zoom_start=6)

        # Función de estilo para evitar error de pickle con lambda
        def style_function_base(feature):
            return {
                'fillColor': '#f8f8f8',  # Gris muy claro como base
                'color': '#666666',
                'weight': 1,
                'fillOpacity': 0.3
            }

        # Agregar el mapa base con color neutral
        GeoJson(
            geojson_data,
            style_function=style_function_base
        ).add_to(mapa_barras)

        # Agregar marcadores con gráficos de barras para departamentos principales
        top_departamentos = distribucion_matriz.nlargest(15, 'TOTAL')  # Top 15 departamentos

        # Coordenadas aproximadas de capitales de departamento (para centrar los gráficos)
        coordenadas_deptos = {
            'BOGOTA': [4.7110, -74.0721],
            'ANTIOQUIA': [6.2442, -75.5812],
            'VALLE DEL CAUCA': [3.4516, -76.5320],
            'CUNDINAMARCA': [4.7110, -74.0721],
            'SANTANDER': [7.1193, -73.1227],
            'ATLANTICO': [10.9639, -74.7964],
            'CALDAS': [5.0703, -75.5138],
            'RISARALDA': [4.8087, -75.6906],
            'BOLIVAR': [10.3932, -75.4832],
            'CAUCA': [2.4448, -76.6147],
            'NORTE DE SANTANDER': [7.8939, -72.5078],
            'HUILA': [1.9344, -75.5277],
            'TOLIMA': [4.4389, -75.2322],
            'BOYACA': [5.4539, -73.3616],
            'NARIÑO': [1.2136, -77.2811]
        }

        for dept in top_departamentos.index[:10]:  # Solo top 10 para no saturar
            if dept in coordenadas_deptos:
                coords = coordenadas_deptos[dept]
                dept_data = distribucion_porcentajes.loc[dept]
                total_users = int(distribucion_matriz.loc[dept, 'TOTAL'])

                # Crear HTML para gráfico de barras mini
                top_3_plat = dept_data.nlargest(3)

                barras_html = f"""
                <div style="background: white; padding: 8px; border-radius: 5px; border: 2px solid #333; min-width: 120px;">
                    <b style="font-size: 10px;">{dept}</b><br>
                    <span style="font-size: 9px; color: #666;">{total_users} usuarios</span><br>
                """

                for plataforma, porcentaje in top_3_plat.items():
                    if porcentaje >= 3:
                        color = colores_claros.get(plataforma, '#808080')
                        ancho = int(porcentaje * 0.8)  # Escalar para que quepa
                        barras_html += f"""
                        <div style="margin: 2px 0;">
                            <span style="font-size: 8px; display: inline-block; width: 40px;">{plataforma[:4]}</span>
                            <div style="display: inline-block; background: {color}; width: {ancho}px; height: 8px; margin-left: 2px;"></div>
                            <span style="font-size: 8px; margin-left: 2px;">{porcentaje:.0f}%</span>
                        </div>
                        """

                barras_html += "</div>"
                
                # HTML para el div icon
                div_html = f"""<div style="background: rgba(255,255,255,0.9); border: 1px solid #333; border-radius: 3px; padding: 2px; font-size: 8px; font-weight: bold; text-align: center;">{dept[:8]}<br>{total_users}</div>"""

                # Agregar marcador con el gráfico
                folium.Marker(
                    location=coords,
                    popup=folium.Popup(barras_html, max_width=200),
                    icon=folium.DivIcon(
                        html=div_html,
                        icon_size=(60, 25),
                        icon_anchor=(30, 12)
                    )
                ).add_to(mapa_barras)

        # Leyendas mejoradas
        legend_barras = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 250px; height: 160px;
                    background-color: white; border:2px solid #333; z-index:9999;
                    font-size:12px; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);">
        <p><b>📊 Distribución Detallada</b></p>
        <p style="font-size:10px; margin-bottom:8px; color:#666;">Haz clic en los marcadores para ver gráficos de barras</p>
        <p style="font-size:10px; color:#666;">• Marcadores = Top 10 departamentos</p>
        <p style="font-size:10px; color:#666;">• Barras = Proporción por plataforma</p>
        <p style="font-size:10px; color:#666;">• Números = Total de usuarios</p>
        </div>
        '''

        mapa_barras.get_root().html.add_child(folium.Element(legend_barras))
        return mapa_barras, top_departamentos

    except Exception as e:
        st.error(f"Error al crear el mapa proporcional: {e}")
        return None, None

# Eliminar el decorador de caché de estas funciones
crear_mapa_beneficiarios = crear_mapa_beneficiarios
crear_mapa_proporcional = crear_mapa_proporcional

# --- CONFIGURACIÓN INICIAL ---

# Cargar datos
df = cargar_datos()
df = crear_grupos_etarios(df)

# Inicializar session state
if 'df' not in st.session_state:
    st.session_state.df = df

# Agregar sidebar para filtros globales
st.sidebar.header("🔧 Filtros Globales")

# Filtro por departamento
departamentos_disponibles = ['Todos'] + sorted(df['DEPARTAMENTO'].unique().tolist())
depto_seleccionado = st.sidebar.selectbox(
    "Seleccionar Departamento:",
    departamentos_disponibles
)

# Filtro por plataforma
plataformas_disponibles = ['Todas'] + sorted(df['PLATAFORMA_EDUCATIVA'].unique().tolist())
plataforma_seleccionada = st.sidebar.selectbox(
    "Seleccionar Plataforma:",
    plataformas_disponibles
)

# Filtro por rango de edad
edad_min, edad_max = st.sidebar.slider(
    "Rango de Edad:",
    min_value=int(df['EDAD'].min()),
    max_value=int(df['EDAD'].max()),
    value=(int(df['EDAD'].min()), int(df['EDAD'].max()))
)

# Aplicar filtros
df_filtrado = df.copy()

if depto_seleccionado != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['DEPARTAMENTO'] == depto_seleccionado]

if plataforma_seleccionada != 'Todas':
    df_filtrado = df_filtrado[df_filtrado['PLATAFORMA_EDUCATIVA'] == plataforma_seleccionada]

df_filtrado = df_filtrado[(df_filtrado['EDAD'] >= edad_min) & (df_filtrado['EDAD'] <= edad_max)]

# Actualizar session state
st.session_state.df = df_filtrado

# Mostrar información de filtros aplicados
if len(df_filtrado) != len(df):
    st.sidebar.success(f"Filtros aplicados: {len(df_filtrado):,} de {len(df):,} registros")

# --- INTERFAZ DE USUARIO ---

# Título y descripción
st.title("🎓 Análisis de Beneficiarios de Plataformas Educativas")
st.markdown("""
Esta aplicación muestra un análisis detallado sobre el uso de plataformas educativas por parte de beneficiarios en Colombia.
Explore las diferentes secciones para conocer la distribución por departamento, demografía y preferencias de plataformas.
""")

# Crear estructura de navegación con pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard General", 
    "🌎 Análisis Geográfico", 
    "👥 Análisis Demográfico",
    "🔍 Análisis Detallado",
    "📝 Datos"
])

# --- PESTAÑA 1: DASHBOARD GENERAL ---
with tab1:
    # Agregar banner principal con gradiente
    st.markdown("""
    <div style="background: linear-gradient(to right, #4e89ae, #43658b);
                color: white;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
        <h1 style="color: white; border-bottom: none; margin-bottom: 10px;">
            🎓 Dashboard de Plataformas Educativas
        </h1>
        <p style="font-size: 1.2em;">
            Análisis interactivo de beneficiarios y uso de plataformas educativas en Colombia
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Estadísticas resumidas con progreso
    st.header("📊 Métricas Clave")
    
    # Usar los datos filtrados
    df_actual = st.session_state.df
    
    # Crear fila de métricas con delta values para comparaciones
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_beneficiarios = len(df_actual)
        delta_valor = round((total_beneficiarios / len(df) - 1) * 100, 1) if len(df) > 0 else 0
        st.metric(
            "Total Beneficiarios", 
            f"{total_beneficiarios:,}",
            delta=f"{delta_valor}% del total" if delta_valor != 0 else None,
            delta_color="off" if delta_valor < 0 else "normal"
        )
    
    with col2:
        edad_promedio = df_actual['EDAD'].mean() if len(df_actual) > 0 else 0
        edad_total = df['EDAD'].mean()
        delta_edad = round(edad_promedio - edad_total, 1)
        st.metric(
            "Edad Promedio", 
            f"{edad_promedio:.1f} años",
            delta=f"{delta_edad:+.1f} años" if delta_edad != 0 else None
        )
    
    with col3:
        plataformas_count = df_actual['PLATAFORMA_EDUCATIVA'].nunique()
        st.metric(
            "Plataformas Educativas", 
            f"{plataformas_count}"
        )
    
    with col4:
        departamentos_count = df_actual['DEPARTAMENTO'].nunique()
        st.metric(
            "Departamentos", 
            f"{departamentos_count}"
        )
    
    # Verificar si hay datos después de los filtros
    if len(df_actual) == 0:
        st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados. Ajusta los filtros en la barra lateral.")
        st.stop()
    
    # Fila de KPIs adicionales usando st.progress y formato visual
    st.markdown("##### Distribución por Género")
    gender_counts = df_actual['GENERO'].value_counts()
    total = gender_counts.sum()
    
    if total > 0:
        cols = st.columns(len(gender_counts))
        for i, (gender, count) in enumerate(gender_counts.items()):
            percentage = count / total * 100
            with cols[i]:
                st.markdown(f"**{gender}**")
                st.progress(percentage / 100)
                st.markdown(f"{count:,} ({percentage:.1f}%)")

    # Mostrar gráfico más popular
    if len(df_actual) > 0:
        most_popular_platform = df_actual['PLATAFORMA_EDUCATIVA'].value_counts().idxmax()
        most_popular_count = df_actual['PLATAFORMA_EDUCATIVA'].value_counts().max()
        most_popular_pct = most_popular_count / len(df_actual) * 100
        
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 15px 0;">
            <h4 style="color: #1e3d59;">🥇 Plataforma más popular: <span style="color: #ff6e40;">{most_popular_platform}</span></h4>
            <p>Con <b>{most_popular_count:,}</b> usuarios, representando el <b>{most_popular_pct:.1f}%</b> del total</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráficos principales con expanders interactivos
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📊 Distribución por Género", expanded=True):
            st.pyplot(grafico_genero(df_actual))
    
    with col2:
        with st.expander("📈 Distribución por Edad", expanded=True):
            st.pyplot(grafico_edad(df_actual))
    
    col3, col4 = st.columns(2)
    
    with col3:
        with st.expander("🏫 Distribución por Plataforma", expanded=True):
            st.pyplot(grafico_plataformas(df_actual))
    
    with col4:
        with st.expander("🌍 Distribución por Departamento", expanded=True):
            st.pyplot(grafico_departamentos(df_actual))

# --- PESTAÑA 2: ANÁLISIS GEOGRÁFICO ---
with tab2:
    # Mejorar el encabezado con más contexto
    st.markdown("""
    <div style="background: linear-gradient(to right, #43658b, #4e89ae);
                color: white;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;">
        <h2 style="color: white; border-bottom: none; margin: 0;">🌎 Análisis Geográfico</h2>
        <p>Visualiza la distribución de beneficiarios por departamento y las plataformas dominantes en cada región.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Añadir selector de visualización
    viz_type = st.radio(
        "Seleccionar tipo de visualización",
        ["Gráfico de Barras", "Mapa Interactivo: Plataforma Dominante", "Mapa Interactivo: Distribución Proporcional"],
        horizontal=True
    )
    
    df_actual = st.session_state.df
    
    if len(df_actual) == 0:
        st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados.")
        st.stop()
    
    if viz_type == "Gráfico de Barras":
        st.pyplot(grafico_departamentos(df_actual))
        
    elif viz_type == "Mapa Interactivo: Plataforma Dominante":
        # Añadir spinner de carga para el mapa
        with st.spinner('Generando mapa interactivo... Esto puede tomar unos momentos'):
            mapa_dominante, distribucion_matriz, distribucion_porcentajes = crear_mapa_beneficiarios(df_actual)
            if mapa_dominante:
                st.success('¡Mapa generado exitosamente!')
                # Ajustar altura del mapa según el tema seleccionado
                folium_static(mapa_dominante, width=800, height=600)
            else:
                st.error("No se pudo generar el mapa. Intenta con otro filtro.")
    
    elif viz_type == "Mapa Interactivo: Distribución Proporcional":
        with st.spinner('Generando mapa de distribución... Esto puede tomar unos momentos'):
            mapa_barras, top_departamentos = crear_mapa_proporcional(df_actual)
            if mapa_barras:
                st.success('¡Mapa generado exitosamente!')
                folium_static(mapa_barras, width=800, height=600)
            else:
                st.error("No se pudo generar el mapa. Intenta con otro filtro.")
    
    # Top 5 departamentos con mejor presentación
    if viz_type != "Gráfico de Barras" and 'distribucion_matriz' in locals() and distribucion_matriz is not None:
        st.subheader("🏆 Top 5 departamentos con mayor participación")
        
        # Mejorar la presentación de la tabla usando una versión estilizada
        top5 = distribucion_matriz.nlargest(5, 'TOTAL')
        top5_data = []
        
        for dept in top5.index:
            total = int(top5.loc[dept, 'TOTAL'])
            if dept in distribucion_porcentajes.index:
                dept_dist = distribucion_porcentajes.loc[dept]
                principal = dept_dist.idxmax()
                porcentaje = dept_dist.max()
                top5_data.append({
                    "Departamento": dept,
                    "Total Usuarios": f"{total:,}",
                    "Plataforma Principal": principal,
                    "Porcentaje": f"{porcentaje:.1f}%"
                })
        
        if top5_data:
            top5_df = pd.DataFrame(top5_data)
            
            # Estilo mejorado para la tabla
            def highlight_top_row(s):
                return ['background-color: rgba(255, 110, 64, 0.2)' if i == 0 
                        else 'background-color: rgba(78, 137, 174, 0.1)' if i < 3
                        else '' for i, _ in enumerate(s)]
            
            styled_df = top5_df.style.apply(highlight_top_row, axis=0)
            st.dataframe(styled_df, use_container_width=True)

# --- PESTAÑA 3: ANÁLISIS DEMOGRÁFICO ---
with tab3:
    st.header("👥 Análisis Demográfico")
    
    df_actual = st.session_state.df
    
    if len(df_actual) == 0:
        st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de género
        st.subheader("Distribución por género")
        st.pyplot(grafico_genero(df_actual))
    
    with col2:
        # Gráfico de edad
        st.subheader("Distribución por edad")
        st.pyplot(grafico_edad(df_actual))
    
    # Gráficos de plataformas por grupo etario
    st.subheader("Uso de plataformas por grupo etario")
    st.pyplot(grafico_plataformas_edad(df_actual))
    
    # Gráfico de barras agrupadas
    st.subheader("Distribución de plataformas por grupo etario")
    st.pyplot(grafico_grupos_plataformas_barras(df_actual))
    
    # Gráfico de plataformas por género
    st.subheader("Uso de plataformas por género")
    st.pyplot(grafico_plataformas_genero(df_actual))
    
    # Análisis de diversidad
    st.subheader("Diversidad de plataformas por departamento")
    fig_div, diversity_df = analisis_diversidad_departamentos(df_actual)
    st.pyplot(fig_div)

# --- PESTAÑA 4: ANÁLISIS DETALLADO ---
with tab4:
    st.header("🔍 Análisis Detallado")
    
    df_actual = st.session_state.df
    
    if len(df_actual) == 0:
        st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados.")
        st.stop()
    
    # Definir función faltante
    def analisis_distribucion_plataformas(df):
        """
        Gráfico de barras de distribución de plataformas educativas
        """
        conteo = df['PLATAFORMA_EDUCATIVA'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=conteo.index, y=conteo.values, palette="Set2", ax=ax)
        plt.title("Distribución de Plataformas Educativas", fontsize=15, fontweight='bold')
        plt.xlabel("Plataforma Educativa", fontsize=12)
        plt.ylabel("Cantidad de Beneficiarios", fontsize=12)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        return fig

    st.subheader("Análisis de distribución por plataformas")
    st.pyplot(analisis_distribucion_plataformas(df_actual))
    
    st.subheader("Análisis demográfico detallado")
    def analisis_demografico_detallado(df):
        """
        Gráfico de barras de beneficiarios por género y grupo etario
        """
        conteo = df.groupby(['GENERO', 'GRUPO_ETARIO']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        conteo.plot(kind='bar', stacked=True, ax=ax, colormap='Set3', edgecolor='black')
        plt.title("Distribución Demográfica Detallada", fontsize=15, fontweight='bold')
        plt.xlabel("Género", fontsize=12)
        plt.ylabel("Cantidad de Beneficiarios", fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title='Grupo Etario', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig
    st.pyplot(analisis_demografico_detallado(df_actual))
    
    st.subheader("Análisis de patrones y preferencias")
    def analisis_patrones_preferencias(df):
        """
        Gráfico de barras de preferencias de plataformas por departamento
        """
        # Preferencia: plataforma principal por departamento
        principal_por_depto = df.groupby('DEPARTAMENTO')['PLATAFORMA_EDUCATIVA'].agg(lambda x: x.value_counts().idxmax())
        conteo_principal = principal_por_depto.value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=conteo_principal.index, y=conteo_principal.values, palette="Set1", ax=ax)
        plt.title("Plataforma Principal por Departamento", fontsize=15, fontweight='bold')
        plt.xlabel("Plataforma Educativa", fontsize=12)
        plt.ylabel("Cantidad de Departamentos", fontsize=12)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        return fig
    st.pyplot(analisis_patrones_preferencias(df_actual))

# --- PESTAÑA 5: DATOS ---
with tab5:
    # Mejorar presentación de la pestaña de datos
    st.markdown("""
    <div style="background: linear-gradient(to right, #43658b, #4e89ae);
                color: white;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;">
        <h2 style="color: white; border-bottom: none; margin: 0;">📝 Explorador de Datos</h2>
        <p>Visualiza y descarga los datos utilizados en este análisis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Añadir opciones para exploración de datos
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Opciones para mostrar columnas específicas
        all_columns = df_filtrado.columns.tolist()
        selected_columns = st.multiselect(
            "Seleccionar columnas para mostrar",
            options=all_columns,
            default=all_columns
        )
    
    with col2:
        # Número de registros a mostrar
        num_rows = st.number_input(
            "Registros a mostrar",
            min_value=5,
            max_value=min(1000, len(df_filtrado)),
            value=min(50, len(df_filtrado)),
            step=5
        )
    
    # Añadir campo de búsqueda
    search_term = st.text_input("🔍 Buscar", placeholder="Escribe para filtrar datos...")
    
    df_actual = st.session_state.df
    
    # Aplicar filtro de búsqueda si se proporcionó un término
    if search_term:
        filtered_data = df_actual[
            df_actual.astype(str).apply(
                lambda row: row.str.contains(search_term, case=False).any(), 
                axis=1
            )
        ]
        st.write(f"Mostrando {len(filtered_data)} registros que coinciden con '{search_term}'")
    else:
        filtered_data = df_actual
    
    # Mostrar el dataframe con las columnas seleccionadas
    if selected_columns:
        st.dataframe(filtered_data[selected_columns].head(num_rows), use_container_width=True)
    else:
        st.dataframe(filtered_data.head(num_rows), use_container_width=True)
    
    # Opciones de descarga con más formatos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📥 Descargar como CSV",
            data=filtered_data[selected_columns].to_csv(index=False).encode('utf-8'),
            file_name='beneficiarios_plataformas.csv',
            mime='text/csv',
            help="Descarga los datos filtrados en formato CSV"
        )
    
    with col2:
        st.download_button(
            label="📥 Descargar como Excel",
            data=filtered_data[selected_columns].to_csv(index=False).encode('utf-8'),
            file_name='beneficiarios_plataformas.xlsx',
            mime='application/vnd.ms-excel',
            help="Descarga los datos filtrados en formato Excel"
        )
    
    with col3:
        st.download_button(
            label="📥 Descargar como JSON",
            data=filtered_data[selected_columns].to_json(orient='records').encode('utf-8'),
            file_name='beneficiarios_plataformas.json',
            mime='application/json',
            help="Descarga los datos filtrados en formato JSON"
        )

    # Resumen estadístico de los datos seleccionados
    with st.expander("📊 Resumen estadístico de datos"):
        # Seleccionar solo columnas numéricas para el resumen
        numeric_cols = filtered_data.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols and len(filtered_data) > 0:
            st.write(filtered_data[numeric_cols].describe())
        else:
            st.write("No hay columnas numéricas para mostrar estadísticas o no hay datos disponibles")

# -------------------------------
# ⚙️ CARGA Y LIMPIEZA DE DATOS
# -------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("beneficiarios.csv", sep=None, engine="python")
    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()
    return df

df = cargar_datos()

# -------------------------------
# 🧪 PREPROCESAMIENTO
# -------------------------------
X = df[["EDAD", "GENERO", "DEPARTAMENTO"]]
y = df["PLATAFORMA_EDUCATIVA"]

preprocesador = ColumnTransformer([
    ("num", StandardScaler(), ["EDAD"]),
    ("cat", OneHotEncoder(handle_unknown='ignore'), ["GENERO", "DEPARTAMENTO"])
])

X_proc = preprocesador.fit_transform(X)

# -------------------------------
# 🌲 ENTRENAMIENTO DEL MODELO
# -------------------------------
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_proc, y)

# -------------------------------
# 🔍 FUNCIÓN PARA PREDECIR
# -------------------------------
def predecir_plataforma(edad, genero, departamento):
    nuevo = pd.DataFrame([[edad, genero.upper(), departamento.upper()]],
                         columns=["EDAD", "GENERO", "DEPARTAMENTO"])
    nuevo_proc = preprocesador.transform(nuevo)

    prediccion = modelo_rf.predict(nuevo_proc)[0]
    probabilidades = modelo_rf.predict_proba(nuevo_proc)[0]

    plataformas = modelo_rf.classes_
    ranking = pd.DataFrame({
        "PLATAFORMA_EDUCATIVA": plataformas,
        "Probabilidad": probabilidades
    }).sort_values(by="Probabilidad", ascending=False)

    return prediccion, ranking

with tab6:
    st.header("🤖 Recomendador de Plataformas Educativas")

    st.markdown("""
    ### 📌 ¿Qué hace este recomendador inteligente?

    Este módulo utiliza un modelo de Machine Learning llamado **Random Forest**, el cual ha sido entrenado con información real de beneficiarios de plataformas educativas en Colombia.

    El objetivo es **predecir la plataforma educativa más adecuada** para ti según tres características:
    - Tu **edad**
    - Tu **género**
    - Tu **departamento de residencia**

    El modelo analiza patrones complejos entre miles de registros previos y determina cuál es la **plataforma más recomendada para personas con tu perfil**.


    """, unsafe_allow_html=True)

    st.subheader("📥 Ingresa tus datos:")

    edad = st.number_input("Edad", min_value=10, max_value=100, value=25)
    genero = st.selectbox("Género", sorted(df["GENERO"].dropna().unique()))
    departamento = st.selectbox("Departamento", sorted(df["DEPARTAMENTO"].dropna().unique()))

    if st.button("🔍 Recomendar Plataforma", key="recomendar_rf"):
        pred, ranking = predecir_plataforma(edad, genero, departamento)
        porcentaje_pred = ranking.loc[ranking["PLATAFORMA_EDUCATIVA"] == pred, "Probabilidad"].values[0] * 100



        # Mostrar el ranking
        st.subheader("📊 Distribución de Probabilidades")
        st.dataframe(ranking)

        # Gráfico
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=ranking, x="PLATAFORMA_EDUCATIVA", y="Probabilidad", palette="coolwarm", ax=ax)
        for i, row in ranking.iterrows():
            ax.text(i, row["Probabilidad"] + 0.01, f"{row['Probabilidad']:.2f}", ha='center', fontsize=9)
        ax.set_title("Probabilidad por Plataforma")
        ax.set_ylabel("Probabilidad")
        ax.set_xlabel("Plataforma Educativa")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
        
        # Recomendación personalizada
        otros = ranking[ranking["PLATAFORMA_EDUCATIVA"] != pred].copy()
        otros["Probabilidad"] = otros["Probabilidad"] * 100
        
        st.markdown(f"""
        ### 🧠 Recomendación Personalizada  
        Para personas de género **{genero.lower()}**, con **{edad} años**, del departamento de **{departamento.title()}**,  
        la plataforma más recomendada es 👉 **{pred}**,  
        con una probabilidad del **{porcentaje_pred:.1f}%**.
        """)
        
        # Mostrar otras probabilidades en un expander
        with st.expander("📌 Ver probabilidades de todas las plataformas"):
            ranking_copy = ranking.copy()
            ranking_copy["Probabilidad (%)"] = (ranking_copy["Probabilidad"] * 100).round(1)
            st.dataframe(ranking_copy[["PLATAFORMA_EDUCATIVA", "Probabilidad (%)"]])



        st.success(f"🎯 Plataforma recomendada: **{pred}**")

# Añadir footer personalizado
st.markdown("""
<div class="footer">
    <p>Desarrollado por las cabras para el bootcamp de TalentoTech - Análisis de beneficiarios de plataformas educativas en Colombia.</p>
    <p>© 2025 - Creado con 🧠 y ❤️</p>
</div>
""", unsafe_allow_html=True)
