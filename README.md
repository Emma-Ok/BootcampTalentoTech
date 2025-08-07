# 🎓 Análisis de Plataformas Educativas en Colombia

![Bootcamp](https://raw.githubusercontent.com/Emma-Ok/BootcampTalentoTech/main/logoMorado-famitpYi.png)

## 📊 Descripción del Proyecto

Este proyecto presenta un **análisis interactivo** de datos sobre el uso de plataformas educativas por beneficiarios en Colombia, desarrollado como proyecto final para el Bootcamp TalentoTech. La aplicación web permite explorar patrones demográficos, geográficos y preferencias de plataformas educativas mediante visualizaciones dinámicas e interactivas.

### ✨ Características Principales

- **Dashboard interactivo** con métricas clave y visualizaciones personalizables
- **Análisis geográfico** con mapas de calor y distribución de beneficiarios por departamento
- **Análisis demográfico** por edad y género
- **Análisis municipal** detallado con identificación de municipios con alta y baja participación
- **Sistema de recomendación** basado en Machine Learning que sugiere plataformas según perfil demográfico
- **Explorador de datos** con funcionalidades de filtrado y descarga
- **Análisis de clustering** para identificar patrones en los datos

## 🚀 Demostración

https://bootcamptalentotech-5x2xyljsu6z9u25axmxm8o.streamlit.app/

## 📋 Objetivos del Proyecto

- **Caracterizar** la población beneficiaria por edad, género y ubicación geográfica
- **Identificar** patrones de uso de plataformas educativas
- **Analizar** la distribución geográfica de beneficiarios
- **Desarrollar** un sistema de recomendación personalizado
- **Proporcionar** visualizaciones interactivas para facilitar la toma de decisiones

## 🔧 Tecnologías Utilizadas

- **Python 3.12+** como lenguaje principal
- **Pandas y NumPy** para manipulación y análisis de datos
- **Matplotlib, Seaborn y Plotly** para visualización
- **Folium** para mapas interactivos
- **Scikit-learn** para modelos de Machine Learning
- **Streamlit** para la interfaz web interactiva

## 💻 Instalación y Ejecución

### Requisitos Previos

- Python 3.12 o superior
- pip (gestor de paquetes de Python)

### Pasos para Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/Emma-Ok/BootcampTalentoTech.git
   cd BootcampTalentoTech
   ```

2. **Crear un entorno virtual** (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Linux/MacOS
   venv\Scripts\activate  # En Windows
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r StreamLit/requirements.txt
   ```

4. **Ejecutar la aplicación**:
   ```bash
   cd StreamLit
   streamlit run main.py
   ```

5. **Acceder a la aplicación** en el navegador:
   - La aplicación se abrirá automáticamente en tu navegador predeterminado
   - O navega a `http://localhost:8501`

## 📊 Módulos Principales

### 1. Dashboard General
Proporciona una visión general de las métricas clave y permite comparar plataformas educativas.

### 2. Análisis Geográfico
Visualiza la distribución geográfica de beneficiarios mediante mapas interactivos con diferentes representaciones:
- Mapa de plataformas dominantes
- Distribución proporcional
- Mapa de calor por departamento

### 3. Análisis Demográfico
Explora distribuciones por edad, género y uso de plataformas por grupos demográficos.

### 4. Análisis Municipal
Identifica municipios con alta y baja participación, permitiendo análisis detallados por departamento.

### 5. Sistema de Recomendación
Utiliza un modelo de Random Forest para recomendar plataformas educativas basadas en edad, género y ubicación.

### 6. Explorador de Datos
Permite filtrar, buscar y descargar datos en diferentes formatos.

### 7. Análisis de Clustering
Identifica patrones mediante técnicas de clustering como K-Means y DBSCAN.

## 👥 Equipo

Proyecto desarrollado por el equipo "Las Cabras" como parte del Bootcamp de Analisis de datos
- Emmanuel Valbuena
- Sebastian Boli
- Daniel Jaramillo
- Yoseth Mosquera
- El Ruso

## 🌐 Enlaces

- [Repositorio GitHub](https://github.com/Emma-Ok/BootcampTalentoTech)
- [Documentación de Streamlit](https://docs.streamlit.io/)

---
