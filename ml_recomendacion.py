import streamlit as st
import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de página
st.set_page_config(page_title="Recomendador de Plataformas", layout="centered")

st.title("🎓 Recomendador de Plataformas Educativas")

# Cargar datos
def cargar_datos():
    df = pd.read_csv("beneficiarios.csv", sep=None, engine="python")
    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()
    return df

df = cargar_datos()

# Preprocesamiento
X = df[["EDAD", "GENERO", "DEPARTAMENTO"]]
y = df["PLATAFORMA_EDUCATIVA"]

preprocesador = ColumnTransformer([
    ("num", StandardScaler(), ["EDAD"]),
    ("cat", OneHotEncoder(), ["GENERO", "DEPARTAMENTO"])
])
X_proc = preprocesador.fit_transform(X)

modelo_knn = NearestNeighbors(n_neighbors=5, metric='cosine')
modelo_knn.fit(X_proc)

# Función de recomendación
def recomendar_plataformas(edad, genero, departamento):
    nuevo = pd.DataFrame([[edad, genero.upper(), departamento.upper()]],
                         columns=["EDAD", "GENERO", "DEPARTAMENTO"])
    nuevo_proc = preprocesador.transform(nuevo)

    distancias, indices = modelo_knn.kneighbors(nuevo_proc)
    plataformas_vecinas = y.iloc[indices[0]]

    ranking = plataformas_vecinas.value_counts().reset_index()
    ranking.columns = ['PLATAFORMA_EDUCATIVA', 'Frecuencia']
    return ranking

# Entrada del usuario
edad = st.number_input("Edad", min_value=10, max_value=100, value=25)
genero = st.selectbox("Género", sorted(df["GENERO"].dropna().unique()))
departamento = st.selectbox("Departamento", sorted(df["DEPARTAMENTO"].dropna().unique()))

# Botón para recomendar
if st.button("Recomendar"):
    ranking = recomendar_plataformas(edad, genero, departamento)

    st.subheader("📊 Plataformas Recomendadas")
    st.dataframe(ranking)

    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=ranking, x="PLATAFORMA_EDUCATIVA", y="Frecuencia", palette="husl", ax=ax)
    ax.set_title("Ranking de Plataformas Recomendadas")
    ax.set_xlabel("Plataforma Educativa")
    ax.set_ylabel("Frecuencia")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(["Te recomendamos estas plataformas"])
    st.pyplot(fig)





